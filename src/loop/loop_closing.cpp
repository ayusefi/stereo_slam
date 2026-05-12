#include "sslam/loop/loop_closing.hpp"

#include "sslam/optim/full_ba.hpp"
#include "sslam/optim/pose_graph.hpp"
#include "sslam/optim/sim3_solver.hpp"

#include <opencv2/core.hpp>

#include <Eigen/LU>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>

namespace sslam {

namespace {

/// Hamming distance between two 32-byte ORB descriptors.
int hamming(const cv::Mat& a, const cv::Mat& b) {
    int dist = 0;
    const auto* pa = a.ptr<uint8_t>(0);
    const auto* pb = b.ptr<uint8_t>(0);
    for (int i = 0; i < 32; ++i) {
        uint8_t v = pa[i] ^ pb[i];
        // Brian Kernighan bit-count
        while (v) { ++dist; v &= v - 1u; }
    }
    return dist;
}

constexpr int    kTHLow  = 50;
constexpr int    kTHHigh = 100;
constexpr float  kLoweRatio = 0.75f;
constexpr int    kMinBowMatches = 20;  ///< min descriptor matches to attempt Sim3
constexpr int    kMinCorrespondences = 20;  ///< min 3-D pairs for Sim3 RANSAC
constexpr int    kMinSim3Inliers = 40;
constexpr double kMinSim3InlierRatio = 0.25;
constexpr double kMinStereoScale = 0.80;
constexpr double kMaxStereoScale = 1.20;

struct LoopCorrespondence {
    int query_idx;
    int match_idx;
};

/// Match descriptors between two KFs guided by their FeatureVectors.
/// Returns pairs (idx_in_q, idx_in_match) where both features are
/// within the same BoW level-4 node.
std::vector<std::pair<int,int>> match_by_bow(const KeyFrame* q,
                                              const KeyFrame* m)
{
    const DBoW2::FeatureVector fv_q = q->feat_vec();
    const DBoW2::FeatureVector fv_m = m->feat_vec();
    const cv::Mat& dq = q->descriptors_left();
    const cv::Mat& dm = m->descriptors_left();

    std::vector<std::pair<int,int>> matches;

    auto it_q = fv_q.begin();
    auto it_m = fv_m.begin();
    while (it_q != fv_q.end() && it_m != fv_m.end()) {
        if (it_q->first == it_m->first) {
            // Same node — cross-match features.
            for (int iq : it_q->second) {
                int best_dist = kTHHigh, second_dist = kTHHigh, best_im = -1;
                const cv::Mat desc_q = dq.row(iq);
                for (int im : it_m->second) {
                    const int d = hamming(desc_q, dm.row(im));
                    if (d < best_dist) {
                        second_dist = best_dist;
                        best_dist   = d;
                        best_im     = im;
                    } else if (d < second_dist) {
                        second_dist = d;
                    }
                }
                if (best_dist < kTHLow &&
                    static_cast<float>(best_dist) < kLoweRatio * second_dist) {
                    matches.emplace_back(iq, best_im);
                }
            }
            ++it_q; ++it_m;
        } else if (it_q->first < it_m->first) {
            it_q = fv_q.lower_bound(it_m->first);
        } else {
            it_m = fv_m.lower_bound(it_q->first);
        }
    }
    return matches;
}

}  // namespace

// ---------------------------------------------------------------------------

LoopClosing::LoopClosing(Map::Ptr map,
                          LocalMapping::Ptr lm,
                          const ORBVocabulary* vocab,
                          KeyFrameDatabase* db)
    : map_(std::move(map))
    , local_mapping_(std::move(lm))
    , vocab_(vocab)
    , db_(db)
    , recognizer_(std::make_unique<PlaceRecognizer>(*db))
{}

LoopClosing::~LoopClosing() {
    if (thread_.joinable()) shutdown();
}

void LoopClosing::start() {
    thread_ = std::thread(&LoopClosing::run, this);
}

void LoopClosing::shutdown() {
    stop_.store(true);
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void LoopClosing::enqueue_keyframe(KeyFrame::Ptr kf) {
    {
        std::scoped_lock lk(queue_mutex_);
        queue_.push_back(std::move(kf));
    }
    queue_cv_.notify_one();
}

// ---------------------------------------------------------------------------

void LoopClosing::run() {
    while (!stop_.load()) {
        KeyFrame::Ptr kf;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [&]{ return stop_.load() || !queue_.empty(); });
            if (stop_.load()) break;
            kf = std::move(queue_.front());
            queue_.pop_front();
        }

        if (!kf || kf->is_bad()) continue;
        if (!kf->bow_computed()) continue;

        // Index this KF so future queries can match against it.
        db_->add(kf.get());

        processing_.store(true);
        static std::atomic<int> lc_kf_count{0};
        const int kf_idx = ++lc_kf_count;
        if (kf_idx % 50 == 0)
            std::cerr << "[LC] processed " << kf_idx << " KFs\n";
        const bool closed = try_close_loop(kf.get());
        if (closed) std::cerr << "[LC] *** LOOP CLOSED at KF " << kf->id() << " ***\n";
        processing_.store(false);
    }
}

// ---------------------------------------------------------------------------

bool LoopClosing::try_close_loop(KeyFrame* q) {
    // 1. Place recognition with temporal consistency filter.
    const auto candidates = recognizer_->query(q);
    if (candidates.empty()) return false;

    // Take the highest-BoW-score candidate.
    const DBoW2::BowVector q_bow = q->bow();
    const KeyFrame* best_match   = nullptr;
    double          best_score   = 0.0;
    for (const KeyFrame* c : candidates) {
        const double s = vocab_->score(q_bow, c->bow());
        if (s > best_score) { best_score = s; best_match = c; }
    }
    if (!best_match) return false;

    // 2. Build 3D-3D correspondences via BoW-guided descriptor matching.
    // Prefer MapPoint world positions (BA-corrected); fall back to raw depth
    // back-projection only when no MP is associated.
    const auto feat_matches = match_by_bow(q, best_match);
    if (feat_matches.size() < kMinBowMatches) return false;

    std::vector<Eigen::Vector3d> pts_q, pts_m;
    std::vector<LoopCorrespondence> sim3_corrs;
    pts_q.reserve(feat_matches.size());
    pts_m.reserve(feat_matches.size());
    sim3_corrs.reserve(feat_matches.size());

    const std::vector<float>& d_q = q->depth();
    const std::vector<float>& d_m = best_match->depth();
    const auto& kp_q = q->keypoints_left();
    const auto& kp_m = best_match->keypoints_left();
    const StereoCamera& cam = *q->camera();
    const Eigen::Matrix4d T_wc_q = q->get_pose().inverse();
    const Eigen::Matrix4d T_wc_m = best_match->get_pose().inverse();

    for (const auto& [iq, im] : feat_matches) {
        // Get BA-refined world positions (MP preferred, depth fallback).
        Eigen::Vector3d P_q_w, P_m_w;
        bool ok_q = false, ok_m = false;

        MapPoint::Ptr mp_q = q->get_map_point(iq);
        if (mp_q && !mp_q->is_bad()) {
            P_q_w = mp_q->get_world_pos();
            ok_q  = true;
        } else if (d_q[iq] > 0.0f) {
            const float z = d_q[iq];
            P_q_w = T_wc_q.topLeftCorner<3,3>() *
                    Eigen::Vector3d((kp_q[iq].pt.x - cam.cx) * z / cam.fx,
                                    (kp_q[iq].pt.y - cam.cy) * z / cam.fy, z)
                    + T_wc_q.topRightCorner<3,1>();
            ok_q = true;
        }

        MapPoint::Ptr mp_m = const_cast<KeyFrame*>(best_match)->get_map_point(im);
        if (mp_m && !mp_m->is_bad()) {
            P_m_w = mp_m->get_world_pos();
            ok_m  = true;
        } else if (d_m[im] > 0.0f) {
            const float z = d_m[im];
            P_m_w = T_wc_m.topLeftCorner<3,3>() *
                    Eigen::Vector3d((kp_m[im].pt.x - cam.cx) * z / cam.fx,
                                    (kp_m[im].pt.y - cam.cy) * z / cam.fy, z)
                    + T_wc_m.topRightCorner<3,1>();
            ok_m = true;
        }

        if (ok_q && ok_m) {
            pts_q.push_back(P_q_w);
            pts_m.push_back(P_m_w);
            sim3_corrs.push_back({iq, im});
        }
    }

    if (pts_q.size() < kMinCorrespondences) return false;

    // 3. Sim3 RANSAC.
    Sim3Solver solver(pts_q, pts_m);
    const auto sim3 = solver.solve();
    if (!sim3.found) return false;
    const double inlier_ratio =
        static_cast<double>(sim3.n_inliers) / static_cast<double>(sim3_corrs.size());
    if (sim3.n_inliers < kMinSim3Inliers ||
        inlier_ratio < kMinSim3InlierRatio ||
        sim3.scale < kMinStereoScale || sim3.scale > kMaxStereoScale) {
        return false;
    }

    // 4. Pause LocalMapping, fuse duplicate MapPoints, run PGO.
    local_mapping_->request_stop();
    local_mapping_->wait_until_stopped();
    if (stop_.load()) { local_mapping_->resume(); return false; }

    // 5. Fuse duplicate MapPoints.
    // For each inlier pair, if both feature indices have MPs, keep the older
    // one (lower id) and redirect the other's observations.
    {
        for (std::size_t pidx = 0; pidx < sim3_corrs.size(); ++pidx) {
            if (!sim3.inlier_mask[pidx]) continue;

            const int iq = sim3_corrs[pidx].query_idx;
            const int im = sim3_corrs[pidx].match_idx;
            MapPoint::Ptr mp_q = q->get_map_point(iq);
            MapPoint::Ptr mp_m = const_cast<KeyFrame*>(best_match)->get_map_point(im);
            if (!mp_q || !mp_m) continue;
            if (mp_q->is_bad() || mp_m->is_bad()) continue;
            if (mp_q.get() == mp_m.get()) continue;

            // Keep the MP with lower id (older, more observations).
            MapPoint::Ptr keep   = (mp_q->id() < mp_m->id()) ? mp_q : mp_m;
            MapPoint::Ptr remove = (mp_q->id() < mp_m->id()) ? mp_m : mp_q;

            // Redirect all observations of 'remove' to 'keep'.
            for (const auto& [kf_obs, fidx] : remove->get_observations()) {
                if (!keep->get_observations().count(kf_obs)) {
                    kf_obs->add_map_point(fidx, keep);
                    keep->add_observation(kf_obs, fidx);
                } else {
                    kf_obs->erase_map_point(fidx);
                }
                remove->remove_observation(kf_obs);
            }
            remove->set_bad();
            keep->compute_descriptor();
        }
    }

    // 6. Essential-graph PGO.
    pose_graph::optimize(*map_,
                          q,
                          const_cast<KeyFrame*>(best_match),
                          sim3.scale, sim3.R, sim3.t);

    local_mapping_->resume();

    // 7. Trigger full BA in the background.
    if (!full_ba_) full_ba_ = std::make_shared<FullBA>(map_);
    full_ba_->trigger();
    ++loop_count_;

    return true;
}

}  // namespace sslam
