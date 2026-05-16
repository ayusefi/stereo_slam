#include "sslam/loop/loop_closing.hpp"

#include "sslam/loop/loop_diagnostics.hpp"
#include "sslam/optim/full_ba.hpp"
#include "sslam/optim/pose_graph.hpp"
#include "sslam/optim/sim3_solver.hpp"
#include "sslam/optim/sim3_opt.hpp"

#include <opencv2/core.hpp>

#include <Eigen/LU>

#include <cmath>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>

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
constexpr int    kMinRansacInliers  = 20;  ///< min RANSAC inliers (coarse gate)
constexpr int    kMinFusedMatches   = 30;  ///< min inliers after SearchByProjection + final Sim3
constexpr int    kMaxLoopCandidatesPerKf = 3;
constexpr double kMinSim3InlierRatio = 0.15;
constexpr double kMinLoopBowScore = 0.08;
constexpr uint64_t kLoopCooldownKfs = 80;
constexpr double kMaxLoopAnyCorrectionM = 35.0;
constexpr double kMaxLoopAdjacentStepM = 12.0;
constexpr double kMaxSim3RmseM = 50.0;

// chi2(0.01, 2) and per-level sigma2 ratio for reprojection thresholds.
constexpr double kChi2      = 9.210;
constexpr double kScaleSq   = 1.2 * 1.2;
// Search radius (px at octave-0) for SearchByProjection after Sim3 opt.
constexpr float  kSBPRadius = 10.0f;
// Max Hamming distance for SearchByProjection descriptor match.
constexpr int    kSBPHamming = 50;

struct LoopCorrespondence {
    int query_idx;
    int match_idx;
};

struct BowMatchCandidate {
    int query_idx;
    int match_idx;
    int distance;
};

struct ScoredLoopCandidate {
    const KeyFrame* kf;
    double score;
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

    std::vector<BowMatchCandidate> candidates;

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
                    candidates.push_back({iq, best_im, best_dist});
                }
            }
            ++it_q; ++it_m;
        } else if (it_q->first < it_m->first) {
            it_q = fv_q.lower_bound(it_m->first);
        } else {
            it_m = fv_m.lower_bound(it_q->first);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const BowMatchCandidate& a, const BowMatchCandidate& b) {
                  return a.distance < b.distance;
              });

    std::vector<std::pair<int,int>> matches;
    matches.reserve(candidates.size());
    std::vector<bool> matched_q(static_cast<std::size_t>(dq.rows), false);
    std::vector<bool> matched_m(static_cast<std::size_t>(dm.rows), false);
    for (const BowMatchCandidate& c : candidates) {
        if (matched_q[static_cast<std::size_t>(c.query_idx)] ||
            matched_m[static_cast<std::size_t>(c.match_idx)])
            continue;
        matched_q[static_cast<std::size_t>(c.query_idx)] = true;
        matched_m[static_cast<std::size_t>(c.match_idx)] = true;
        matches.emplace_back(c.query_idx, c.match_idx);
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
    {
        std::scoped_lock lk(queue_mutex_);
        queue_.clear();
    }
    stop_.store(true);
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void LoopClosing::wait_until_idle() {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    idle_cv_.wait(lk, [this] {
        return queue_.empty() && !processing_.load();
    });
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
    while (true) {
        KeyFrame::Ptr kf;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [&]{ return stop_.load() || !queue_.empty(); });
            if (queue_.empty()) {
                if (stop_.load()) break;
                continue;
            }
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
        idle_cv_.notify_all();
    }
}

// ---------------------------------------------------------------------------

bool LoopClosing::try_close_loop(KeyFrame* q) {
    if (loop_count_.load() > 0 && q->id() < last_loop_kf_id_ + kLoopCooldownKfs)
        return false;

    // Helper: flush one attempt record to the optional logger.
    auto log_attempt = [&](const LoopAttemptStats& s) {
        if (logger_) logger_->record(s);
    };

    // 1. Place recognition with temporal consistency filter.
    const auto candidates = recognizer_->query(q);
    if (candidates.empty()) return false;

    const DBoW2::BowVector q_bow = q->bow();
    std::vector<ScoredLoopCandidate> scored;
    scored.reserve(candidates.size());
    for (const KeyFrame* c : candidates) {
        if (!c || c->is_bad()) continue;
        const double s = vocab_->score(q_bow, c->bow());
        if (s >= kMinLoopBowScore) scored.push_back({c, s});
    }
    if (scored.empty()) return false;
    std::sort(scored.begin(), scored.end(),
              [](const ScoredLoopCandidate& a, const ScoredLoopCandidate& b) {
                  return a.score > b.score;
              });
    if (static_cast<int>(scored.size()) > kMaxLoopCandidatesPerKf)
        scored.resize(static_cast<std::size_t>(kMaxLoopCandidatesPerKf));

    for (const ScoredLoopCandidate& candidate : scored) {
        const KeyFrame* best_match = candidate.kf;
        const double best_score = candidate.score;

        LoopAttemptStats stats;
        stats.query_kf_id      = q->id();
        stats.candidate_kf_id  = best_match->id();
        stats.bow_score        = best_score;

    // 2. Build 3D-3D correspondences via BoW-guided descriptor matching.
    // Prefer MapPoint world positions (BA-corrected); fall back to raw depth
    // back-projection only when no MP is associated.
    const auto feat_matches = match_by_bow(q, best_match);
    stats.bow_matches = static_cast<int>(feat_matches.size());
    if (feat_matches.size() < kMinBowMatches) {
        stats.reject_reason = "insufficient_bow_matches";
        log_attempt(stats);
        continue;
    }

    std::vector<Eigen::Vector3d> pts_q, pts_m;
    std::vector<Eigen::Vector2d> obs_q, obs_m;
    std::vector<double> max_err_q, max_err_m;
    std::vector<LoopCorrespondence> sim3_corrs;
    pts_q.reserve(feat_matches.size());
    pts_m.reserve(feat_matches.size());
    obs_q.reserve(feat_matches.size());
    obs_m.reserve(feat_matches.size());
    max_err_q.reserve(feat_matches.size());
    max_err_m.reserve(feat_matches.size());
    sim3_corrs.reserve(feat_matches.size());

    const std::vector<float>& d_q = q->depth();
    const std::vector<float>& d_m = best_match->depth();
    const auto& kp_q = q->keypoints_left();
    const auto& kp_m = best_match->keypoints_left();
    const StereoCamera& cam = *q->camera();
    const Eigen::Matrix4d T_cw_q = q->get_pose();
    const Eigen::Matrix4d T_cw_m = best_match->get_pose();
    const Eigen::Matrix4d T_wc_q = T_cw_q.inverse();
    const Eigen::Matrix4d T_wc_m = T_cw_m.inverse();

    for (const auto& [iq, im] : feat_matches) {
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
            obs_q.push_back({kp_q[iq].pt.x, kp_q[iq].pt.y});
            obs_m.push_back({kp_m[im].pt.x, kp_m[im].pt.y});
            const double sigma2_q = std::pow(kScaleSq, kp_q[iq].octave);
            const double sigma2_m = std::pow(kScaleSq, kp_m[im].octave);
            max_err_q.push_back(kChi2 * sigma2_q);
            max_err_m.push_back(kChi2 * sigma2_m);
            sim3_corrs.push_back({iq, im});
        }
    }

    if (pts_q.size() < kMinCorrespondences) {
        stats.correspondences_3d = static_cast<int>(pts_q.size());
        stats.reject_reason = "insufficient_correspondences";
        log_attempt(stats);
        continue;
    }
    stats.correspondences_3d = static_cast<int>(pts_q.size());

    // 3. Translational consensus prefilter (TCPF) — runs before Sim3 RANSAC.
    //
    // For stereo SLAM (s≈1, no scale drift), every true correspondence
    // (same physical MapPoint seen from both KFs) satisfies:
    //   d_i = P_q_i - P_m_i  ≈  drift_world   (the accumulated SLAM drift)
    //
    // True correspondences therefore cluster in displacement space near the
    // actual drift vector.  False BoW correspondences (perceptual aliases)
    // scatter at arbitrary displacements, but in structured urban scenes they
    // can form a spurious cluster at some OTHER drift value — typically LARGER
    // than the true drift.
    //
    // Physical bound: the maximum plausible drift grows linearly with the KF
    // span (kTCPFDriftPerKf m per KF).  We seed the RANSAC search only from
    // displacement vectors within this bound, so spurious clusters at larger
    // displacements are never selected.  True-cluster correspondences are kept;
    // everything else is discarded before RANSAC runs.
    //
    // No-filter fallback: if no cluster satisfies the bound, all
    // correspondences are forwarded unchanged (identical to the previous
    // behaviour).
    {
        constexpr double kTCPFRadius     = 5.0;   // cluster radius [m]
        constexpr int    kTCPFMinVotes   = 15;    // min cluster size to apply
        constexpr double kTCPFDriftPerKf = 0.10;  // max drift per KF [m/KF]
        constexpr int    kTCPFIters      = 300;

        const std::size_t n_corr   = pts_q.size();
        const double      max_drift =
            kTCPFDriftPerKf * static_cast<double>(
                q->id() > best_match->id()
                    ? q->id() - best_match->id()
                    : best_match->id() - q->id());

        // Compute per-correspondence displacement vectors.
        std::vector<Eigen::Vector3d> disp(n_corr);
        for (std::size_t i = 0; i < n_corr; ++i)
            disp[i] = pts_q[i] - pts_m[i];

        // 1-point RANSAC: find the best vote-winning cluster whose seed
        // displacement satisfies the drift bound.
        std::mt19937 rng_tcpf(static_cast<uint32_t>(n_corr * 37 + 19));
        std::uniform_int_distribution<std::size_t> dist_tcpf(0, n_corr - 1);

        int             best_votes  = 0;
        Eigen::Vector3d best_center = Eigen::Vector3d::Zero();
        for (int iter = 0; iter < kTCPFIters; ++iter) {
            const Eigen::Vector3d& seed = disp[dist_tcpf(rng_tcpf)];
            if (seed.norm() > max_drift) continue;
            int votes = 0;
            for (std::size_t i = 0; i < n_corr; ++i)
                if ((disp[i] - seed).norm() < kTCPFRadius) ++votes;
            if (votes > best_votes) { best_votes = votes; best_center = seed; }
        }

        if (best_votes >= kTCPFMinVotes) {
            // Refine the cluster center as the mean of its inliers.
            Eigen::Vector3d center = Eigen::Vector3d::Zero();
            int cnt = 0;
            for (std::size_t i = 0; i < n_corr; ++i)
                if ((disp[i] - best_center).norm() < kTCPFRadius)
                    { center += disp[i]; ++cnt; }
            if (cnt > 0) center /= cnt;

            // Apply filter only if the refined center satisfies both the
            // minimum-drift bound (avoids near-zero noise clusters) and the
            // maximum-drift bound.
            if (center.norm() <= max_drift + kTCPFRadius) {
                std::vector<Eigen::Vector3d>    fq, fm;
                std::vector<Eigen::Vector2d>    foq, fom;
                std::vector<double>             feq, fem;
                std::vector<LoopCorrespondence> fc;
                for (std::size_t i = 0; i < n_corr; ++i) {
                    if ((disp[i] - center).norm() < kTCPFRadius) {
                        fq.push_back(pts_q[i]);       fm.push_back(pts_m[i]);
                        foq.push_back(obs_q[i]);      fom.push_back(obs_m[i]);
                        feq.push_back(max_err_q[i]);  fem.push_back(max_err_m[i]);
                        fc.push_back(sim3_corrs[i]);
                    }
                }
                if (fc.size() >= static_cast<std::size_t>(kMinCorrespondences)) {
                    pts_q      = std::move(fq);   pts_m      = std::move(fm);
                    obs_q      = std::move(foq);  obs_m      = std::move(fom);
                    max_err_q  = std::move(feq);  max_err_m  = std::move(fem);
                    sim3_corrs = std::move(fc);
                    stats.correspondences_3d = static_cast<int>(sim3_corrs.size());
                }
            }
        }
        // If no valid cluster: pts_q/pts_m are unchanged → unfiltered fallback.
    }

    // 3a. Sim3 RANSAC — fix_scale=true + bidirectional reprojection inlier test.
    Sim3Solver::Params sim3_params;
    sim3_params.fix_scale   = true;
    sim3_params.min_inliers = kMinRansacInliers;
    sim3_params.max_iterations = 300;
    Sim3Solver solver(pts_q, pts_m,
                      obs_q, obs_m, max_err_q, max_err_m,
                      T_cw_q, T_cw_m,
                      cam.fx, cam.fy, cam.cx, cam.cy,
                      sim3_params);
    auto ransac = solver.solve();

    if (!ransac.found) {
        stats.reject_reason = "sim3_ransac_failed";
        log_attempt(stats);
        continue;
    }
    const double inlier_ratio =
        static_cast<double>(ransac.n_inliers) / static_cast<double>(sim3_corrs.size());
    stats.sim3_inliers      = ransac.n_inliers;
    stats.sim3_inlier_ratio = inlier_ratio;

    // 3b. g2o Sim3 refinement on RANSAC inliers (ORB-SLAM2 OptimizeSim3).
    // sigma2 per correspondence (not chi2-scaled; optimize_sim3 does the scaling).
    std::vector<double> sig2_q(pts_q.size()), sig2_m(pts_q.size());
    for (std::size_t i = 0; i < pts_q.size(); ++i) {
        sig2_q[i] = std::pow(kScaleSq, kp_q[sim3_corrs[i].query_idx].octave);
        sig2_m[i] = std::pow(kScaleSq, kp_m[sim3_corrs[i].match_idx].octave);
    }
    const auto sim3_opt = optim::optimize_sim3(
        pts_q, pts_m, obs_q, obs_m, sig2_q, sig2_m,
        ransac.inlier_mask, T_cw_q, T_cw_m,
        ransac.scale, ransac.R, ransac.t, cam, /*fix_scale=*/true);

    // Use a seed Sim3 for SearchByProjection. Prefer the g2o-refined seed,
    // but if that refinement collapses to too few inliers, keep the RANSAC
    // seed alive and let projection expansion + final OptimizeSim3 decide.
    // This matters for end-of-sequence loops where the initial BoW/RANSAC seed
    // is plausible but the first small inlier set is brittle.
    double            s_ref = sim3_opt.scale;
    Eigen::Matrix3d   R_ref = sim3_opt.R;
    Eigen::Vector3d   t_ref = sim3_opt.t;
    std::vector<bool> seed_mask = sim3_opt.inlier_mask;
    int               seed_inliers = sim3_opt.n_inliers;

    if (sim3_opt.n_inliers < kMinRansacInliers) {
        s_ref = ransac.scale;
        R_ref = ransac.R;
        t_ref = ransac.t;
        seed_mask = ransac.inlier_mask;
        seed_inliers = ransac.n_inliers;
    }

    stats.sim3_inliers = seed_inliers;
    stats.sim3_inlier_ratio =
        static_cast<double>(seed_inliers) / static_cast<double>(sim3_corrs.size());

    // Fill seed Sim3 scale and RMSE on refined inliers.
    stats.sim3_scale = s_ref;
    {
        double sum_sq = 0.0;
        int    n      = 0;
        for (std::size_t i = 0; i < pts_q.size(); ++i) {
            if (i < seed_mask.size() && !seed_mask[i]) continue;
            const Eigen::Vector3d err = s_ref * R_ref * pts_q[i] + t_ref - pts_m[i];
            sum_sq += err.squaredNorm();
            ++n;
        }
        stats.sim3_rmse_m = (n > 0) ? std::sqrt(sum_sq / n) : 0.0;
    }
    if (stats.sim3_rmse_m > kMaxSim3RmseM) {
        stats.reject_reason = "sim3_rmse_too_large";
        log_attempt(stats);
        continue;
    }

    // 3c. SearchByProjection with refined Sim3:
    //     Project match-KF MapPoints into query KF; add new correspondences.
    //     Inverse Sim3: p_q_world = R_ref^T * (p_m_world - t_ref) / s_ref
    //     Then project: p_cam_q = T_cw_q * p_q_world
    {
        const Eigen::Matrix3d R_inv = R_ref.transpose();
        const double          s_inv = 1.0 / s_ref;

        // Build a lookup: query feature index → whether it's already matched.
        std::unordered_set<int> already_matched_q;
        for (const auto& c : sim3_corrs) already_matched_q.insert(c.query_idx);

        const auto& kp_q_all = q->keypoints_left();
        const auto& kp_m_all = const_cast<KeyFrame*>(best_match)->keypoints_left();
        const cv::Mat& dq = q->descriptors_left();
        const cv::Mat& dm = const_cast<KeyFrame*>(best_match)->descriptors_left();
        const std::size_t n_m = best_match->num_features();

        for (std::size_t jm = 0; jm < n_m; ++jm) {
            MapPoint::Ptr mp_jm = const_cast<KeyFrame*>(best_match)->get_map_point(
                static_cast<int>(jm));
            if (!mp_jm || mp_jm->is_bad()) continue;

            // Transform match-world point into query-world.
            const Eigen::Vector3d P_m_world = mp_jm->get_world_pos();
            const Eigen::Vector3d P_q_world = s_inv * R_inv * (P_m_world - t_ref);

            // Project into query camera.
            const Eigen::Vector3d P_cam_q =
                T_cw_q.topLeftCorner<3,3>() * P_q_world + T_cw_q.topRightCorner<3,1>();
            if (P_cam_q.z() <= 0.0) continue;

            const double u = cam.fx * P_cam_q.x() / P_cam_q.z() + cam.cx;
            const double v = cam.fy * P_cam_q.y() / P_cam_q.z() + cam.cy;
            if (u < 0 || u >= cam.width || v < 0 || v >= cam.height) continue;

            const float oct_scale = static_cast<float>(
                std::pow(1.2, kp_m_all[jm].octave));
            const float radius = kSBPRadius * oct_scale;

            // Find closest query feature within the projected window.
            int best_d = kSBPHamming + 1, best_iq = -1;
            const cv::Mat desc_m = dm.row(static_cast<int>(jm));
            for (std::size_t iq = 0; iq < kp_q_all.size(); ++iq) {
                if (already_matched_q.count(static_cast<int>(iq))) continue;
                const cv::KeyPoint& kpq = kp_q_all[iq];
                const float du = kpq.pt.x - static_cast<float>(u);
                const float dv = kpq.pt.y - static_cast<float>(v);
                if (du * du + dv * dv > radius * radius) continue;
                const int d = hamming(dq.row(static_cast<int>(iq)), desc_m);
                if (d < best_d) { best_d = d; best_iq = static_cast<int>(iq); }
            }
            if (best_iq < 0) continue;

            // Add as a new correspondence (3D points from MP and backprojection).
            const auto& kpq = kp_q_all[static_cast<std::size_t>(best_iq)];
            Eigen::Vector3d P_q_w_corr, P_m_w_corr;
            bool ok_q2 = false, ok_m2 = false;

            MapPoint::Ptr mp_q2 = q->get_map_point(best_iq);
            if (mp_q2 && !mp_q2->is_bad()) {
                P_q_w_corr = mp_q2->get_world_pos(); ok_q2 = true;
            } else {
                const float z = q->depth()[static_cast<std::size_t>(best_iq)];
                if (z > 0.0f) {
                    P_q_w_corr = T_wc_q.topLeftCorner<3,3>() *
                        Eigen::Vector3d((kpq.pt.x - cam.cx) * z / cam.fx,
                                        (kpq.pt.y - cam.cy) * z / cam.fy, z)
                        + T_wc_q.topRightCorner<3,1>();
                    ok_q2 = true;
                }
            }
            P_m_w_corr = P_m_world; ok_m2 = true;

            if (ok_q2 && ok_m2) {
                pts_q.push_back(P_q_w_corr);
                pts_m.push_back(P_m_w_corr);
                obs_q.push_back({kpq.pt.x, kpq.pt.y});
                obs_m.push_back({kp_m_all[jm].pt.x, kp_m_all[jm].pt.y});
                sig2_q.push_back(std::pow(kScaleSq, kpq.octave));
                sig2_m.push_back(std::pow(kScaleSq, kp_m_all[jm].octave));
                max_err_q.push_back(kChi2 * sig2_q.back());
                max_err_m.push_back(kChi2 * sig2_m.back());
                sim3_corrs.push_back({best_iq, static_cast<int>(jm)});
                already_matched_q.insert(best_iq);
            }
        }
    }

    // Count total matches with valid 3D points (inliers + SearchByProjection).
    const int n_fused = static_cast<int>(sim3_corrs.size());
    if (n_fused < kMinFusedMatches) {
        stats.reject_reason = "insufficient_fused_matches";
        log_attempt(stats);
        continue;
    }

    // 3d. Re-optimise Sim3 on the expanded match set. Original BoW matches keep
    // the selected seed inlier mask; projection matches seed as inliers.
    std::vector<bool> expanded_mask(sim3_corrs.size(), true);
    for (std::size_t i = 0; i < seed_mask.size() && i < expanded_mask.size(); ++i) {
        expanded_mask[i] = seed_mask[i];
    }

    const auto final_sim3 = optim::optimize_sim3(
        pts_q, pts_m, obs_q, obs_m, sig2_q, sig2_m,
        expanded_mask, T_cw_q, T_cw_m,
        s_ref, R_ref, t_ref, cam, /*fix_scale=*/true);

    stats.sim3_inliers = final_sim3.n_inliers;
    stats.sim3_inlier_ratio =
        static_cast<double>(final_sim3.n_inliers) / static_cast<double>(sim3_corrs.size());

    if (final_sim3.n_inliers < kMinFusedMatches) {
        stats.reject_reason = "sim3_final_insufficient_inliers";
        log_attempt(stats);
        continue;
    }
    if (stats.sim3_inlier_ratio < kMinSim3InlierRatio) {
        stats.reject_reason = "sim3_inlier_ratio_too_low";
        log_attempt(stats);
        continue;
    }

    s_ref = final_sim3.scale;
    R_ref = final_sim3.R;
    t_ref = final_sim3.t;
    stats.sim3_scale = s_ref;
    {
        double sum_sq = 0.0;
        int    n      = 0;
        for (std::size_t i = 0; i < pts_q.size(); ++i) {
            if (i < final_sim3.inlier_mask.size() && !final_sim3.inlier_mask[i]) continue;
            const Eigen::Vector3d err = s_ref * R_ref * pts_q[i] + t_ref - pts_m[i];
            sum_sq += err.squaredNorm();
            ++n;
        }
        stats.sim3_rmse_m = (n > 0) ? std::sqrt(sum_sq / n) : 0.0;
    }
    if (stats.sim3_rmse_m > kMaxSim3RmseM) {
        stats.reject_reason = "sim3_rmse_too_large";
        log_attempt(stats);
        continue;
    }

    // 4. Pause LocalMapping, fuse duplicate MapPoints, run PGO.
    local_mapping_->request_stop();
    local_mapping_->wait_until_stopped();

    std::unique_lock<std::shared_mutex> map_update_lk(map_->update_mutex_);

    const auto pgo_preview = pose_graph::preview(
        *map_, q, const_cast<KeyFrame*>(best_match), s_ref, R_ref, t_ref);

    // Fill topology and correction stats from preview.
    stats.graph_vertices         = pgo_preview.graph_vertices;
    stats.graph_edges            = pgo_preview.graph_edges;
    stats.graph_components       = pgo_preview.graph_components;
    stats.max_pose_correction_m  = pgo_preview.max_center_correction_m;
    stats.max_pose_correction_deg = pgo_preview.max_rotation_correction_deg;
    if (!pgo_preview.valid ||
        pgo_preview.max_center_correction_m > kMaxLoopAnyCorrectionM ||
        pgo_preview.max_adjacent_step_m > kMaxLoopAdjacentStepM) {
        stats.reject_reason = !pgo_preview.valid
            ? (pgo_preview.graph_components > 1 ? "disconnected_graph"
                                                 : "pgo_invalid")
            : (pgo_preview.max_center_correction_m > kMaxLoopAnyCorrectionM
                   ? "pgo_correction_too_large"
                   : "pgo_adjacent_step_too_large");
        std::cerr << "[LC] reject q=" << q->id()
                  << " m=" << best_match->id()
                  << " reason=" << stats.reject_reason
                  << " query_corr=" << pgo_preview.query_center_correction_m
                  << " max_corr=" << pgo_preview.max_center_correction_m
                  << " max_kf=" << pgo_preview.max_center_kf_id
                  << " max_adj_step=" << pgo_preview.max_adjacent_step_m
                  << " max_adj_kf=" << pgo_preview.max_adjacent_step_kf_id << "\n";
        log_attempt(stats);
        local_mapping_->resume();
        continue;
    }

    stats.accepted = true;
    log_attempt(stats);
    std::cerr << "[LC] accept q=" << q->id()
              << " m=" << best_match->id()
              << " ransac_in=" << ransac.n_inliers
              << " opt_in=" << final_sim3.n_inliers
              << " fused=" << n_fused
              << " scale=" << s_ref
              << " score=" << best_score
              << " query_corr=" << pgo_preview.query_center_correction_m
              << " max_corr=" << pgo_preview.max_center_correction_m
              << " max_adj_step=" << pgo_preview.max_adjacent_step_m << "\n";

    // 5. Fuse duplicate MapPoints via MapPoint::Replace.
    {
        for (std::size_t pidx = 0; pidx < sim3_corrs.size(); ++pidx) {
            if (pidx < final_sim3.inlier_mask.size() && !final_sim3.inlier_mask[pidx]) continue;

            const int iq = sim3_corrs[pidx].query_idx;
            const int im = sim3_corrs[pidx].match_idx;
            MapPoint::Ptr mp_q2 = q->get_map_point(iq);
            MapPoint::Ptr mp_m2 = const_cast<KeyFrame*>(best_match)->get_map_point(im);
            if (!mp_q2 || !mp_m2) continue;
            if (mp_q2->is_bad() || mp_m2->is_bad()) continue;
            if (mp_q2.get() == mp_m2.get()) continue;

            // Keep the older MP (lower id), replace the newer one.
            if (mp_q2->id() < mp_m2->id())
                mp_m2->replace(mp_q2);
            else
                mp_q2->replace(mp_m2);
        }
    }

    // 6. Essential-graph PGO — use the g2o-refined Sim3.
    pose_graph::optimize(*map_,
                          q,
                          const_cast<KeyFrame*>(best_match),
                          s_ref, R_ref, t_ref);

    local_mapping_->resume();

    // Full BA is intentionally not started here.  The pose-graph correction
    // is the loop-closing action; a full global BA can be run as an explicit
    // offline refinement later.  Launching it from this thread made shutdown
    // wait on a large async solve and prevented trajectory export.
    last_loop_kf_id_ = q->id();
    ++loop_count_;

    return true;
    }

    return false;
}

}  // namespace sslam
