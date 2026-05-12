#include "sslam/mapping/local_mapping.hpp"

#include "sslam/loop/loop_closing.hpp"
#include "sslam/mapping/triangulation.hpp"

#include <algorithm>
#include <chrono>
#include <unordered_set>
#include <vector>

namespace sslam {

LocalMapping::LocalMapping(Map::Ptr map,
                           std::shared_ptr<const StereoCamera> cam)
    : map_(std::move(map)), cam_(std::move(cam)) {}

LocalMapping::~LocalMapping() { shutdown(); }

void LocalMapping::start() {
    thread_ = std::thread(&LocalMapping::run, this);
}

void LocalMapping::shutdown() {
    {
        // Drain the pending queue so the thread exits quickly instead of
        // processing a backlog of stale KFs.
        std::scoped_lock lk(queue_mutex_);
        queue_.clear();
    }
    stop_ = true;
    queue_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void LocalMapping::enqueue_keyframe(KeyFrame::Ptr kf) {
    {
        std::scoped_lock lk(queue_mutex_);
        queue_.push_back(std::move(kf));
    }
    queue_cv_.notify_one();
}

bool LocalMapping::is_idle() const {
    std::scoped_lock lk(queue_mutex_);
    return queue_.empty() && !processing_;
}

void LocalMapping::wait_until_idle() {
    std::unique_lock lk(queue_mutex_);
    idle_cv_.wait(lk, [this] {
        return queue_.empty() && !processing_;
    });
}

LocalMapping::BaStats LocalMapping::ba_stats() const {
    std::scoped_lock lk(stats_mutex_);
    return ba_stats_;
}

void LocalMapping::request_stop() { stop_requested_ = true; queue_cv_.notify_all(); }
void LocalMapping::resume()       { stop_requested_ = false; stopped_ = false; queue_cv_.notify_all(); }
bool LocalMapping::is_stopped() const { return stopped_; }

void LocalMapping::wait_until_stopped() {
    std::unique_lock lk(queue_mutex_);
    idle_cv_.wait(lk, [this] { return stopped_.load() || stop_.load(); });
}

// ---------------------------------------------------------------------------

void LocalMapping::run() {
    while (!stop_) {
        // Wait for a KF or stop signal.
        KeyFrame::Ptr kf;
        {
            std::unique_lock lk(queue_mutex_);
            queue_cv_.wait(lk, [this] {
                return stop_ || stop_requested_ || !queue_.empty();
            });
            if (stop_ && queue_.empty()) break;
            if (!queue_.empty()) {
                kf = std::move(queue_.front());
                queue_.pop_front();
                processing_ = true;
            }
        }

        // Honour pause requests from loop closing.
        while (stop_requested_ && !stop_) {
            stopped_ = true;
            idle_cv_.notify_all();  // wake wait_until_stopped() callers
            std::unique_lock lk(queue_mutex_);
            queue_cv_.wait_for(lk, std::chrono::milliseconds(3), [this] {
                return stop_ || !stop_requested_;
            });
        }
        stopped_ = false;

        if (!kf || kf->is_bad()) {
            mark_processing_done();
            continue;
        }

        triangulate_new_mappoints(kf.get());
        // Compute BoW for the new KF so KeyFrameDatabase can index it.
        if (vocab_) kf->compute_bow(*vocab_);
        const auto ba_t0 = std::chrono::steady_clock::now();
        ba::local_bundle_adjustment(kf.get(), *cam_, ba_params_);
        const auto ba_t1 = std::chrono::steady_clock::now();
        record_ba_time(
            std::chrono::duration<double, std::milli>(ba_t1 - ba_t0).count());
        // Cull AFTER BA so the χ²-based outlier reclassification informs
        // observation counts; then prune now-redundant KeyFrames.
        cull_mappoints(kf.get());
        cull_keyframes(kf.get());

        // Forward the fully-processed KF to LoopClosing (if wired).
        if (loop_closing_ && kf->bow_computed())
            loop_closing_->enqueue_keyframe(kf);

        mark_processing_done();
    }
}

void LocalMapping::mark_processing_done() {
    {
        std::scoped_lock lk(queue_mutex_);
        processing_ = false;
    }
    idle_cv_.notify_all();
}

void LocalMapping::record_ba_time(double ms) {
    std::scoped_lock lk(stats_mutex_);
    ++ba_stats_.runs;
    ba_stats_.total_ms += ms;
    ba_stats_.max_ms = std::max(ba_stats_.max_ms, ms);
}

// ---------------------------------------------------------------------------
// Triangulate new MapPoints from matches with covisible KFs.
// ---------------------------------------------------------------------------

void LocalMapping::triangulate_new_mappoints(KeyFrame* kf) {
    // Projection matrix for kf: P = K * [R | t]
    const Eigen::Matrix4d T_cw = kf->get_pose();
    const Eigen::Matrix3d K    = cam_->K();
    Eigen::Matrix<double, 3, 4> P1;
    P1.block<3, 3>(0, 0) = K * T_cw.block<3, 3>(0, 0);
    P1.block<3, 1>(0, 3) = K * T_cw.block<3, 1>(0, 3);

    const std::vector<cv::KeyPoint>& kps1 = kf->keypoints_left();

    // Limit search to the strongest covisible KFs (sorted by weight) to keep
    // triangulation O(1) in map size rather than O(N_kfs).
    auto neighbours = kf->get_covisibility_keyframes(0);
    if (static_cast<int>(neighbours.size()) > params_.max_triangulation_neighbours)
        neighbours.resize(static_cast<std::size_t>(params_.max_triangulation_neighbours));

    for (KeyFrame* nb : neighbours) {
        if (!nb || nb->is_bad()) continue;

        const Eigen::Matrix4d T_cw2 = nb->get_pose();
        Eigen::Matrix<double, 3, 4> P2;
        P2.block<3, 3>(0, 0) = K * T_cw2.block<3, 3>(0, 0);
        P2.block<3, 1>(0, 3) = K * T_cw2.block<3, 1>(0, 3);

        const std::vector<cv::KeyPoint>& kps2 = nb->keypoints_left();
        const cv::Mat& descs2 = nb->descriptors_left();

        for (int i = 0; i < static_cast<int>(kps1.size()); ++i) {
            // Skip if kf already has a valid MP at this feature.
            if (kf->get_map_point(i) && !kf->get_map_point(i)->is_bad())
                continue;

            // Descriptor matching against nb (tighter than TH_HIGH to avoid
            // wrong associations that would cause BA to diverge).
            const cv::Mat d1 = kf->descriptors_left().row(i);
            int best_dist = 50, second_dist = 50, best_j = -1;
            for (int j = 0; j < static_cast<int>(kps2.size()); ++j) {
                const int d = cv::norm(d1, descs2.row(j), cv::NORM_HAMMING);
                if (d < best_dist) {
                    second_dist = best_dist;
                    best_dist   = d;
                    best_j      = j;
                } else if (d < second_dist) {
                    second_dist = d;
                }
            }
            // Require Lowe ratio: best must be clearly better than second.
            if (best_j < 0 || (second_dist < 50 && best_dist > second_dist * 0.75f))
                continue;

            // Skip if nb already has a valid MP at this feature.
            if (nb->get_map_point(best_j) && !nb->get_map_point(best_j)->is_bad())
                continue;

            const Eigen::Vector2d x1(kps1[i].pt.x, kps1[i].pt.y);
            const Eigen::Vector2d x2(kps2[best_j].pt.x, kps2[best_j].pt.y);

            const Eigen::Vector3d pw = triangulate_linear(P1, P2, x1, x2);

            if (!check_triangulated(P1, P2, T_cw, T_cw2, pw, x1, x2))
                continue;

            // Create new MapPoint and register with both KFs.
            const uint64_t mp_id = map_->allocate_mappoint_id();
            auto mp = std::make_shared<MapPoint>(mp_id, pw, kf);
            mp->add_observation(kf, i);
            mp->add_observation(nb, best_j);
            mp->compute_descriptor();
            mp->update_normal_and_depth();

            kf->add_map_point(i, mp);
            nb->add_map_point(best_j, mp);
            map_->add_mappoint(mp);
        }
    }

    // Update covisibility once after all neighbour passes.  Doing it inside
    // the inner loop made the neighbour iteration order observable to the
    // graph and yielded non-deterministic edge sets across runs.
    kf->update_connections();
    for (KeyFrame* nb : neighbours) {
        if (nb && !nb->is_bad()) nb->update_connections();
    }
}

// ---------------------------------------------------------------------------
// Cull bad MapPoints.
// ---------------------------------------------------------------------------

void LocalMapping::cull_mappoints(KeyFrame* current_kf) {
    if (!current_kf) return;

    // Restrict the inspection to MapPoints in the local window of
    // current_kf — both ones it observes directly and ones observed by
    // its strongest covisibility neighbours.  Walking the whole map per
    // KeyFrame culled MPs that distant KFs had only just created.
    std::unordered_set<MapPoint*> local_mps;
    {
        const auto own_mps = current_kf->get_map_points();
        for (const auto& mp : own_mps) if (mp) local_mps.insert(mp.get());
    }
    auto neighbours = current_kf->get_covisibility_keyframes(0);
    if (static_cast<int>(neighbours.size()) > params_.max_cull_neighbours)
        neighbours.resize(static_cast<std::size_t>(params_.max_cull_neighbours));
    for (KeyFrame* nb : neighbours) {
        if (!nb || nb->is_bad()) continue;
        const auto nb_mps = nb->get_map_points();
        for (const auto& mp : nb_mps) if (mp) local_mps.insert(mp.get());
    }

    // Snapshot all map points (shared_ptr keeps them alive while we work).
    const auto all_mps = map_->get_all_mappoints();
    for (const auto& mp : all_mps) {
        if (!mp || mp->is_bad()) continue;
        if (!local_mps.count(mp.get())) continue;
        if (current_kf->id() < mp->created_kf_id() +
                                   static_cast<uint64_t>(params_.mappoint_grace_kfs)) {
            continue;
        }
        if (mp->n_observations() < params_.min_mappoint_observations)
            mp->set_bad();
    }
}

// ---------------------------------------------------------------------------
// Cull redundant KeyFrames.
// ---------------------------------------------------------------------------

void LocalMapping::cull_keyframes(KeyFrame* kf) {
    // Only inspect a local window around the current KeyFrame.
    auto local_kfs = kf->get_covisibility_keyframes(0);
    if (static_cast<int>(local_kfs.size()) > params_.max_cull_neighbours)
        local_kfs.resize(static_cast<std::size_t>(params_.max_cull_neighbours));

    for (KeyFrame* ck : local_kfs) {
        if (!ck || ck->is_bad()) continue;
        // Don't cull the very first KF (id 0) — it anchors the map origin.
        if (ck->id() == 0) continue;
        // Skip KFs that are still inside the grace window: their MPs
        // haven't had a chance to be re-observed by enough neighbours yet.
        if (kf->id() < ck->id() +
                            static_cast<uint64_t>(params_.mappoint_grace_kfs))
            continue;

        const auto mps = ck->get_map_points();
        if (mps.empty()) continue;

        int n_redundant = 0;
        for (const auto& mp : mps) {
            if (!mp || mp->is_bad()) continue;
            // Count observations from KFs other than ck.
            int n_other = 0;
            for (const auto& [obs_kf, fidx] : mp->get_observations()) {
                if (obs_kf != ck && !obs_kf->is_bad()) ++n_other;
            }
            if (n_other >= 3) ++n_redundant;
        }

        const double redundancy =
            static_cast<double>(n_redundant) / static_cast<double>(mps.size());
        if (redundancy >= 0.9)
            ck->set_bad();
    }
}

}  // namespace sslam
