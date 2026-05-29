#include "sslam/types/mappoint.hpp"

#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

namespace sslam {

MapPoint::MapPoint(uint64_t id, const Eigen::Vector3d& pos_w, KeyFrame* ref_kf)
        : id_(id),
            created_kf_id_(ref_kf ? ref_kf->id() : 0),
            ref_kf_(ref_kf),
            pos_w_(pos_w) {}

// --- World position ------------------------------------------------------

Eigen::Vector3d MapPoint::get_world_pos() const {
    std::scoped_lock lk(pos_mutex_);
    return pos_w_;
}

void MapPoint::set_world_pos(const Eigen::Vector3d& pos_w) {
    std::scoped_lock lk(pos_mutex_);
    pos_w_ = pos_w;
}

// --- Observations --------------------------------------------------------

void MapPoint::add_observation(KeyFrame* kf, int feat_idx) {
    std::scoped_lock lk(obs_mutex_);
    observations_[kf] = feat_idx;
}

void MapPoint::remove_observation(KeyFrame* kf) {
    std::scoped_lock lk(obs_mutex_);
    observations_.erase(kf);
}

int MapPoint::get_feat_idx(KeyFrame* kf) const {
    std::scoped_lock lk(obs_mutex_);
    const auto it = observations_.find(kf);
    return it != observations_.end() ? it->second : -1;
}

MapPoint::ObsMap MapPoint::get_observations() const {
    std::scoped_lock lk(obs_mutex_);
    return observations_;
}

int MapPoint::n_observations() const {
    std::scoped_lock lk(obs_mutex_);
    return static_cast<int>(observations_.size());
}

// --- Descriptor ----------------------------------------------------------

cv::Mat MapPoint::get_descriptor() const {
    std::scoped_lock lk(obs_mutex_);
    return descriptor_.clone();
}

void MapPoint::compute_descriptor() {
    // Collect one descriptor row per valid observation.
    std::vector<cv::Mat> descs;
    {
        std::scoped_lock lk(obs_mutex_);
        // Iterate observations in KeyFrame-id order.  observations_ is an
        // unordered_map keyed by KeyFrame*, whose iteration order depends on
        // pointer hashing (ASLR).  The representative-descriptor selection
        // below keeps the first candidate on a median-distance tie, so an
        // unstable order would pick different descriptors across runs and
        // make BoW matching / loop detection non-deterministic.
        std::vector<std::pair<KeyFrame*, int>> obs_sorted(
            observations_.begin(), observations_.end());
        std::sort(obs_sorted.begin(), obs_sorted.end(),
                  [](const auto& a, const auto& b) {
                      return a.first->id() < b.first->id();
                  });
        descs.reserve(obs_sorted.size());
        for (const auto& [kf, idx] : obs_sorted) {
            if (kf->is_bad()) continue;
            const cv::Mat row = kf->descriptors_left().row(idx);
            if (!row.empty())
                descs.push_back(row.clone());
        }
    }

    if (descs.empty()) return;

    if (descs.size() == 1u) {
        std::scoped_lock lk(obs_mutex_);
        descriptor_ = descs[0].clone();
        return;
    }

    // Pairwise Hamming distances.
    const int n = static_cast<int>(descs.size());
    std::vector<std::vector<int>> dist(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            const int d = cv::norm(descs[i], descs[j], cv::NORM_HAMMING);
            dist[i][j] = dist[j][i] = d;
        }

    // Pick descriptor with minimum median Hamming distance (ORB-SLAM2 §IV-B).
    int best_idx = 0, best_median = INT_MAX;
    for (int i = 0; i < n; ++i) {
        std::vector<int> row(dist[i].begin(), dist[i].end());
        std::sort(row.begin(), row.end());
        const int med = row[n / 2];
        if (med < best_median) {
            best_median = med;
            best_idx    = i;
        }
    }

    std::scoped_lock lk(obs_mutex_);
    descriptor_ = descs[static_cast<std::size_t>(best_idx)].clone();
}

// --- Normal and depth range ----------------------------------------------

void MapPoint::update_normal_and_depth() {
    if (!ref_kf_) return;

    Eigen::Vector3d pos;
    { std::scoped_lock lk(pos_mutex_); pos = pos_w_; }

    ObsMap obs;
    { std::scoped_lock lk(obs_mutex_); obs = observations_; }
    if (obs.empty()) return;

    // Mean viewing direction.
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (const auto& [kf, idx] : obs) {
        const Eigen::Vector3d dir = (pos - kf->camera_center()).normalized();
        normal += dir;
    }
    normal.normalize();

    // Scale-invariance range from the reference keypoint's octave.
    // Use the actual ORB extractor scale pyramid stored in the reference KF;
    // fall back to the ORB defaults (1.2, 8 levels) if not yet populated.
    const std::vector<float>& sf = ref_kf_->scale_factors();

    const auto ref_it = obs.find(ref_kf_);
    if (ref_it == obs.end()) return;

    const int   octave   = ref_kf_->keypoints_left()[ref_it->second].octave;
    const float dist_ref = static_cast<float>(
        (pos - ref_kf_->camera_center()).norm());

    float d_max;
    if (!sf.empty() && octave < static_cast<int>(sf.size())) {
        d_max = dist_ref * sf[static_cast<std::size_t>(octave)];
    } else {
        // Fallback: ORB defaults (scale_factor=1.2)
        float level_scale = 1.0f;
        for (int i = 0; i < octave; ++i) level_scale *= 1.2f;
        d_max = dist_ref * level_scale;
    }

    const float top_scale = sf.empty() ? std::pow(1.2f, 7) : sf.back();
    const float d_min     = d_max / top_scale;

    std::scoped_lock lk(pos_mutex_);
    normal_       = normal;
    min_distance_ = d_min;
    max_distance_ = d_max;
}

Eigen::Vector3d MapPoint::mean_normal() const {
    std::scoped_lock lk(pos_mutex_);
    return normal_;
}

float MapPoint::min_distance() const {
    std::scoped_lock lk(pos_mutex_);
    return min_distance_;
}

float MapPoint::max_distance() const {
    std::scoped_lock lk(pos_mutex_);
    return max_distance_;
}

// --- Lifecycle -----------------------------------------------------------

void MapPoint::set_bad() {
    // Atomic flip; bail if some other thread already retired this MP.
    bool expected = false;
    if (!bad_.compare_exchange_strong(expected, true,
                                      std::memory_order_acq_rel)) {
        return;
    }

    // Snapshot and clear observations under the MP lock so concurrent
    // readers see an empty map immediately after the bad flag is set.
    ObsMap snapshot;
    {
        std::scoped_lock lk(obs_mutex_);
        snapshot = std::move(observations_);
        observations_.clear();
    }

    // Erase the back-references from the observing KeyFrames.
    // Each call takes only that KF's obs_mutex_ — no nesting with our locks.
    for (const auto& [kf, feat_idx] : snapshot) {
        if (kf) kf->erase_map_point(feat_idx);
    }

    // Drop ourselves from the owning map's index so future scans don't see us.
    if (map_) map_->remove_mappoint(id_);
}

void MapPoint::replace(const MapPoint::Ptr& other) {
    if (!other || other.get() == this || other->is_bad()) return;

    // Snapshot own observations and set replaced_with_ atomically under our lock.
    ObsMap snapshot;
    {
        std::scoped_lock lk(obs_mutex_);
        // Already replaced or being retired by another thread — bail.
        if (replaced_with_ || bad_.load(std::memory_order_relaxed)) return;
        replaced_with_ = other.get();
        snapshot = std::move(observations_);
        observations_.clear();
    }

    // Redirect every observation to `other`.
    for (const auto& [kf, feat_idx] : snapshot) {
        if (!kf) continue;
        if (other->get_feat_idx(kf) < 0) {
            // other doesn't have this KF yet — hand it over.
            kf->add_map_point(feat_idx, other);
            other->add_observation(kf, feat_idx);
        } else {
            // Conflict: other already covers this KF — just drop the slot.
            kf->erase_map_point(feat_idx);
        }
    }

    other->compute_descriptor();

    // Now retire ourselves. observations_ is already empty so set_bad will
    // only flip the flag and remove from map — no double-erase.
    set_bad();
}

MapPoint* MapPoint::get_replaced() const {
    std::scoped_lock lk(obs_mutex_);
    return replaced_with_;
}

}  // namespace sslam
