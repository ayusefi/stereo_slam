#include "sslam/types/mappoint.hpp"

#include "sslam/types/keyframe.hpp"

#include <algorithm>
#include <climits>
#include <cmath>

namespace sslam {

MapPoint::MapPoint(uint64_t id, const Eigen::Vector3d& pos_w, KeyFrame* ref_kf)
    : id_(id), ref_kf_(ref_kf), pos_w_(pos_w) {}

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
        descs.reserve(observations_.size());
        for (const auto& [kf, idx] : observations_) {
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
    // ORBExtractor defaults: scale_factor = 1.2, num_levels = 8.
    constexpr float kScaleFactor = 1.2f;
    constexpr int   kNumLevels   = 8;

    const auto ref_it = obs.find(ref_kf_);
    if (ref_it == obs.end()) return;

    const int   octave    = ref_kf_->keypoints_left()[ref_it->second].octave;
    const float dist_ref  = static_cast<float>(
        (pos - ref_kf_->camera_center()).norm());

    float level_scale = 1.0f;
    for (int i = 0; i < octave; ++i) level_scale *= kScaleFactor;

    const float d_max = dist_ref * level_scale;
    const float d_min = d_max / std::pow(kScaleFactor, kNumLevels - 1);

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

}  // namespace sslam
