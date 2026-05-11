#include "sslam/types/keyframe.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace sslam {

KeyFrame::KeyFrame(uint64_t id, const Frame& f,
                   std::shared_ptr<const StereoCamera> cam)
    : id_(id),
      timestamp_(f.timestamp),
      camera_(std::move(cam)),
      keypoints_left_(f.keypoints_left),
      descriptors_left_(f.descriptors_left.clone()),
      right_u_(f.right_u),
      depth_(f.depth) {
    std::scoped_lock lk(pose_mutex_);
    T_cw_ = f.T_cw;
}

// --- Pose ----------------------------------------------------------------

Eigen::Matrix4d KeyFrame::get_pose() const {
    std::scoped_lock lk(pose_mutex_);
    return T_cw_;
}

void KeyFrame::set_pose(const Eigen::Matrix4d& T_cw) {
    std::scoped_lock lk(pose_mutex_);
    T_cw_ = T_cw;
}

Eigen::Vector3d KeyFrame::camera_center() const {
    std::scoped_lock lk(pose_mutex_);
    const Eigen::Matrix3d R_wc = T_cw_.topLeftCorner<3, 3>().transpose();
    return -R_wc * T_cw_.topRightCorner<3, 1>();
}

// --- MapPoint observations -----------------------------------------------

void KeyFrame::add_map_point(int feat_idx, MapPoint::Ptr mp) {
    std::scoped_lock lk(obs_mutex_);
    observations_[feat_idx] = std::move(mp);
}

MapPoint::Ptr KeyFrame::get_map_point(int feat_idx) const {
    std::scoped_lock lk(obs_mutex_);
    const auto it = observations_.find(feat_idx);
    return it != observations_.end() ? it->second : nullptr;
}

void KeyFrame::erase_map_point(int feat_idx) {
    std::scoped_lock lk(obs_mutex_);
    observations_.erase(feat_idx);
}

std::vector<MapPoint::Ptr> KeyFrame::get_map_points() const {
    std::scoped_lock lk(obs_mutex_);
    std::vector<MapPoint::Ptr> mps;
    mps.reserve(observations_.size());
    for (const auto& [idx, mp] : observations_)
        if (mp && !mp->is_bad()) mps.push_back(mp);
    return mps;
}

int KeyFrame::tracked_map_points(int min_obs) const {
    std::scoped_lock lk(obs_mutex_);
    int count = 0;
    for (const auto& [idx, mp] : observations_)
        if (mp && !mp->is_bad() && mp->n_observations() >= min_obs)
            ++count;
    return count;
}

// --- Covisibility graph --------------------------------------------------

void KeyFrame::add_connection(KeyFrame* kf, int weight) {
    std::scoped_lock lk(obs_mutex_);
    covisibility_[kf] = weight;
}

void KeyFrame::update_connections() {
    // Step 1: snapshot own MP observations (release lock before accessing MPs).
    std::vector<MapPoint::Ptr> own_mps;
    {
        std::scoped_lock lk(obs_mutex_);
        own_mps.reserve(observations_.size());
        for (const auto& [idx, mp] : observations_)
            if (mp && !mp->is_bad()) own_mps.push_back(mp);
    }

    // Step 2: for each MP, find other KFs that also observe it.
    // MP::get_observations() acquires MP's obs_mutex_ internally — safe here
    // since we hold no lock at this point.
    std::unordered_map<KeyFrame*, int> kf_counter;
    for (const auto& mp : own_mps)
        for (const auto& [other_kf, feat_idx] : mp->get_observations()) {
            if (other_kf == this) continue;
            ++kf_counter[other_kf];
        }

    // Step 3: overwrite own covisibility edges.
    {
        std::scoped_lock lk(obs_mutex_);
        covisibility_.clear();
        for (const auto& [kf, w] : kf_counter)
            covisibility_[kf] = w;
    }

    // Step 4: notify peer KFs — they acquire their own obs_mutex_, no nesting.
    // note: if two KFs call update_connections() concurrently they
    // may write each other's add_connection simultaneously; acceptable since
    // writes are idempotent and Lock ordering (Map > KF obs) is preserved.
    for (const auto& [other_kf, weight] : kf_counter)
        other_kf->add_connection(this, weight);
}

std::vector<KeyFrame*> KeyFrame::get_covisibility_keyframes(int min_weight) const {
    std::scoped_lock lk(obs_mutex_);

    std::vector<std::pair<int, KeyFrame*>> sorted;
    sorted.reserve(covisibility_.size());
    for (const auto& [kf, w] : covisibility_)
        if (w >= min_weight) sorted.emplace_back(w, kf);

    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<KeyFrame*> result;
    result.reserve(sorted.size());
    for (const auto& [w, kf] : sorted)
        result.push_back(kf);
    return result;
}

}  // namespace sslam
