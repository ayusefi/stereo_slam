#include "sslam/types/keyframe.hpp"

#include "sslam/types/map.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace sslam {

namespace {

Eigen::Matrix4d inverse_se3(const Eigen::Matrix4d& T_cw) {
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R_wc = T_cw.topLeftCorner<3, 3>().transpose();
    T_wc.topLeftCorner<3, 3>()  = R_wc;
    T_wc.topRightCorner<3, 1>() = -R_wc * T_cw.topRightCorner<3, 1>();
    return T_wc;
}

}  // namespace

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

void KeyFrame::set_bad() {
    // KF id 0 anchors the map origin — never cull it.
    if (id_ == 0) return;

    // Double-checked: bail if already bad.
    bool expected = false;
    if (!bad_.compare_exchange_strong(expected, true,
                                      std::memory_order_acq_rel)) {
        return;
    }

    // --- Compute and store T_to_parent BEFORE any connection cleanup ------
    Eigen::Matrix4d T_self;
    {
        std::scoped_lock lk(pose_mutex_);
        T_self = T_cw_;
    }
    Eigen::Matrix4d T_to_parent = Eigen::Matrix4d::Identity();
    if (parent_ && !parent_->is_bad()) {
        T_to_parent = T_self * inverse_se3(parent_->get_pose());
    }
    {
        std::scoped_lock lk(pose_mutex_);
        T_bad_to_parent_ = T_to_parent;
    }

    // --- Snapshot children and covisibility, then clear own records -------
    std::unordered_set<KeyFrame*> children_copy;
    std::vector<std::pair<KeyFrame*, int>> cov_copy;  // (kf, weight)
    {
        std::scoped_lock lk(obs_mutex_);
        children_copy = children_;
        children_.clear();
        cov_copy.reserve(covisibility_.size());
        for (const auto& [kf, w] : covisibility_)
            cov_copy.push_back({kf, w});
        covisibility_.clear();
    }

    // --- Re-parent each child --------------------------------------------
    // Try to give each child the highest-weight non-bad covisible of THIS KF
    // as its new parent. Fall back to grandparent. If none, leave child
    // connected to this (bad) KF — chain-walking in get_pose_through_spanning_tree
    // will still resolve the pose correctly.
    // Sort covisibles by weight descending so the best candidate is tried first.
    std::sort(cov_copy.begin(), cov_copy.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (KeyFrame* child : children_copy) {
        if (child->is_bad()) continue;
        bool found = false;
        for (const auto& [cov_kf, w] : cov_copy) {
            if (cov_kf == child || cov_kf->is_bad()) continue;
            child->set_parent(cov_kf);
            found = true;
            break;
        }
        if (!found && parent_ && !parent_->is_bad()) {
            child->set_parent(parent_);
        }
        // If still nothing: child keeps its pointer to this bad KF; chain-walk handles it.
    }

    // --- Remove this KF from each covisible's connection table -----------
    for (const auto& [cov_kf, w] : cov_copy) {
        if (!cov_kf) continue;
        std::scoped_lock lk(cov_kf->obs_mutex_);
        cov_kf->covisibility_.erase(this);
    }

    // --- Remove from parent's children list (if any) --------------------
    if (parent_) {
        std::scoped_lock lk(parent_->obs_mutex_);
        parent_->children_.erase(this);
    }

    // --- Remove from Map -------------------------------------------------
    if (map_) map_->remove_keyframe(id_);
}

// --- Spanning tree -------------------------------------------------------

void KeyFrame::set_parent(KeyFrame* new_parent) {
    KeyFrame* old_parent = parent_;
    parent_ = new_parent;

    // Remove from old parent's children list.
    if (old_parent && old_parent != new_parent) {
        std::scoped_lock lk(old_parent->obs_mutex_);
        old_parent->children_.erase(this);
    }
    // Register as child of new parent.
    if (new_parent) {
        std::scoped_lock lk(new_parent->obs_mutex_);
        new_parent->children_.insert(this);
    }
}

void KeyFrame::add_child(KeyFrame* kf) {
    if (!kf) return;
    std::scoped_lock lk(obs_mutex_);
    children_.insert(kf);
}

void KeyFrame::remove_child(KeyFrame* kf) {
    if (!kf) return;
    std::scoped_lock lk(obs_mutex_);
    children_.erase(kf);
}

Eigen::Matrix4d KeyFrame::get_pose_through_spanning_tree() const {
    Eigen::Matrix4d T_acc = Eigen::Matrix4d::Identity();
    const KeyFrame* kf = this;

    for (int depth = 0; kf && kf->is_bad() && kf->parent_ && depth < 10000; ++depth) {
        Eigen::Matrix4d T_to_parent;
        {
            std::scoped_lock lk(kf->pose_mutex_);
            T_to_parent = kf->T_bad_to_parent_;
        }
        T_acc = T_acc * T_to_parent;
        kf = kf->parent_;
    }

    if (!kf) return T_acc;
    return T_acc * kf->get_pose();
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

    // Sort by weight descending.  `covisibility_` is a pointer-keyed
    // unordered_map, so iterating it yields an ASLR-dependent order; if we only
    // compared weights, keyframes with equal covisibility weight would come out
    // in a different order on every run.  That ordering propagates into the
    // local-BA window, TrackLocalMap neighbour set and triangulation peers, so
    // a non-deterministic tie-break makes the whole trajectory irreproducible.
    // Break ties on the (stable, monotonically assigned) keyframe id to impose
    // a total order independent of memory layout.
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second->id() < b.second->id();
              });

    std::vector<KeyFrame*> result;
    result.reserve(sorted.size());
    for (const auto& [w, kf] : sorted)
        result.push_back(kf);
    return result;
}

int KeyFrame::get_covisibility_weight(const KeyFrame* kf) const {
    std::scoped_lock lk(obs_mutex_);
    const auto it = covisibility_.find(const_cast<KeyFrame*>(kf));
    return (it != covisibility_.end()) ? it->second : 0;
}

}  // namespace sslam
