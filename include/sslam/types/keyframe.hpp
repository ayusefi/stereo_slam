#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"
#include "sslam/types/mappoint.hpp"

#include <unordered_set>

#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sslam {

class Map;  // forward declaration

/// A Frame promoted to a permanent, map-resident keyframe.
///
/// Co-owns the MapPoints it observes via shared_ptr alongside Map.
/// Back-references from MapPoints to this KF use raw non-owning pointers.
///
/// Thread-safety:
///   pose_mutex_  — guards T_cw_
///   obs_mutex_   — guards observations_, covisibility_
///
/// Lock ordering (see CODING_STYLE.md §Concurrency):
///   Map::mutex_ → KeyFrame::obs_mutex_ → MapPoint::pos_mutex_
class KeyFrame {
   public:
    using Ptr = std::shared_ptr<KeyFrame>;

    /// Build from a tracked Frame.  Feature arrays are shallow-copied
    /// (cv::Mat uses reference counting); descriptors are cloned for safety.
    KeyFrame(uint64_t id, const Frame& f,
             std::shared_ptr<const StereoCamera> cam);

    uint64_t id()        const { return id_; }
    double   timestamp() const { return timestamp_; }
    bool     is_bad()    const { return bad_.load(std::memory_order_relaxed); }
    void     set_bad();

    // --- Pose (guarded by pose_mutex_) ------------------------------------

    /// Returns T_cw (world → camera), SE(3) as 4×4.
    Eigen::Matrix4d get_pose() const;
    void            set_pose(const Eigen::Matrix4d& T_cw);

    /// Pose for trajectory export.  If this KF was culled, follow its
    /// stored spanning-tree transform to the first non-bad parent instead
    /// of falling back to a raw frame pose.
    Eigen::Matrix4d get_pose_through_spanning_tree() const;

    /// Camera centre in world coordinates: $-R^T t$.
    Eigen::Vector3d camera_center() const;

    // --- Feature data (immutable after construction) ----------------------

    std::size_t num_features() const { return keypoints_left_.size(); }

    const std::vector<cv::KeyPoint>& keypoints_left()   const { return keypoints_left_; }
    const cv::Mat&                   descriptors_left() const { return descriptors_left_; }
    const std::vector<float>&        right_u()          const { return right_u_; }
    const std::vector<float>&        depth()            const { return depth_; }
    std::shared_ptr<const StereoCamera> camera()        const { return camera_; }

    /// ORB scale-factor pyramid (set once after construction by Tracking).
    /// scale_factors()[0] = 1.0; scale_factors()[i] = scale_factor^i.
    /// Used by MapPoint::update_normal_and_depth and BA information matrices.
    void set_scale_factors(const std::vector<float>& sf) { scale_factors_ = sf; }
    const std::vector<float>& scale_factors() const { return scale_factors_; }

    // --- MapPoint observations (guarded by obs_mutex_) -------------------

    void          add_map_point(int feat_idx, MapPoint::Ptr mp);
    MapPoint::Ptr get_map_point(int feat_idx) const;
    void          erase_map_point(int feat_idx);

    /// All non-null, non-bad MPs observed by this KF.
    std::vector<MapPoint::Ptr> get_map_points() const;

    /// Count MPs with at least min_obs total observations (across all KFs).
    int tracked_map_points(int min_obs = 1) const;

    // --- Covisibility graph (guarded by obs_mutex_) ----------------------

    /// Set (or overwrite) the covisibility weight with kf.
    void add_connection(KeyFrame* kf, int weight);

    /// Rebuild covisibility edges from current observations_.
    /// Call after all add_map_point() calls for a freshly-inserted KF.
    /// Also notifies peer KFs of the new edge (calls their add_connection).
    void update_connections();

    /// Return covisible KFs with weight >= min_weight, sorted descending.
    std::vector<KeyFrame*> get_covisibility_keyframes(int min_weight = 0) const;

    // --- Spanning tree ----------------------------------------------------
    KeyFrame* parent() const { return parent_; }

    /// Set spanning-tree parent. Updates children_ on both old and new parent.
    void set_parent(KeyFrame* kf);

    /// Children management (called automatically by set_parent).
    void add_child(KeyFrame* kf);
    void remove_child(KeyFrame* kf);

    /// Set owning Map back-pointer. Called by Map::add_keyframe().
    void set_map(Map* m) { map_ = m; }

    // --- Bag-of-Words (guarded by bow_mutex_) ----------------------------

    /// Compute BoW and FeatureVector from this KF's left descriptors.
    /// No-op if already computed.  Thread-safe.
    template <class Vocab>
    void compute_bow(const Vocab& vocab) {
        std::scoped_lock lk(bow_mutex_);
        if (!bow_.empty()) return;
        std::vector<cv::Mat> descs;
        descs.reserve(static_cast<std::size_t>(descriptors_left_.rows));
        for (int i = 0; i < descriptors_left_.rows; ++i)
            descs.push_back(descriptors_left_.row(i));
        vocab.transform(descs, bow_, feat_vec_, 4);
    }

    DBoW2::BowVector   bow()      const { std::scoped_lock lk(bow_mutex_); return bow_; }
    DBoW2::FeatureVector feat_vec() const { std::scoped_lock lk(bow_mutex_); return feat_vec_; }
    bool bow_computed()            const { std::scoped_lock lk(bow_mutex_); return !bow_.empty(); }

   private:
    const uint64_t id_;
    double         timestamp_{0.0};
    std::atomic<bool> bad_{false};

    std::shared_ptr<const StereoCamera> camera_;

    // Feature data — immutable after construction --------------------------
    std::vector<cv::KeyPoint> keypoints_left_;
    cv::Mat                   descriptors_left_;
    std::vector<float>        right_u_;
    std::vector<float>        depth_;
    std::vector<float>        scale_factors_;  ///< ORB pyramid: [1, s, s^2, ...], set by Tracking

    // --- Guarded by pose_mutex_ ------------------------------------------
    mutable std::mutex pose_mutex_;
    Eigen::Matrix4d    T_cw_{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d    T_bad_to_parent_{Eigen::Matrix4d::Identity()};

    // --- Guarded by obs_mutex_ -------------------------------------------
    mutable std::mutex                     obs_mutex_;
    std::unordered_map<int, MapPoint::Ptr> observations_;  ///< feat_idx → MP (shared ownership)
    std::unordered_map<KeyFrame*, int>     covisibility_;  ///< raw non-owning KF* → shared obs count

    // Spanning tree (raw non-owning pointers)
    KeyFrame*                            parent_{nullptr};
    std::unordered_set<KeyFrame*>        children_;  ///< guarded by obs_mutex_
    Map*                                 map_{nullptr};

    // --- Guarded by bow_mutex_ -------------------------------------------
    mutable std::mutex   bow_mutex_;
    DBoW2::BowVector     bow_;
    DBoW2::FeatureVector feat_vec_;
};

}  // namespace sslam
