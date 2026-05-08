#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"
#include "sslam/types/mappoint.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sslam {

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
    bool     is_bad()    const { return bad_; }
    void     set_bad()         { bad_ = true; }

    // --- Pose (guarded by pose_mutex_) ------------------------------------

    /// Returns T_cw (world → camera), SE(3) as 4×4.
    Eigen::Matrix4d get_pose() const;
    void            set_pose(const Eigen::Matrix4d& T_cw);

    /// Camera centre in world coordinates: $-R^T t$.
    Eigen::Vector3d camera_center() const;

    // --- Feature data (immutable after construction) ----------------------

    std::size_t num_features() const { return keypoints_left_.size(); }

    const std::vector<cv::KeyPoint>& keypoints_left()   const { return keypoints_left_; }
    const cv::Mat&                   descriptors_left() const { return descriptors_left_; }
    const std::vector<float>&        right_u()          const { return right_u_; }
    const std::vector<float>&        depth()            const { return depth_; }
    std::shared_ptr<const StereoCamera> camera()        const { return camera_; }

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

    // --- Spanning tree (Phase 3+) ----------------------------------------
    KeyFrame* parent() const { return parent_; }
    void      set_parent(KeyFrame* kf) { parent_ = kf; }

   private:
    const uint64_t id_;
    double         timestamp_{0.0};
    bool           bad_{false};

    std::shared_ptr<const StereoCamera> camera_;

    // Feature data — immutable after construction --------------------------
    std::vector<cv::KeyPoint> keypoints_left_;
    cv::Mat                   descriptors_left_;
    std::vector<float>        right_u_;
    std::vector<float>        depth_;

    // --- Guarded by pose_mutex_ ------------------------------------------
    mutable std::mutex pose_mutex_;
    Eigen::Matrix4d    T_cw_{Eigen::Matrix4d::Identity()};

    // --- Guarded by obs_mutex_ -------------------------------------------
    mutable std::mutex                     obs_mutex_;
    std::unordered_map<int, MapPoint::Ptr> observations_;  ///< feat_idx → MP (shared ownership)
    std::unordered_map<KeyFrame*, int>     covisibility_;  ///< raw non-owning KF* → shared obs count

    // Spanning tree parent (raw non-owning, Phase 3+)
    KeyFrame* parent_{nullptr};
};

}  // namespace sslam
