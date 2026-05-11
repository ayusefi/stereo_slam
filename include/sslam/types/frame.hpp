#pragma once

#include "sslam/camera/stereo_camera.hpp"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sslam {

class KeyFrame;  // forward-declaration; only a non-owning pointer is held.

/// A single stereo capture as it travels through the tracking pipeline.
///
/// Lifecycle:
///   1. Constructed from a `StereoFrame` (raw images + timestamp).
///   2. Frontend fills `keypoints_left` / `descriptors_left` and, after
///      stereo matching, `right_u` and `depth` (parallel arrays, same size
///      as `keypoints_left`; -1.0f means "no stereo match" → mono only).
///   3. Tracking sets `T_cw` (world → camera SE(3)) once a pose is solved.
///   4. The frame is either dropped or promoted to a `KeyFrame`.
///
/// Heavy image data is stored by value here for simplicity. Once
/// memory pressure becomes real, we'll move them to shared_ptr
/// and let only KeyFrames retain pixels.
struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    std::size_t  index{0};
    double       timestamp{0.0};

    cv::Mat      left;            // CV_8UC1
    cv::Mat      right;           // CV_8UC1

    // --- Filled by the frontend ----------------------------------------
    std::vector<cv::KeyPoint> keypoints_left;
    cv::Mat                   descriptors_left;   // CV_8U, rows = #keypoints
    std::vector<float>        right_u;            // x in right image, -1 if none
    std::vector<float>        depth;              // metres,   -1 if none

    // --- Filled by tracking ----------------------------------
    /// Pose of the world expressed in the camera frame (SE(3)). Identity
    /// for the very first frame.
    Eigen::Matrix4d T_cw{Eigen::Matrix4d::Identity()};

    // --- Map-anchored pose (set after a KeyFrame exists) -----------------
    /// Reference KeyFrame at the time this frame was tracked (raw,
    /// non-owning; KeyFrames are co-owned by Map).  Null for frames
    /// processed before the first KeyFrame is inserted.
    KeyFrame*       ref_kf{nullptr};
    /// Relative pose: T_cw expressed in ref_kf's camera frame, i.e.
    /// `T_ref = T_cw * inverse(ref_kf->get_pose())` evaluated at tracking
    /// time.  Lets us re-anchor `T_cw` after Local BA moves `ref_kf`:
    ///   `T_cw_corrected = T_ref * ref_kf->get_pose()`.
    Eigen::Matrix4d T_ref{Eigen::Matrix4d::Identity()};

    // --- Shared metadata -------------------------------------------------
    std::shared_ptr<const StereoCamera> camera;

    Frame() = default;

    /// Build from a freshly-loaded stereo capture.
    Frame(std::size_t idx, double ts, cv::Mat l, cv::Mat r,
          std::shared_ptr<const StereoCamera> cam)
        : index(idx), timestamp(ts),
          left(std::move(l)), right(std::move(r)),
          camera(std::move(cam)) {}

    std::size_t num_features() const { return keypoints_left.size(); }
};

}  // namespace sslam
