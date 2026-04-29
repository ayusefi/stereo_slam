#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <string>

namespace sslam {

/// Calibrated, *already-rectified* stereo camera model.
///
/// After rectification the left and right images share intrinsics, and the
/// right camera is at (-baseline, 0, 0) relative to the left.
///
///   K = [[fx, 0, cx],
///        [ 0,fy, cy],
///        [ 0, 0,  1]]
///
/// For KITTI we read K_l and K_r from `calib.txt`; both are identical after
/// rectification by construction. The baseline `b` is recovered from the
/// projection matrix P_r = K [I | -K^{-1} * t] where t.x = -fx * b.
struct StereoCamera {
    double fx{0}, fy{0}, cx{0}, cy{0};
    double baseline{0};   // metres, always positive
    int    width{0}, height{0};

    Eigen::Matrix3d K() const {
        Eigen::Matrix3d k;
        k << fx, 0, cx,
              0, fy, cy,
              0,  0,  1;
        return k;
    }

    /// Convert (u, v, disparity) → 3D point in left-camera frame.
    /// disparity must be > 0.
    Eigen::Vector3d backproject(double u, double v, double disparity) const;
};

}  // namespace sslam
