#pragma once

#include "sslam/camera/stereo_camera.hpp"

#include <Eigen/Core>

#include <cmath>
#include <vector>

namespace sslam {
namespace ba {

/// Parameters for the motion-only bundle-adjustment optimiser.
struct Params {
    /// Number of outer iterations.  After each outer iteration every edge is
    /// re-classified as inlier or outlier using chi2_threshold on the current
    /// estimate; outlier edges are excluded from the following iteration.
    int outer_iterations{4};
    /// Number of inner (Levenberg–Marquardt) iterations per outer step.
    int inner_iterations{10};
    /// Huber kernel half-width δ for a 3-DoF stereo residual.
    /// δ = sqrt(7.815) places the boundary at the 3-DoF χ² 95th percentile.
    double huber_delta{std::sqrt(7.815)};
    /// Chi-squared threshold for inlier/outlier reclassification (3-DoF, 95%).
    double chi2_threshold{7.815};
};

/// Result of motion-only pose optimisation.
struct PoseOptResult {
    Eigen::Matrix4d T_cw;    ///< Refined world-to-camera SE(3) (4×4).
    int             n_inliers{0};  ///< Edges classified as inliers after the
                                   ///  final outer iteration.
};

/// Optimise the camera pose given a set of fixed 3-D world points and their
/// stereo observations in the current frame.
///
/// Residual per observation: (u_l - u_l_proj, v - v_proj, u_r - u_r_proj)
///   where u_r_proj = u_l_proj - (fx * baseline) / z_c.
///
/// @param T_cw_init   Initial world-to-camera pose from PnP (4×4).
/// @param pts3d       Fixed 3-D world points (one per observation).
/// @param obs_stereo  Stereo observations in the current frame: each column
///                    is (u_l, v, u_r) for the matching point in pts3d.
/// @param cam         Camera calibration.
/// @param p           Optimiser parameters (optional, uses defaults).
/// @return            Refined pose and final inlier count.
PoseOptResult optimize_pose(
    const Eigen::Matrix4d&              T_cw_init,
    const std::vector<Eigen::Vector3d>& pts3d,
    const std::vector<Eigen::Vector3d>& obs_stereo,
    const StereoCamera&                 cam,
    const Params&                       p = {});

}  // namespace ba
}  // namespace sslam
