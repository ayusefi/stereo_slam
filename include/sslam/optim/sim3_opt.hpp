#pragma once

#include "sslam/camera/stereo_camera.hpp"

#include <Eigen/Core>

#include <vector>

namespace sslam {
namespace optim {

struct Sim3OptResult {
    double            scale{1.0};
    Eigen::Matrix3d   R;
    Eigen::Vector3d   t;
    int               n_inliers{0};
    std::vector<bool> inlier_mask;  ///< One entry per input correspondence.
};

/// Refine a Sim3 estimate from RANSAC using bidirectional
/// EdgeSim3ProjectXYZ g2o optimisation (ORB-SLAM2 OptimizeSim3).
///
/// Conventions (same as Sim3Solver):
///   p_m_world = s * R * p_q_world + t   (maps query-world → match-world)
///
/// The g2o problem has:
///   - One free VertexSim3Expmap (query-cam → match-cam Sim3, fix_scale=true
///     for stereo, so scale is constrained to 1).
///   - Fixed VertexPointXYZ per inlier (stored in CAMERA frame).
///   - EdgeSim3ProjectXYZ    (forward):  query-cam point → project in match cam.
///   - EdgeInverseSim3ProjectXYZ (back): match-cam point → project in query cam.
///   - Huber kernel delta = sqrt(10.0).
///
/// Two rounds: 5 LM iters with Huber, classify outliers by chi2 > 9.210*sigma2,
/// then 5 more iters without outliers and without robust kernel.
///
/// @param pts_q_w  World-frame points from query KF (one per correspondence).
/// @param pts_m_w  World-frame points from match KF.
/// @param obs_q    2-D observations in query KF  (u, v).
/// @param obs_m    2-D observations in match KF.
/// @param sigma2_q Per-octave sigma^2 values for query (1.2^(2*octave)).
/// @param sigma2_m Per-octave sigma^2 values for match.
/// @param init_mask Which correspondences to use (from RANSAC inlier_mask).
/// @param T_cw_q   World-to-camera pose of query KF.
/// @param T_cw_m   World-to-camera pose of match KF.
/// @param s, R, t  Initial Sim3 from RANSAC (maps query-world → match-world).
/// @param cam      Shared camera calibration.
/// @param fix_scale If true, hold scale = 1 throughout (use for stereo/RGB-D).
Sim3OptResult optimize_sim3(
    const std::vector<Eigen::Vector3d>& pts_q_w,
    const std::vector<Eigen::Vector3d>& pts_m_w,
    const std::vector<Eigen::Vector2d>& obs_q,
    const std::vector<Eigen::Vector2d>& obs_m,
    const std::vector<double>& sigma2_q,
    const std::vector<double>& sigma2_m,
    const std::vector<bool>& init_mask,
    const Eigen::Matrix4d& T_cw_q,
    const Eigen::Matrix4d& T_cw_m,
    double s, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const StereoCamera& cam,
    bool fix_scale = true);

}  // namespace optim
}  // namespace sslam
