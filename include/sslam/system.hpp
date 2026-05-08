#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

namespace sslam {

/// Write a trajectory in KITTI odometry format.
///
/// Each line contains the 12 values of the 3×4 world-to-camera *inverse*
/// (i.e. T_wc = T_cw^{-1}) stored row-major:
///   r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz
///
/// @param path       Output file path.  Parent directory must exist.
/// @param T_cw_vec   One 4×4 T_cw per frame (world → camera, SE(3)).
/// @throws std::runtime_error if the file cannot be opened.
void save_trajectory_kitti(const std::string&                  path,
                           const std::vector<Eigen::Matrix4d>& T_cw_vec);

}  // namespace sslam
