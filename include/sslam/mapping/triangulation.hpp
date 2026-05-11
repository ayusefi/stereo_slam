#pragma once

#include <Eigen/Core>

namespace sslam {

/// Linear (DLT) triangulation of a single 3-D point from two views.
///
/// Given projection matrices P1, P2 and normalised image points x1, x2,
/// builds the 4×4 DLT system A and returns the dehomogenised world point.
///
/// @param P1  3×4 camera projection matrix for view 1 (K [R1 | t1]).
/// @param P2  3×4 camera projection matrix for view 2 (K [R2 | t2]).
/// @param x1  Pixel observation in view 1 (u, v).
/// @param x2  Pixel observation in view 2 (u, v).
/// @return    3-D world point w.
Eigen::Vector3d triangulate_linear(const Eigen::Matrix<double, 3, 4>& P1,
                                   const Eigen::Matrix<double, 3, 4>& P2,
                                   const Eigen::Vector2d& x1,
                                   const Eigen::Vector2d& x2);

/// Quality gate for an already-triangulated point.
///
/// Checks positive depth in both cameras, minimum parallax angle, and
/// reprojection error in both views.
///
/// @param P1, P2        Projection matrices (same as triangulate_linear).
/// @param T_cw1, T_cw2  World-to-camera transforms (4×4 SE3).
/// @param pw            Candidate 3-D world point.
/// @param x1, x2        Pixel observations (same as triangulate_linear).
/// @param max_reproj_err  Maximum reprojection error in pixels (default 2 px).
/// @param min_parallax_deg  Minimum parallax angle in degrees (default 1°).
/// @return true if the point passes all checks.
bool check_triangulated(const Eigen::Matrix<double, 3, 4>& P1,
                        const Eigen::Matrix<double, 3, 4>& P2,
                        const Eigen::Matrix4d& T_cw1,
                        const Eigen::Matrix4d& T_cw2,
                        const Eigen::Vector3d& pw,
                        const Eigen::Vector2d& x1,
                        const Eigen::Vector2d& x2,
                        double max_reproj_err   = 2.0,
                        double min_parallax_deg = 1.0);

}  // namespace sslam
