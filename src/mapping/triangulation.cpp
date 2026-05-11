#include "sslam/mapping/triangulation.hpp"

#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <cmath>

namespace sslam {

Eigen::Vector3d triangulate_linear(const Eigen::Matrix<double, 3, 4>& P1,
                                   const Eigen::Matrix<double, 3, 4>& P2,
                                   const Eigen::Vector2d& x1,
                                   const Eigen::Vector2d& x2) {
    // Build the 4×4 DLT system A w = 0.
    // Each observation contributes 2 rows:
    //   x * P^{3T} - P^{1T}
    //   y * P^{3T} - P^{2T}
    Eigen::Matrix4d A;
    A.row(0) = x1.x() * P1.row(2) - P1.row(0);
    A.row(1) = x1.y() * P1.row(2) - P1.row(1);
    A.row(2) = x2.x() * P2.row(2) - P2.row(0);
    A.row(3) = x2.y() * P2.row(2) - P2.row(1);

    // Solve via SVD; the solution is the last right singular vector.
    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(
        A, Eigen::ComputeFullV);
    const Eigen::Vector4d w = svd.matrixV().col(3);

    // Dehomogenise.
    return w.head<3>() / w(3);
}

bool check_triangulated(const Eigen::Matrix<double, 3, 4>& P1,
                        const Eigen::Matrix<double, 3, 4>& P2,
                        const Eigen::Matrix4d& T_cw1,
                        const Eigen::Matrix4d& T_cw2,
                        const Eigen::Vector3d& pw,
                        const Eigen::Vector2d& x1,
                        const Eigen::Vector2d& x2,
                        double max_reproj_err,
                        double min_parallax_deg) {
    // --- Positive depth in both cameras ------------------------------------
    const Eigen::Vector3d pc1 =
        T_cw1.block<3, 3>(0, 0) * pw + T_cw1.block<3, 1>(0, 3);
    const Eigen::Vector3d pc2 =
        T_cw2.block<3, 3>(0, 0) * pw + T_cw2.block<3, 1>(0, 3);
    if (pc1.z() <= 0.0 || pc2.z() <= 0.0) return false;

    // --- Parallax angle check ----------------------------------------------
    // Camera centres in world frame: -R^T t
    const Eigen::Vector3d c1 =
        -T_cw1.block<3, 3>(0, 0).transpose() * T_cw1.block<3, 1>(0, 3);
    const Eigen::Vector3d c2 =
        -T_cw2.block<3, 3>(0, 0).transpose() * T_cw2.block<3, 1>(0, 3);

    const Eigen::Vector3d ray1 = (pw - c1).normalized();
    const Eigen::Vector3d ray2 = (pw - c2).normalized();
    const double cos_angle = ray1.dot(ray2);
    // cos_angle close to 1 means nearly parallel rays (small parallax).
    constexpr double kDegToRad = M_PI / 180.0;
    if (cos_angle > std::cos(min_parallax_deg * kDegToRad)) return false;

    // --- Reprojection error in view 1 -------------------------------------
    const Eigen::Vector3d proj1 = P1 * pw.homogeneous();
    const Eigen::Vector2d rep1  = proj1.head<2>() / proj1.z();
    if ((rep1 - x1).norm() > max_reproj_err) return false;

    // --- Reprojection error in view 2 -------------------------------------
    const Eigen::Vector3d proj2 = P2 * pw.homogeneous();
    const Eigen::Vector2d rep2  = proj2.head<2>() / proj2.z();
    if ((rep2 - x2).norm() > max_reproj_err) return false;

    return true;
}

}  // namespace sslam
