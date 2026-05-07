// Motion-only BA acceptance test.
//
// Synthesises 200 3-D world points, projects them with a known SE(3) pose,
// adds independent Gaussian pixel noise (σ = 1 px), and randomly flips 10%
// of observations to uniformly random pixels (outliers).  Runs optimize_pose
// and checks that the recovered pose is within 0.5° / 0.05 m of the truth.

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/optim/ba.hpp"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace {

/// Rotation matrix from axis-angle (Rodrigues).
Eigen::Matrix3d rot(const Eigen::Vector3d& axis, double angle_rad) {
    return Eigen::AngleAxisd(angle_rad, axis.normalized()).toRotationMatrix();
}

/// Build a synthetic StereoCamera.
sslam::StereoCamera make_cam() {
    sslam::StereoCamera c;
    c.fx       = 718.856;
    c.fy       = 718.856;
    c.cx       = 607.193;
    c.cy       = 185.216;
    c.baseline = 0.537166;
    c.width    = 1241;
    c.height   = 376;
    return c;
}

/// Angle between two rotation matrices (degrees).
double rot_error_deg(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
    const Eigen::Matrix3d dR   = Ra.transpose() * Rb;
    const double          cos_ = (dR.trace() - 1.0) * 0.5;
    return std::acos(std::clamp(cos_, -1.0, 1.0)) * 180.0 / M_PI;
}

}  // namespace

TEST(BAOptimizePose, RecoversPoseFromNoisyOutlierObservations) {
    const sslam::StereoCamera cam = make_cam();
    const double bf = cam.fx * cam.baseline;

    // Ground-truth T_cw: small rotation + 0.3 m translation
    Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
    T_gt.topLeftCorner<3, 3>()  = rot({0, 1, 0}, 0.05);   // ~3° around Y
    T_gt.topRightCorner<3, 1>() = Eigen::Vector3d(0.3, 0.0, 0.0);

    // Initial guess: identity (no prior knowledge)
    Eigen::Matrix4d T_init = Eigen::Matrix4d::Identity();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> unif_x(50.0, cam.width - 50.0);
    std::uniform_real_distribution<double> unif_y(30.0, cam.height - 30.0);
    std::uniform_real_distribution<double> unif_z(5.0, 30.0);
    std::normal_distribution<double>       noise(0.0, 1.0);

    constexpr int   N_POINTS  = 200;
    constexpr double OUTLIER_FRACTION = 0.1;

    std::vector<Eigen::Vector3d> pts3d;
    std::vector<Eigen::Vector3d> obs_stereo;
    pts3d.reserve(N_POINTS);
    obs_stereo.reserve(N_POINTS);

    for (int i = 0; i < N_POINTS; ++i) {
        // Sample a random 3-D world point visible in the camera.
        // Project back through T_gt to get a point in the camera frame.
        const double u = unif_x(rng);
        const double v = unif_y(rng);
        const double z = unif_z(rng);

        // p_c (camera frame under T_gt)
        const Eigen::Vector3d p_c(
            (u - cam.cx) * z / cam.fx,
            (v - cam.cy) * z / cam.fy,
            z);

        // T_wc (inverse of T_gt)
        const Eigen::Matrix3d R_gt = T_gt.topLeftCorner<3, 3>();
        const Eigen::Vector3d t_gt = T_gt.topRightCorner<3, 1>();
        const Eigen::Vector3d p_w  = R_gt.transpose() * (p_c - t_gt);
        pts3d.push_back(p_w);

        // Clean stereo observation
        const double u_l = u;
        const double v_l = v;
        const double u_r = u_l - bf / z;

        // Add pixel noise
        const double u_l_n = u_l + noise(rng);
        const double v_n   = v_l + noise(rng);
        const double u_r_n = u_r + noise(rng);

        // 10% outliers: replace with random pixel
        if (static_cast<double>(i) / N_POINTS < OUTLIER_FRACTION) {
            obs_stereo.push_back({unif_x(rng), unif_y(rng),
                                  unif_x(rng) - 50.0});
        } else {
            obs_stereo.push_back({u_l_n, v_n, u_r_n});
        }
    }

    const auto result = sslam::ba::optimize_pose(T_init, pts3d, obs_stereo, cam);

    const Eigen::Matrix3d R_rec = result.T_cw.topLeftCorner<3, 3>();
    const Eigen::Vector3d t_rec = result.T_cw.topRightCorner<3, 1>();
    const Eigen::Matrix3d R_gt  = T_gt.topLeftCorner<3, 3>();
    const Eigen::Vector3d t_gt  = T_gt.topRightCorner<3, 1>();

    const double rot_err = rot_error_deg(R_gt, R_rec);
    const double t_err   = (t_gt - t_rec).norm();

    EXPECT_LT(rot_err, 0.5)   << "rotation error " << rot_err << "°";
    EXPECT_LT(t_err,   0.05)  << "translation error " << t_err << " m";
    EXPECT_GE(result.n_inliers, 150);  // at least 75% of the 180 clean points
}
