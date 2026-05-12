/// test_sim3_solver.cpp
///
/// Acceptance criteria:
///   - On a synthetic Sim3 (known s, R, t) with σ=5 cm noise + 30% outliers,
///     the solver recovers s, R, t to < 1% relative error (ROADMAP §5.2).
///   - With too few points (< 20), solve() returns found=false.

#include "sslam/optim/sim3_solver.hpp"

#include <gtest/gtest.h>
#include <Eigen/Geometry>

#include <cmath>
#include <random>
#include <vector>

namespace {

constexpr int    kNPoints   = 200;
constexpr double kNoiseSigma = 0.05;  // metres
constexpr double kOutlierFrac = 0.30;
constexpr double kMaxRelErr   = 0.01;  // 1 %

/// Build a random unit SO(3) rotation from axis-angle with given angle.
Eigen::Matrix3d random_rotation(std::mt19937& rng, double angle_rad) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    Eigen::Vector3d axis(d(rng), d(rng), d(rng));
    axis.normalize();
    return Eigen::AngleAxisd(angle_rad, axis).toRotationMatrix();
}

TEST(Sim3Solver, RecoversSyntheticSim3) {
    std::mt19937 rng(42);
    std::normal_distribution<double>  noise(0.0, kNoiseSigma);
    std::uniform_real_distribution<double> outlier(-10.0, 10.0);
    std::uniform_real_distribution<double> coord(-5.0, 5.0);

    // Ground-truth Sim3: p2 = s * R * p1 + t
    const double         s_gt = 1.15;
    const Eigen::Matrix3d R_gt = random_rotation(rng, 0.3);
    const Eigen::Vector3d t_gt(0.5, -0.3, 0.2);

    std::vector<Eigen::Vector3d> pts1, pts2;
    pts1.reserve(kNPoints);
    pts2.reserve(kNPoints);

    const int n_outliers = static_cast<int>(kNPoints * kOutlierFrac);

    for (int i = 0; i < kNPoints; ++i) {
        Eigen::Vector3d p1(coord(rng), coord(rng), coord(rng));
        pts1.push_back(p1);

        if (i < n_outliers) {
            pts2.emplace_back(outlier(rng), outlier(rng), outlier(rng));
        } else {
            Eigen::Vector3d p2 = s_gt * R_gt * p1 + t_gt;
            p2 += Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
            pts2.push_back(p2);
        }
    }

    sslam::Sim3Solver solver(pts1, pts2);
    const auto result = solver.solve();

    ASSERT_TRUE(result.found) << "Sim3 solver did not find a solution";
    EXPECT_GE(result.n_inliers, 20);

    // Scale error.
    const double scale_err = std::abs(result.scale - s_gt) / s_gt;
    EXPECT_LT(scale_err, kMaxRelErr)
        << "Scale error " << scale_err * 100.0 << "% > 1%";

    // Rotation error (Frobenius norm of R^T R_est - I).
    const double rot_err = (result.R.transpose() * R_gt -
                            Eigen::Matrix3d::Identity()).norm();
    EXPECT_LT(rot_err, kMaxRelErr * 10.0)
        << "Rotation Frobenius error " << rot_err;

    // Translation error (relative to scale * ||t_gt||).
    const double t_err =
        (result.t - t_gt).norm() / (s_gt * (1.0 + t_gt.norm()));
    EXPECT_LT(t_err, kMaxRelErr)
        << "Translation relative error " << t_err * 100.0 << "%";
}

TEST(Sim3Solver, RejectsTooFewPoints) {
    std::vector<Eigen::Vector3d> pts1(10, Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> pts2(10, Eigen::Vector3d::Ones());
    sslam::Sim3Solver solver(pts1, pts2);
    const auto result = solver.solve();
    EXPECT_FALSE(result.found);
}

}  // namespace
