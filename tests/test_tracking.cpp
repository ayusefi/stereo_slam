#include "sslam/tracking/motion_model.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

// Build a pure-translation SE(3): camera at world position (tx, ty, tz).
// For a camera at world position p_wc, T_cw has t_cw = -p_wc (assuming R=I).
Eigen::Matrix4d translation_pose(double tx, double ty, double tz) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = tx;
    T(1, 3) = ty;
    T(2, 3) = tz;
    return T;
}

// Build a pure-rotation SE(3) about the Y axis by angle_rad.
Eigen::Matrix4d rotation_y_pose(double angle_rad) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    T(0, 0) =  c; T(0, 2) = s;
    T(2, 0) = -s; T(2, 2) = c;
    return T;
}

}  // namespace

// ---------------------------------------------------------------------------
// ConstantVelocityMotionModel tests
// ---------------------------------------------------------------------------

TEST(ConstantVelocityMotionModel, IsInvalidBeforeFirstUpdate) {
    sslam::ConstantVelocityMotionModel mm;
    EXPECT_FALSE(mm.is_valid());
}

TEST(ConstantVelocityMotionModel, HoldsLastPoseWhenInvalid) {
    // Before update(), predict() must return the input pose unchanged.
    sslam::ConstantVelocityMotionModel mm;
    const Eigen::Matrix4d T = translation_pose(-0.1, 0.0, 0.0);
    const Eigen::Matrix4d pred = mm.predict(T);
    EXPECT_TRUE(pred.isApprox(T, 1e-9));
}

TEST(ConstantVelocityMotionModel, IsValidAfterUpdate) {
    sslam::ConstantVelocityMotionModel mm;
    mm.update(translation_pose(-0.1, 0, 0), translation_pose(0, 0, 0));
    EXPECT_TRUE(mm.is_valid());
}

TEST(ConstantVelocityMotionModel, PredictsPureTranslation) {
    // Camera moves 0.1 m in world X each step (T_cw.t advances by -0.1 in X).
    // T_prev_cw = [I | 0 ]   (cam at world origin → t_cw = 0)
    // T_curr_cw = [I | -0.1] (cam at world X=0.1  → t_cw = -0.1)
    // Velocity = T_curr_cw * inv(T_prev_cw) = T_curr_cw * I = T_curr_cw
    // Prediction: T_next = velocity * T_curr_cw = [I | -0.2]
    sslam::ConstantVelocityMotionModel mm;
    const Eigen::Matrix4d T_prev = translation_pose(0.0, 0.0, 0.0);
    const Eigen::Matrix4d T_curr = translation_pose(-0.1, 0.0, 0.0);
    mm.update(T_curr, T_prev);

    const Eigen::Matrix4d T_pred = mm.predict(T_curr);
    const Eigen::Matrix4d T_expected = translation_pose(-0.2, 0.0, 0.0);
    EXPECT_TRUE(T_pred.isApprox(T_expected, 1e-9))
        << "predicted:\n" << T_pred << "\nexpected:\n" << T_expected;
}

TEST(ConstantVelocityMotionModel, PredictsPureRotation) {
    // Camera rotates 5° around Y per step.
    const double deg5 = 5.0 * M_PI / 180.0;
    const Eigen::Matrix4d T_prev = rotation_y_pose(0.0);
    const Eigen::Matrix4d T_curr = rotation_y_pose(deg5);

    sslam::ConstantVelocityMotionModel mm;
    mm.update(T_curr, T_prev);

    const Eigen::Matrix4d T_pred     = mm.predict(T_curr);
    const Eigen::Matrix4d T_expected = rotation_y_pose(2.0 * deg5);

    EXPECT_TRUE(T_pred.isApprox(T_expected, 1e-9))
        << "predicted:\n" << T_pred << "\nexpected:\n" << T_expected;
}

TEST(ConstantVelocityMotionModel, MultipleUpdatesReflectLatestVelocity) {
    // First motion: 0.1 m/step in X.
    // Second motion: 0.3 m/step in X. Prediction must use the second velocity.
    sslam::ConstantVelocityMotionModel mm;
    mm.update(translation_pose(-0.1, 0, 0), translation_pose(0, 0, 0));
    mm.update(translation_pose(-0.4, 0, 0), translation_pose(-0.1, 0, 0));

    const Eigen::Matrix4d T_curr     = translation_pose(-0.4, 0, 0);
    const Eigen::Matrix4d T_pred     = mm.predict(T_curr);
    const Eigen::Matrix4d T_expected = translation_pose(-0.7, 0, 0);

    EXPECT_TRUE(T_pred.isApprox(T_expected, 1e-9))
        << "predicted:\n" << T_pred << "\nexpected:\n" << T_expected;
}
