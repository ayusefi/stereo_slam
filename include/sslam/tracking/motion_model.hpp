#pragma once

#include <Eigen/Core>

namespace sslam {

/// Constant-velocity SE(3) motion model for frame-to-frame pose prediction.
///
/// Stores the relative transform (velocity) between the last two frames and
/// extrapolates it to predict the next pose.
///
/// Convention: T_cw = [R | t] (4×4) maps world points into the camera frame.
/// Velocity is stored as T_c2_c1 = T_c2w * T_wc1.
class ConstantVelocityMotionModel {
   public:
    ConstantVelocityMotionModel() = default;

    /// Record the transition T_prev_cw → T_curr_cw and store the velocity.
    /// @param T_curr_cw  Current world-to-camera SE(3) (4×4).
    /// @param T_prev_cw  Previous world-to-camera SE(3) (4×4).
    void update(const Eigen::Matrix4d& T_curr_cw,
                const Eigen::Matrix4d& T_prev_cw);

    /// Predict the next world-to-camera pose from the stored velocity.
    ///
    /// Returns @p T_curr_cw unchanged when is_valid() is false (zero-velocity
    /// assumption — holds the last known position on the first frame pair).
    ///
    /// @param T_curr_cw  Most recent known world-to-camera pose (4×4).
    Eigen::Matrix4d predict(const Eigen::Matrix4d& T_curr_cw) const;

    /// True after at least one call to update().
    bool is_valid() const { return valid_; }

    /// Drop the stored velocity so the next predict() returns its input
    /// unchanged.  Use after a tracking failure where the previous velocity
    /// estimate is known to be stale.
    void reset() {
        velocity_ = Eigen::Matrix4d::Identity();
        valid_    = false;
    }

   private:
    /// T_c2_c1 — relative transform from frame c1 to frame c2.
    Eigen::Matrix4d velocity_{Eigen::Matrix4d::Identity()};
    bool            valid_{false};
};

}  // namespace sslam
