#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/frontend/feature_matcher.hpp"
#include "sslam/frontend/orb_extractor.hpp"
#include "sslam/frontend/stereo_matcher.hpp"
#include "sslam/tracking/motion_model.hpp"
#include "sslam/types/frame.hpp"

#include <opencv2/core.hpp>

#include <memory>

namespace sslam {

/// Tracking state for the current frame.
enum class TrackingState {
    NOT_INITIALIZED,  ///< No frame processed yet.
    OK,               ///< Pose estimated successfully.
    LOST,             ///< PnP failed; fewer than min_inliers_pnp inliers.
};

/// Frame-to-frame stereo visual odometry tracker.
///
/// Owns the full per-frame pipeline:
///   1. ORB extraction on left + right images.
///   2. Stereo L↔R matching — fills frame depth.
///   3. Constant-velocity pose prediction.
///   4. Frame-to-frame projection matching against the previous frame.
///   5. PnP RANSAC (EPnP initial guess, iterative refinement) → T_cw.
///   6. Motion model update.
///
/// Not thread-safe. Call process_frame() from a single thread.
class Tracking {
   public:
    struct Params {
        /// Minimum PnP inliers; fewer causes a LOST state.
        int   min_inliers_pnp{30};
        /// Reprojection error threshold in pixels for PnP RANSAC.
        float pnp_reprojection_error{2.0f};
        /// Maximum RANSAC iterations for PnP.
        int   pnp_max_iterations{100};
        /// RANSAC confidence level [0, 1].
        float pnp_confidence{0.99f};

        /// Radius multiplier for the wide-search retry.
        /// Applied when the narrow pass (radius_scale=1) yields fewer than
        /// min_inliers_pnp*2 matches, or when the motion model is not yet
        /// initialised (zero-velocity assumption can be far off).
        float wide_radius_scale{4.0f};

        ORBExtractor::Params   orb{};
        StereoMatcher::Params  stereo{};
        FeatureMatcher::Params feature{};
    };

    /// Per-frame tracking result.
    struct FrameResult {
        Frame::Ptr    frame;         ///< Processed frame with T_cw set.
        TrackingState state{TrackingState::NOT_INITIALIZED};
        int           n_stereo{0};   ///< Stereo L↔R matches (features with depth).
        int           n_matches{0};  ///< Frame-to-frame descriptor matches.
        int           n_inliers{0};  ///< PnP RANSAC inliers.
    };

    /// @param cam  Shared calibration; must not be null.
    explicit Tracking(std::shared_ptr<const StereoCamera> cam);
    Tracking(std::shared_ptr<const StereoCamera> cam, const Params& p);

    /// Process one stereo capture and return the result.
    ///
    /// @param idx    Monotonic frame index.
    /// @param ts     Timestamp (seconds).
    /// @param left   CV_8UC1 left image (must be already rectified).
    /// @param right  CV_8UC1 right image (must be already rectified).
    FrameResult process_frame(std::size_t idx, double ts,
                              const cv::Mat& left, const cv::Mat& right);

    TrackingState state() const { return state_; }

   private:
    std::shared_ptr<const StereoCamera> cam_;
    Params                              params_;
    ORBExtractor                        orb_extractor_;
    StereoMatcher                       stereo_matcher_;
    FeatureMatcher                      feature_matcher_;
    ConstantVelocityMotionModel         motion_model_;
    Frame::Ptr                          prev_frame_;
    TrackingState                       state_{TrackingState::NOT_INITIALIZED};
};

}  // namespace sslam
