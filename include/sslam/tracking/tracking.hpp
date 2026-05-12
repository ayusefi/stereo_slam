#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/frontend/feature_matcher.hpp"
#include "sslam/frontend/orb_extractor.hpp"
#include "sslam/frontend/stereo_matcher.hpp"
#include "sslam/mapping/local_mapping.hpp"
#include "sslam/tracking/motion_model.hpp"
#include "sslam/types/frame.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

        // --- KeyFrame insertion policy -----------------------------------
        /// Insert a KF after this many frames since the last one.
        int   kf_max_frames_since{8};
        /// Insert a KF if PnP inliers drop below this (degraded tracking).
        int   kf_min_tracked_points{50};

        /// Sanity cap on per-frame camera-centre translation (metres).
        /// If the estimated pose implies a larger jump, the frame is
        /// treated as LOST instead of committing the outlier solution.
        /// 5 m covers 180 km/h at KITTI's 10 Hz capture rate.
        float max_frame_translation{5.0f};

        /// Minimum number of local-map MP matches for TrackLocalMap to
        /// take effect.  Below this threshold, fall back to frame-to-frame
        /// projection matching so the first few frames work correctly.
        int min_local_map_matches{30};

        ORBExtractor::Params   orb{};
        StereoMatcher::Params  stereo{};
        FeatureMatcher::Params feature{};
    };

    /// Per-frame tracking result.
    struct FrameResult {
        Frame::Ptr    frame;         ///< Processed frame with T_cw set.
        TrackingState state{TrackingState::NOT_INITIALIZED};
        int           n_stereo{0};   ///< Stereo L↔R matches (features with depth).
        int           n_matches{0};  ///< Correspondences used for PnP (local-map MPs or frame-to-frame matches).
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

    /// Access the persistent map built during tracking.
    const Map::Ptr& map() const { return map_; }

    /// Access the local mapping thread (non-null after construction).
    const LocalMapping::Ptr& local_mapping() const { return local_mapping_; }

    /// Resolved per-frame trajectory in world frame.
    /// Each entry is reconstructed as `T_ref * ref_kf->get_pose()` so any
    /// Local BA correction applied to the reference KeyFrame is reflected
    /// in the returned poses.  Frames recorded before the first KeyFrame
    /// existed (only the very first frame, in practice) fall back to
    /// their captured `T_cw`.
    std::vector<Eigen::Matrix4d> resolved_trajectory() const;

   private:
    std::shared_ptr<const StereoCamera> cam_;
    Params                              params_;
    ORBExtractor                        orb_extractor_;
    StereoMatcher                       stereo_matcher_;
    FeatureMatcher                      feature_matcher_;
    ConstantVelocityMotionModel         motion_model_;
    Frame::Ptr                          prev_frame_;
    TrackingState                       state_{TrackingState::NOT_INITIALIZED};

    // --- Map and KeyFrame insertion state --------------------------------
    Map::Ptr          map_;
    LocalMapping::Ptr local_mapping_;
    KeyFrame::Ptr     last_kf_;
    Eigen::Matrix4d   last_kf_raw_T_cw_{Eigen::Matrix4d::Identity()};
    uint64_t      next_kf_id_{0};
    uint64_t      last_kf_frame_idx_{~uint64_t{0}};
    int           nframes_since_kf_{0};

    // Per-frame (ref_kf, T_ref) snapshots for resolved_trajectory().
    struct TrajectoryEntry {
        KeyFrame*        ref_kf;        ///< Non-owning; KF lives in Map.
        Eigen::Matrix4d  T_ref;         ///< T_cw in ref_kf's camera frame.
        Eigen::Matrix4d  T_cw_fallback; ///< Used when ref_kf is null.
    };
    std::vector<TrajectoryEntry> trajectory_entries_;

    /// Insert a KeyFrame if insertion criteria are met.
    /// Must be called before updating prev_frame_.
    void maybe_insert_keyframe(
        const Frame& curr_frame,
        const std::vector<std::pair<int, int>>& matches,
        int n_inliers);

    /// Anchor `frame` to the latest KeyFrame (sets `ref_kf` and `T_ref`)
    /// and append a `TrajectoryEntry` for it.  Trailing underscore marks
    /// it as private mutator.
    void anchor_and_record_(Frame& frame);

    /// Project local MapPoints (from last_kf_ and its covisible neighbours)
    /// into `frame` using the predicted pose `T_pred`.  For each projected
    /// MP, search within a fixed pixel radius for the best-matching
    /// unmatched feature (Hamming + Lowe ratio).  On success, appends to
    /// pts3d / pts2d / obs_stereo and returns true when the total number of
    /// matches is >= params_.min_local_map_matches.
    ///
    /// When this returns false the caller should fall back to frame-to-frame
    /// matching (e.g. the first few frames before a covisibility graph exists).
    bool match_local_map_(const Frame& frame,
                          const Eigen::Matrix4d& T_pred,
                          std::vector<cv::Point3f>& pts3d,
                          std::vector<cv::Point2f>& pts2d,
                          std::vector<Eigen::Vector3d>& obs_stereo);
};

}  // namespace sslam
