#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace sslam {

/// Stereo L↔R feature matcher for rectified image pairs.
///
/// Algorithm (mirrors ORB-SLAM2 §III-B):
///   1. For each left keypoint, collect right candidates in a row band
///      [v - r, v + r] where r scales with the keypoint's octave.
///   2. Match ORB descriptors (Hamming) with Lowe ratio test.
///   3. Sub-pixel refine the right x-coordinate with an 11×11 SAD window
///      and a parabolic fit on the three columns around the integer match.
///   4. Reject disparities outside [min_disparity, max_disparity].
///   5. Write right_u[i] and depth[i] = fx·b / disp into the Frame.
class StereoMatcher {
   public:
    struct Params {
        /// Row tolerance (px) at octave 0; scaled by 1.2^octave for coarser levels.
        float row_tolerance{2.0f};
        /// Maximum Hamming distance to accept a descriptor match.
        /// ORB-SLAM2 uses TH_HIGH=100 for stereo (TH_LOW=50 is for tracking).
        int   hamming_threshold{100};
        /// Lowe ratio: reject if best_dist > ratio * second_best_dist.
        /// Only applied when a genuine second candidate exists in the row band;
        /// set to 1.0 to disable.
        float lowe_ratio{0.9f};
        /// Half-size of the SAD refinement window → (2·half+1)² patch.
        int   sad_win_half{5};
        /// Minimum valid disparity (px); matches below this are discarded.
        float min_disparity{0.0f};
        /// Maximum valid disparity (px); negative = auto (fx·b / 0.5 m).
        float max_disparity{-1.0f};
    };

    /// Construct with default Params.
    explicit StereoMatcher(std::shared_ptr<const StereoCamera> cam);
    /// Construct with custom Params.
    StereoMatcher(std::shared_ptr<const StereoCamera> cam, const Params& p);

    /// Populate frame.right_u and frame.depth.
    ///
    /// Prerequisites: frame.left, frame.right, frame.keypoints_left, and
    /// frame.descriptors_left must be filled before calling this.
    ///
    /// @param frame        Frame to write results into.
    /// @param right_kps    Keypoints detected in the right image.
    /// @param right_descs  Corresponding ORB descriptors (CV_8U).
    void match(Frame& frame,
               const std::vector<cv::KeyPoint>& right_kps,
               const cv::Mat& right_descs);

    const Params& params() const { return params_; }

   private:
    std::shared_ptr<const StereoCamera> cam_;
    Params params_;
};

}  // namespace sslam
