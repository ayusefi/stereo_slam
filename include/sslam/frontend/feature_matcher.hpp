#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"

#include <Eigen/Core>

#include <memory>
#include <utility>
#include <vector>

namespace sslam {

/// Frame-to-frame feature matcher based on map-point projection.
///
/// Given a previous Frame whose keypoints have been stereo-triangulated
/// (depth > 0), each 3-D point is projected into the current frame using a
/// predicted pose. The best descriptor match within a scaled search radius
/// is accepted if it passes the Hamming threshold and the Lowe ratio test.
///
/// This is the "match by projection" path used during normal tracking.
/// The "match by BoW" path (for relocalization) is added in a later phase.
class FeatureMatcher {
   public:
    struct Params {
        /// Search radius (px) at octave 0; multiplied by scale_factor^octave.
        float search_radius{10.0f};
        /// Must match the ORBExtractor scale_factor used to build the frames.
        float scale_factor{1.2f};
        /// Maximum Hamming distance (TH_HIGH = 100 for ORB).
        int   hamming_threshold{100};
        /// Lowe ratio; applied only when a genuine second candidate exists.
        float lowe_ratio{0.9f};
    };

    explicit FeatureMatcher(std::shared_ptr<const StereoCamera> cam);
    FeatureMatcher(std::shared_ptr<const StereoCamera> cam, const Params& p);

    /// Match stereo-triangulated points from @p prev into @p curr by projection.
    ///
    /// For each keypoint i in @p prev with depth[i] > 0, the 3-D point is
    /// unprojected from prev's camera frame to world (using prev.T_cw), then
    /// projected into curr using @p T_curr_cw. The best ORB descriptor match
    /// within a radius of `search_radius * scale_factor^octave * radius_scale`
    /// px is kept if it passes the Hamming threshold and Lowe ratio test.
    ///
    /// @param prev          Previous frame (must have depth filled).
    /// @param curr          Current frame (must have keypoints_left / descriptors_left).
    /// @param T_curr_cw     Predicted world-to-camera SE(3) for @p curr (4×4).
    /// @param radius_scale  Multiplier on the per-octave search radius.
    ///                      Use > 1 for a wider pass when the motion model
    ///                      is unavailable or a narrow pass returned too few
    ///                      matches (e.g. 4.0 gives ~4× the search area).
    /// @return              (prev_idx, curr_idx) pairs for each accepted match.
    std::vector<std::pair<int, int>> match_by_projection(
        const Frame& prev,
        const Frame& curr,
        const Eigen::Matrix4d& T_curr_cw,
        float radius_scale = 1.0f) const;

    const Params& params() const { return params_; }

   private:
    std::shared_ptr<const StereoCamera> cam_;
    Params params_;
};

}  // namespace sslam
