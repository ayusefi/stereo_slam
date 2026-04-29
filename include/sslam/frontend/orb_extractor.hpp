#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <list>
#include <vector>

namespace sslam {

/// ORB feature extractor with image-pyramid + quadtree feature
/// distribution, mirroring ORB-SLAM2's `ORBextractor`.
///
/// Why not just `cv::ORB`? cv::ORB clusters keypoints in textured regions
/// and starves textureless ones. Even spatial coverage matters a lot for
/// PnP stability, so we:
///   1. Build an image pyramid (`num_levels`, `scale_factor`).
///   2. Detect FAST corners on every level (with a fallback threshold for
///      regions without enough texture at the higher one).
///   3. Distribute corners with a recursive quadtree to keep at most
///      `target_features_per_level` corners per level, evenly spread.
///   4. Compute orientation (intensity centroid) and the BRIEF-256
///      descriptor on the corresponding level image.
///
/// Output `keypoints` are returned in *original image* coordinates (with
/// `octave` set to the level they were detected on, so depth/scale code
/// downstream can recover the level if it needs to).
class ORBExtractor {
   public:
    struct Params {
        int   num_features{2000};
        int   num_levels{8};
        float scale_factor{1.2f};
        int   ini_fast_threshold{20};
        int   min_fast_threshold{7};
    };

    ORBExtractor() : ORBExtractor(Params{}) {}
    explicit ORBExtractor(const Params& p);

    /// Detect + describe. `image` must be CV_8UC1.
    void detect(const cv::Mat& image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::Mat& descriptors);

    const Params&            params()  const { return params_; }
    const std::vector<float>& scale_factors() const { return scale_factors_; }

   private:
    void build_pyramid(const cv::Mat& image);
    void compute_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypoints);
    std::vector<cv::KeyPoint> distribute_quadtree(
        const std::vector<cv::KeyPoint>& candidates,
        int min_x, int max_x, int min_y, int max_y, int target_n);

    Params              params_;
    std::vector<cv::Mat> pyramid_;
    std::vector<int>     features_per_level_;
    std::vector<float>   scale_factors_;
    std::vector<float>   inv_scale_factors_;
};

}  // namespace sslam
