#pragma once

#include "sslam/camera/stereo_camera.hpp"

#include <opencv2/core.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace sslam {

struct StereoFrame {
    std::size_t index{0};
    double      timestamp{0.0};   // seconds
    cv::Mat     left;             // CV_8UC1 (KITTI grayscale image_0)
    cv::Mat     right;            // CV_8UC1 (KITTI grayscale image_1)
};

/// Reads a KITTI Odometry stereo sequence directory:
///   <seq>/calib.txt
///   <seq>/times.txt
///   <seq>/image_0/000000.png ...   (left grayscale, already rectified)
///   <seq>/image_1/000000.png ...   (right grayscale, already rectified)
class KittiLoader {
   public:
    explicit KittiLoader(const std::filesystem::path& sequence_dir);

    const StereoCamera& camera()    const { return camera_; }
    std::size_t         size()      const { return timestamps_.size(); }

    /// Loads frame `i` from disk. Throws on out-of-range or missing image.
    StereoFrame load(std::size_t i) const;

   private:
    std::filesystem::path     dir_;
    StereoCamera              camera_;
    std::vector<double>       timestamps_;
};

}  // namespace sslam
