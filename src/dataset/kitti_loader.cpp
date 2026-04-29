#include "sslam/dataset/kitti_loader.hpp"

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sslam {

namespace fs = std::filesystem;

namespace {

// Parse one KITTI calib.txt line of the form
//   "P0: a b c d e f g h i j k l"
// into a 3x4 matrix. Returns true on success.
bool parse_proj_line(const std::string& line, const std::string& tag,
                     Eigen::Matrix<double, 3, 4>& P) {
    if (line.compare(0, tag.size(), tag) != 0) return false;
    std::istringstream iss(line.substr(tag.size()));
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            if (!(iss >> P(r, c))) return false;
    return true;
}

}  // namespace

KittiLoader::KittiLoader(const fs::path& sequence_dir) : dir_(sequence_dir) {
    if (!fs::is_directory(dir_)) {
        throw std::runtime_error("KittiLoader: not a directory: " + dir_.string());
    }

    // --- calib.txt ---
    Eigen::Matrix<double, 3, 4> P0 = Eigen::Matrix<double, 3, 4>::Zero();
    Eigen::Matrix<double, 3, 4> P1 = Eigen::Matrix<double, 3, 4>::Zero();
    {
        std::ifstream cf(dir_ / "calib.txt");
        if (!cf) throw std::runtime_error("KittiLoader: missing calib.txt");
        std::string line;
        bool got_p0 = false, got_p1 = false;
        while (std::getline(cf, line)) {
            if (parse_proj_line(line, "P0:", P0)) got_p0 = true;
            else if (parse_proj_line(line, "P1:", P1)) got_p1 = true;
        }
        if (!got_p0 || !got_p1) {
            throw std::runtime_error("KittiLoader: calib.txt missing P0/P1");
        }
    }

    camera_.fx = P0(0, 0);
    camera_.fy = P0(1, 1);
    camera_.cx = P0(0, 2);
    camera_.cy = P0(1, 2);
    // For rectified stereo P1 = K [I | t] with t.x = -fx * baseline.
    camera_.baseline = -P1(0, 3) / P1(0, 0);
    if (camera_.baseline <= 0.0) {
        throw std::runtime_error("KittiLoader: non-positive baseline parsed");
    }

    // --- times.txt ---
    {
        std::ifstream tf(dir_ / "times.txt");
        if (!tf) throw std::runtime_error("KittiLoader: missing times.txt");
        double t;
        while (tf >> t) timestamps_.push_back(t);
        if (timestamps_.empty()) {
            throw std::runtime_error("KittiLoader: times.txt empty");
        }
    }

    // --- peek first image to learn resolution ---
    cv::Mat first = cv::imread((dir_ / "image_0" / "000000.png").string(),
                               cv::IMREAD_GRAYSCALE);
    if (first.empty()) {
        throw std::runtime_error("KittiLoader: failed to read image_0/000000.png");
    }
    camera_.width  = first.cols;
    camera_.height = first.rows;
}

StereoFrame KittiLoader::load(std::size_t i) const {
    if (i >= timestamps_.size()) {
        throw std::out_of_range("KittiLoader::load index out of range");
    }

    std::ostringstream name;
    name << std::setw(6) << std::setfill('0') << i << ".png";

    StereoFrame f;
    f.index     = i;
    f.timestamp = timestamps_[i];
    f.left      = cv::imread((dir_ / "image_0" / name.str()).string(), cv::IMREAD_GRAYSCALE);
    f.right     = cv::imread((dir_ / "image_1" / name.str()).string(), cv::IMREAD_GRAYSCALE);

    if (f.left.empty() || f.right.empty()) {
        throw std::runtime_error("KittiLoader: failed to read frame " + name.str());
    }
    return f;
}

}  // namespace sslam
