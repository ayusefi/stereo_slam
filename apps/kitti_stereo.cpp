// Phase 1 demo: load KITTI, run our ORB extractor on every left frame,
// and visualise the keypoints. No tracking yet — that is Phase 1.3+.
//
// Usage:  ./kitti_stereo /path/to/kitti/sequences/00 [--no-display]

#include "sslam/dataset/kitti_loader.hpp"
#include "sslam/frontend/orb_extractor.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <kitti-sequence-dir> [--no-display]\n";
        return 1;
    }
    const bool display = !(argc >= 3 && std::string(argv[2]) == "--no-display");

    try {
        sslam::KittiLoader loader(argv[1]);
        const auto& cam = loader.camera();
        std::cout << "Loaded KITTI sequence: " << argv[1] << "\n"
                  << "  frames   : " << loader.size() << "\n"
                  << "  size     : " << cam.width << "x" << cam.height << "\n"
                  << "  fx,fy    : " << cam.fx << ", " << cam.fy << "\n"
                  << "  cx,cy    : " << cam.cx << ", " << cam.cy << "\n"
                  << "  baseline : " << cam.baseline << " m\n";

        sslam::ORBExtractor extractor;

        const std::string win = "sslam :: ORB keypoints (left)";
        if (display) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

        double total_ms = 0.0;
        std::size_t total_kps = 0;

        for (std::size_t i = 0; i < loader.size(); ++i) {
            auto frame = loader.load(i);

            std::vector<cv::KeyPoint> kps;
            cv::Mat desc;
            const auto t0 = std::chrono::steady_clock::now();
            extractor.detect(frame.left, kps, desc);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms  += ms;
            total_kps += kps.size();

            if (display) {
                cv::Mat vis;
                cv::drawKeypoints(frame.left, kps, vis, {0, 255, 0},
                                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                cv::putText(vis,
                            "frame " + std::to_string(i) +
                                "  kps " + std::to_string(kps.size()) +
                                "  " + std::to_string(static_cast<int>(ms)) + " ms",
                            {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            {0, 255, 0}, 2);
                cv::imshow(win, vis);
                if (const int k = cv::waitKey(1); k == 27 || k == 'q') break;
            }
        }
        std::cout << "Average ORB extract: "
                  << (total_ms / loader.size()) << " ms/frame, "
                  << (total_kps / loader.size()) << " kps/frame\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
