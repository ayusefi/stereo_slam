// KITTI stereo visual odometry demo.
//
// Runs the full frontend pipeline on every frame:
//   ORB extraction → stereo L↔R matching → PnP tracking → pose estimate.
//
// Visualisation:
//   Green  circle  — left keypoint WITH a stereo match (has depth)
//   Red    circle  — left keypoint WITHOUT a stereo match (mono only)
//   Yellow line    — connects left match to its right-image x on the same row
//
// Per-frame stats printed to stdout: state, stereo %, fm matches, PnP inliers.
//
// Usage:  ./kitti_stereo <kitti-sequence-dir> [--no-display]

#include "sslam/dataset/kitti_loader.hpp"
#include "sslam/tracking/tracking.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <kitti-sequence-dir> [--no-display]\n";
        return 1;
    }
    const bool display = !(argc >= 3 && std::string(argv[2]) == "--no-display");

    try {
        sslam::KittiLoader loader(argv[1]);
        const auto cam_ptr =
            std::make_shared<const sslam::StereoCamera>(loader.camera());

        std::cout << "Loaded KITTI sequence: " << argv[1] << "\n"
                  << "  frames   : " << loader.size() << "\n"
                  << "  size     : " << cam_ptr->width << "x" << cam_ptr->height << "\n"
                  << "  fx,fy    : " << cam_ptr->fx << ", " << cam_ptr->fy << "\n"
                  << "  cx,cy    : " << cam_ptr->cx << ", " << cam_ptr->cy << "\n"
                  << "  baseline : " << cam_ptr->baseline << " m\n";

        sslam::Tracking tracker(cam_ptr);

        const std::string win = "sslam :: stereo VO";
        if (display) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

        double      total_ms      = 0.0;
        double      total_s_ratio = 0.0;
        std::size_t n_lost        = 0;

        for (std::size_t i = 0; i < loader.size(); ++i) {
            const auto raw = loader.load(i);

            const auto t0 = std::chrono::steady_clock::now();
            const auto result = tracker.process_frame(
                i, raw.timestamp, raw.left, raw.right);
            const auto t1 = std::chrono::steady_clock::now();

            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;

            // --- Per-frame statistics -----------------------------------
            const auto& frame = *result.frame;
            const int n_feats = static_cast<int>(frame.num_features());

            std::vector<float> depths;
            depths.reserve(n_feats);
            for (int k = 0; k < n_feats; ++k)
                if (frame.depth[k] > 0.0f)
                    depths.push_back(frame.depth[k]);

            const float ratio = n_feats > 0
                ? static_cast<float>(result.n_stereo) / n_feats : 0.0f;
            total_s_ratio += ratio;

            float median_depth = 0.0f;
            if (!depths.empty()) {
                std::nth_element(depths.begin(),
                                 depths.begin() + depths.size() / 2,
                                 depths.end());
                median_depth = depths[depths.size() / 2];
            }

            const bool lost = result.state == sslam::TrackingState::LOST;
            if (lost) ++n_lost;

            std::cout << "frame " << i
                      << (lost ? "  [LOST]" : "  [OK]  ")
                      << "  stereo=" << result.n_stereo
                      << " (" << static_cast<int>(ratio * 100.f) << "%)"
                      << "  fm=" << result.n_matches
                      << "  inliers=" << result.n_inliers
                      << "  med_d=" << median_depth << " m"
                      << "  " << static_cast<int>(ms) << " ms\n";

            // --- Visualisation ------------------------------------------
            if (display) {
                cv::Mat vis;
                cv::hconcat(frame.left, frame.right, vis);
                cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
                const int offset = frame.left.cols;

                for (int k = 0; k < n_feats; ++k) {
                    const cv::Point2f pt_l = frame.keypoints_left[k].pt;
                    if (frame.right_u[k] >= 0.0f) {
                        cv::circle(vis, pt_l, 3, {0, 200, 0}, 1);
                        cv::line(vis, pt_l,
                                 cv::Point2f(offset + frame.right_u[k], pt_l.y),
                                 {0, 200, 200}, 1);
                    } else {
                        cv::circle(vis, pt_l, 3, {0, 0, 200}, 1);
                    }
                }

                const cv::Scalar text_col = lost ? cv::Scalar{0, 0, 220}
                                                 : cv::Scalar{200, 200, 200};
                cv::putText(vis,
                    "frame " + std::to_string(i) +
                    (lost ? "  LOST" : "  OK") +
                    "  stereo " + std::to_string(result.n_stereo) + "/" +
                    std::to_string(n_feats) +
                    "  inliers " + std::to_string(result.n_inliers),
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.65, text_col, 2);

                cv::imshow(win, vis);
                if (const int k = cv::waitKey(1); k == 27 || k == 'q') break;
            }
        }

        std::cout << "\nSummary:"
                  << "\n  avg latency  : " << (total_ms / loader.size()) << " ms/frame"
                  << "\n  avg stereo % : "
                  << static_cast<int>(total_s_ratio / loader.size() * 100.0) << "%"
                  << "\n  lost frames  : " << n_lost << " / " << loader.size() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

