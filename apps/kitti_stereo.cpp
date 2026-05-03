// Phase 1 demo: load KITTI, run ORB on left + right, perform stereo matching,
// and visualise results.
//
// Visualisation:
//   Green  circle  — left keypoint WITH a stereo match (has depth)
//   Red    circle  — left keypoint WITHOUT a stereo match (mono only)
//   Yellow line    — connects left match to its right-image x on the same row
//
// Per-frame stats printed to stdout: matched %, mean/median depth (m).
//
// Usage:  ./kitti_stereo <kitti-sequence-dir> [--no-display]

#include "sslam/dataset/kitti_loader.hpp"
#include "sslam/frontend/feature_matcher.hpp"
#include "sslam/frontend/orb_extractor.hpp"
#include "sslam/frontend/stereo_matcher.hpp"
#include "sslam/types/frame.hpp"

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

        sslam::ORBExtractor extractor;
        sslam::StereoMatcher  stereo_matcher(cam_ptr);
        sslam::FeatureMatcher feature_matcher(cam_ptr);

        const std::string win = "sslam :: stereo matches";
        if (display) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

        double total_ms       = 0.0;
        double total_s_ratio  = 0.0;  // stereo match ratio
        double total_f_matches = 0.0; // frame-to-frame match count
        std::size_t n_frame_pairs = 0;

        sslam::Frame prev_frame;  // carry across iterations
        bool has_prev = false;

        for (std::size_t i = 0; i < loader.size(); ++i) {
            auto raw = loader.load(i);

            sslam::Frame frame(i, raw.timestamp, raw.left, raw.right, cam_ptr);

            // Extract ORB on both images
            std::vector<cv::KeyPoint> right_kps;
            cv::Mat right_descs;

            const auto t0 = std::chrono::steady_clock::now();
            extractor.detect(frame.left,  frame.keypoints_left,  frame.descriptors_left);
            extractor.detect(frame.right, right_kps,             right_descs);
            stereo_matcher.match(frame, right_kps, right_descs);
            const auto t1 = std::chrono::steady_clock::now();

            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_ms += ms;

            // --- Frame-to-frame matching --------------------------------
            int n_fm = 0;
            if (has_prev) {
                // Use prev pose as T_pred (zero-velocity assumption until
                // the motion model is wired in).
                const auto fm = feature_matcher.match_by_projection(
                    prev_frame, frame, prev_frame.T_cw);
                n_fm = static_cast<int>(fm.size());
                total_f_matches += n_fm;
                ++n_frame_pairs;
            }

            // --- Per-frame statistics -----------------------------------
            int n_matched = 0;
            std::vector<float> depths;
            depths.reserve(frame.num_features());

            for (std::size_t k = 0; k < frame.num_features(); ++k) {
                if (frame.depth[k] > 0.0f) {
                    ++n_matched;
                    depths.push_back(frame.depth[k]);
                }
            }

            const float ratio =
                frame.num_features() > 0
                    ? static_cast<float>(n_matched) / frame.num_features()
                    : 0.0f;
            total_s_ratio += ratio;

            float mean_depth  = 0.0f;
            float median_depth = 0.0f;
            if (!depths.empty()) {
                mean_depth = std::accumulate(depths.begin(), depths.end(), 0.0f) /
                             depths.size();
                std::nth_element(depths.begin(),
                                 depths.begin() + depths.size() / 2,
                                 depths.end());
                median_depth = depths[depths.size() / 2];
            }

            std::cout << "frame " << i
                      << "  kps_L=" << frame.num_features()
                      << "  stereo=" << n_matched
                      << "  (" << static_cast<int>(ratio * 100.0f) << "%)"
                      << "  fm=" << n_fm
                      << "  mean_d=" << mean_depth << " m"
                      << "  med_d="  << median_depth << " m"
                      << "  " << static_cast<int>(ms) << " ms\n";

            prev_frame = frame;
            has_prev   = true;

            // --- Visualisation ------------------------------------------
            if (display) {
                // Side-by-side canvas (left | right)
                cv::Mat vis;
                cv::hconcat(frame.left, frame.right, vis);
                cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);

                const int offset = frame.left.cols;  // right panel x-shift

                for (std::size_t k = 0; k < frame.num_features(); ++k) {
                    const cv::KeyPoint& kp = frame.keypoints_left[k];
                    const cv::Point2f   pt_l(kp.pt.x, kp.pt.y);

                    if (frame.right_u[k] >= 0.0f) {
                        // Matched — green on left, yellow line to right
                        cv::circle(vis, pt_l, 3, {0, 200, 0}, 1);
                        const cv::Point2f pt_r(offset + frame.right_u[k], kp.pt.y);
                        cv::line(vis, pt_l, pt_r, {0, 200, 200}, 1);
                    } else {
                        // Unmatched — red on left only
                        cv::circle(vis, pt_l, 3, {0, 0, 200}, 1);
                    }
                }

                cv::putText(vis,
                            "frame " + std::to_string(i) +
                                "  matched " + std::to_string(n_matched) + "/" +
                                std::to_string(frame.num_features()) +
                                "  med_d " + std::to_string(static_cast<int>(median_depth)) + " m",
                            {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.65,
                            {200, 200, 200}, 2);

                cv::imshow(win, vis);
                if (const int k = cv::waitKey(1); k == 27 || k == 'q') break;
            }
        }

        std::cout << "\nSummary:"
                  << "\n  avg latency   : " << (total_ms / loader.size()) << " ms/frame"
                  << "\n  avg stereo %  : "
                  << static_cast<int>(total_s_ratio / loader.size() * 100.0) << "%"
                  << "\n  avg fm matches: "
                  << (n_frame_pairs > 0 ? total_f_matches / n_frame_pairs : 0.0)
                  << "\n";

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

