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
// Usage:  ./kitti_stereo <kitti-sequence-dir> [--no-display] [--output <traj.txt>]

#include "sslam/dataset/kitti_loader.hpp"
#include "sslam/frontend/orb_vocabulary.hpp"
#include "sslam/loop/keyframe_database.hpp"
#include "sslam/loop/loop_closing.hpp"
#include "sslam/loop/loop_diagnostics.hpp"
#include "sslam/optim/full_ba.hpp"
#include "sslam/system.hpp"
#include "sslam/tracking/tracking.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/viewer/map_viewer.hpp"

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <kitti-sequence-dir> [--no-display] [--output <traj.txt>]"
                     " [--max-frames N] [--gt <poses.txt>] [--loop-log <log.jsonl>]"
                     " [--config <config.yaml>]\n";
        return 1;
    }

    bool        display     = false;
    bool        map_display = true;
    std::string output_path;
    std::string gt_path;
    std::string loop_log_path;
    std::string config_path;
    std::size_t max_frames  = std::numeric_limits<std::size_t>::max();
    for (int a = 2; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--no-display") {
            display     = false;
            map_display = false;
        } else if (arg == "--output" && a + 1 < argc) {
            output_path = argv[++a];
        } else if (arg == "--gt" && a + 1 < argc) {
            gt_path = argv[++a];
        } else if (arg == "--loop-log" && a + 1 < argc) {
            loop_log_path = argv[++a];
        } else if (arg == "--config" && a + 1 < argc) {
            config_path = argv[++a];
        } else if (arg == "--max-frames" && a + 1 < argc) {
            max_frames = static_cast<std::size_t>(std::stoul(argv[++a]));
        }
    }

    // Load ground-truth poses if provided (KITTI format: 12 values per line).
    std::vector<Eigen::Matrix4d> gt_poses;
    if (!gt_path.empty()) {
        std::ifstream f(gt_path);
        if (!f) throw std::runtime_error("Cannot open GT file: " + gt_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    ss >> T(r, c);
            gt_poses.push_back(T);
        }
        std::cout << "  GT poses     : " << gt_poses.size() << " frames\n";
    }

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

        // --- Load optional config (YAML) -------------------------------------
        sslam::LoopClosing::Params lc_params;
        sslam::LocalMapping::Params lm_params;
        std::string vocab_path = "thirdparty/vocab/ORBvoc.txt";
        if (!config_path.empty()) {
            cv::FileStorage fs(config_path, cv::FileStorage::READ);
            if (!fs.isOpened())
                throw std::runtime_error("Cannot open config: " + config_path);
            // Loop closing params
            if (!fs["loop"]["min_bow_score"].empty())
                lc_params.min_bow_score = static_cast<double>(fs["loop"]["min_bow_score"]);
            if (!fs["loop"]["cooldown_kfs"].empty())
                lc_params.cooldown_kfs = static_cast<int>(fs["loop"]["cooldown_kfs"]);
            if (!fs["loop"]["min_bow_matches"].empty())
                lc_params.min_bow_matches = static_cast<int>(fs["loop"]["min_bow_matches"]);
            if (!fs["loop"]["min_correspondences"].empty())
                lc_params.min_correspondences = static_cast<int>(fs["loop"]["min_correspondences"]);
            if (!fs["loop"]["min_ransac_inliers"].empty())
                lc_params.min_ransac_inliers = static_cast<int>(fs["loop"]["min_ransac_inliers"]);
            if (!fs["loop"]["min_fused_matches"].empty())
                lc_params.min_fused_matches = static_cast<int>(fs["loop"]["min_fused_matches"]);
            if (!fs["loop"]["max_candidates_per_kf"].empty())
                lc_params.max_candidates_per_kf = static_cast<int>(fs["loop"]["max_candidates_per_kf"]);
            if (!fs["loop"]["min_sim3_inlier_ratio"].empty())
                lc_params.min_sim3_inlier_ratio = static_cast<double>(fs["loop"]["min_sim3_inlier_ratio"]);
            if (!fs["loop"]["max_sim3_rmse_m"].empty())
                lc_params.max_sim3_rmse_m = static_cast<double>(fs["loop"]["max_sim3_rmse_m"]);
            if (!fs["loop"]["max_pgo_adjacent_step_m"].empty())
                lc_params.max_pgo_adjacent_step_m = static_cast<double>(fs["loop"]["max_pgo_adjacent_step_m"]);
            if (!fs["loop"]["vocab_path"].empty())
                vocab_path = static_cast<std::string>(fs["loop"]["vocab_path"]);
            // LocalMapping / BA params
            if (!fs["local_mapping"]["local_ba_keyframe_window"].empty())
                lm_params.ba.max_local_kfs = static_cast<int>(fs["local_mapping"]["local_ba_keyframe_window"]);
            std::cout << "Config         : " << config_path << "\n";
        }

        sslam::Tracking tracker(cam_ptr);
        tracker.local_mapping()->set_params(lm_params);

        // --- Loop closing (optional — skipped if vocabulary is absent) -----
        std::unique_ptr<sslam::ORBVocabulary>     vocab;
        std::unique_ptr<sslam::KeyFrameDatabase>  kf_db;
        std::unique_ptr<sslam::LoopLogger>        loop_logger;
        sslam::LoopClosing::Ptr                   loop_closer;
        {
            std::ifstream probe(vocab_path);
            if (probe.good()) {
                vocab = std::make_unique<sslam::ORBVocabulary>();
                std::cout << "Loading ORB vocabulary ... " << std::flush;
                vocab->load(vocab_path);
                std::cout << "done\n";
                kf_db = std::make_unique<sslam::KeyFrameDatabase>(*vocab);
                tracker.local_mapping()->set_vocabulary(vocab.get());
                tracker.local_mapping()->set_keyframe_database(kf_db.get());
                // Also wire vocab+db to Tracking for relocalization.
                tracker.set_vocabulary(vocab.get());
                tracker.set_keyframe_database(kf_db.get());
                loop_closer = std::make_shared<sslam::LoopClosing>(
                    tracker.map(),
                    tracker.local_mapping(),
                    vocab.get(),
                    kf_db.get(),
                    lc_params);
                tracker.local_mapping()->set_loop_closing(loop_closer.get());
                if (!loop_log_path.empty()) {
                    loop_logger = std::make_unique<sslam::LoopLogger>(loop_log_path);
                    loop_closer->set_loop_logger(loop_logger.get());
                    std::cout << "Loop log       : " << loop_log_path << "\n";
                }
                loop_closer->start();
                std::cout << "Loop closing   : enabled\n";
            } else {
                std::cout << "Loop closing   : disabled (vocabulary not found at "
                          << vocab_path << ")\n";
            }
        }

        sslam::MapViewer::Ptr viewer;
        if (map_display) {
            viewer = std::make_shared<sslam::MapViewer>(tracker.map());
            viewer->start();
        }

        const std::string win = "sslam :: stereo VO";
        if (display) cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

        double      total_ms      = 0.0;
        double      total_s_ratio = 0.0;
        std::size_t n_lost        = 0;
        double      sum_sq_err    = 0.0;  // for ATE RMSE
        std::size_t n_gt_frames   = 0;

        std::vector<Eigen::Matrix4d> trajectory;
        trajectory.reserve(loader.size());
        // We no longer pre-populate `trajectory` per frame; it is rebuilt
        // from `tracker.resolved_trajectory()` after the run so that any
        // BA correction landed during processing is reflected.

        // Deterministic mode (opt-in via SSLAM_DETERMINISTIC=1): after every
        // frame, drain LocalMapping and LoopClosing so the next frame always
        // sees a fully-settled map.  In the default concurrent mode, Tracking
        // reads keyframe/map-point poses while LocalMapping is still refining
        // them on another thread, so how much optimisation has "landed" by the
        // time frame i is processed depends on wall-clock timing — which makes
        // the trajectory differ from run to run.  Serialising the pipeline
        // removes that timing dependence and yields bit-reproducible results
        // (at the cost of real-time throughput), which is what we want for
        // benchmarking and regression testing.
        const char* det_env = std::getenv("SSLAM_DETERMINISTIC");
        const bool deterministic = det_env && std::string(det_env) == "1";
        if (deterministic) {
            // Force OpenCV to run single-threaded.  cv::solvePnPRansac (used in
            // tracking) distributes its RANSAC iterations across OpenCV's
            // internal worker threads; that work split is non-deterministic, so
            // near an inlier threshold two runs can pick different minimal
            // samples and disagree on whether a frame is tracked or lost — after
            // which the whole trajectory diverges.  Serialising OpenCV removes
            // that source of run-to-run variation.
            cv::setNumThreads(1);
            std::cout << "Deterministic  : enabled (pipeline + OpenCV serialised)\n";
        }

        for (std::size_t i = 0; i < std::min(loader.size(), max_frames); ++i) {
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

            trajectory.push_back(result.frame->T_cw);
            // Note: this entry is the *raw* per-frame pose at the moment
            // of capture; the saved trajectory below uses the BA-corrected
            // resolved poses.

            if (viewer) {
                Eigen::Matrix4d T_view_cw = result.frame->T_cw;
                if (frame.ref_kf) {
                    T_view_cw = frame.T_ref *
                                frame.ref_kf->get_pose_through_spanning_tree();
                }
                viewer->set_current_pose(T_view_cw);
            }

            // Per-frame translational error against GT (if provided).
            double t_err = -1.0;
            if (i < gt_poses.size()) {
                // Estimate is T_cw; KITTI ground truth is T_wc.
                const Eigen::Matrix4d& T_est = result.frame->T_cw;
                const Eigen::Matrix4d& T_gt  = gt_poses[i];
                const Eigen::Vector3d t_est =
                    -T_est.block<3,3>(0,0).transpose() * T_est.block<3,1>(0,3);
                const Eigen::Vector3d t_gt = T_gt.block<3,1>(0,3);
                t_err = (t_est - t_gt).norm();
                sum_sq_err += t_err * t_err;
                ++n_gt_frames;
            }

            std::cout << "frame " << i
                      << (lost ? "  [LOST]" : "  [OK]  ")
                      << "  stereo=" << result.n_stereo
                      << " (" << static_cast<int>(ratio * 100.f) << "%)"
                      << "  fm=" << result.n_matches
                      << "  inliers=" << result.n_inliers
                      << "  med_d=" << median_depth << " m"
                      << "  " << static_cast<int>(ms) << " ms"
                      << (t_err >= 0.0
                          ? ("  err=" + std::to_string(t_err).substr(0, 5) + "m")
                          : "")
                      << "\n";

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

            if (deterministic) {
                tracker.local_mapping()->wait_until_idle();
                if (loop_closer) loop_closer->wait_until_idle();
            }
        }

        // End-of-sequence export should include tail keyframes.  Drain mapping
        // first so every processed KF is forwarded to LoopClosing, then drain
        // LoopClosing so late sequence loops can still be corrected before the
        // trajectory is written.  FullBA is not launched from LoopClosing, so
        // this no longer blocks on a large background global solve.
        tracker.local_mapping()->wait_until_idle();
        if (loop_closer) loop_closer->wait_until_idle();
        tracker.local_mapping()->shutdown();
        if (loop_closer) loop_closer->shutdown();
        const auto ba_stats = tracker.local_mapping()->ba_stats();

        // --- Final global bundle adjustment (opt-in) -------------------------
        // A final batch BA over the whole map is the classic "Full BA" stage of
        // ORB-SLAM2.  In this pipeline, however, benchmarking showed it
        // *degrades* the trajectory on KITTI: on a loop-free sequence it only
        // re-converges to the drifted estimate and adds noise (seq 07
        // 1.65m -> 3.82m aligned), and even on a loop sequence it undoes part
        // of the pose-graph correction (seq 00 3.35m -> 5.44m aligned).  The
        // dominant accuracy lever here is the sparse essential-graph PGO run at
        // loop-closure time, not a final global BA.  It is therefore disabled
        // by default and only run when SSLAM_FINAL_BA=1 is set, for experiments.
        const int n_loops_closed = loop_closer ? loop_closer->loop_count() : 0;
        const char* ba_env = std::getenv("SSLAM_FINAL_BA");
        const bool final_ba_enabled = ba_env && std::string(ba_env) == "1";
        if (n_loops_closed > 0 && final_ba_enabled) {
            std::cout << "Final global BA ... " << std::flush;
            const auto t_ba0 = std::chrono::steady_clock::now();
            sslam::FullBA full_ba(tracker.map());
            full_ba.trigger();
            full_ba.wait();
            const auto t_ba1 = std::chrono::steady_clock::now();
            std::cout << "done ("
                      << static_cast<int>(
                             std::chrono::duration<double, std::milli>(
                                 t_ba1 - t_ba0).count())
                      << " ms)\n";
        }

        const std::size_t n_frames_run = std::min(loader.size(), max_frames);
        std::size_t active_kfs = 0;
        for (const auto& kf : tracker.map()->get_all_keyframes())
            if (kf && !kf->is_bad()) ++active_kfs;

        std::size_t active_mps = 0;
        for (const auto& mp : tracker.map()->get_all_mappoints())
            if (mp && !mp->is_bad()) ++active_mps;

        std::cout << "\nSummary:"
                  << "\n  avg latency  : " << (total_ms / n_frames_run) << " ms/frame"
                  << "\n  avg stereo % : "
                  << static_cast<int>(total_s_ratio / n_frames_run * 100.0) << "%"
                  << "\n  lost frames  : " << n_lost << " / " << n_frames_run
                  << "\n  keyframes    : " << active_kfs << " active / "
                  << tracker.map()->keyframe_count() << " total"
                  << "\n  map points   : " << active_mps << " active / "
                  << tracker.map()->mappoint_count() << " total"
                  << "\n  local BA     : " << ba_stats.runs << " runs, avg "
                  << ba_stats.avg_ms() << " ms, max " << ba_stats.max_ms << " ms"
                  << "\n  loop closures: "
                  << (loop_closer ? std::to_string(loop_closer->loop_count()) : "disabled");
        if (n_gt_frames > 0)
            std::cout << "\n  ATE RMSE     : "
                      << std::sqrt(sum_sq_err / n_gt_frames) << " m  (raw, no alignment)";

        // Recompute ATE on the BA-anchored resolved trajectory.
        const auto resolved = tracker.resolved_trajectory();
        if (!gt_poses.empty() && !resolved.empty()) {
            const std::size_t n =
                std::min(resolved.size(), gt_poses.size());
            double sq = 0.0;
            std::size_t k = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const Eigen::Matrix4d& T_est = resolved[i];
                const Eigen::Matrix4d& T_gt  = gt_poses[i];
                const Eigen::Vector3d t_est =
                    -T_est.block<3,3>(0,0).transpose() * T_est.block<3,1>(0,3);
                const Eigen::Vector3d t_gt = T_gt.block<3,1>(0,3);
                sq += (t_est - t_gt).squaredNorm();
                ++k;
            }
            if (k > 0)
                std::cout << "\n  ATE (resolved): "
                          << std::sqrt(sq / k) << " m  (BA-anchored)";
        }
        std::cout << "\n";

        if (!output_path.empty()) {
            // Use the BA-anchored trajectory: each frame's pose is
            // reconstructed against its reference KeyFrame's current pose,
            // so Local BA corrections propagate into the saved file.
            const auto resolved = tracker.resolved_trajectory();
            sslam::save_trajectory_kitti(output_path, resolved);
            std::cout << "  trajectory   : " << output_path << "\n";
        }

        if (viewer) {
            const auto resolved = tracker.resolved_trajectory();
            if (!resolved.empty()) viewer->set_current_pose(resolved.back());
            std::cout << "  [viewer] Examine the map. Close the Pangolin window to exit.\n";
            viewer->wait_until_closed();
        }

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

