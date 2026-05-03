#include "sslam/frontend/feature_matcher.hpp"
#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <random>

namespace {

// KITTI seq-00 approximate intrinsics.
std::shared_ptr<sslam::StereoCamera> make_cam() {
    auto cam = std::make_shared<sslam::StereoCamera>();
    cam->fx = 718.856; cam->fy = 718.856;
    cam->cx = 607.193; cam->cy = 185.216;
    cam->baseline = 0.5372;
    cam->width = 1241; cam->height = 376;
    return cam;
}

// Build a Frame with N synthetic keypoints all at octave 0.
// Each keypoint gets a unique 32-byte descriptor (seeded deterministically).
// depth[i] is set to the supplied depths vector; right_u is computed from it.
// T_cw is Identity.
sslam::Frame make_frame(
    std::shared_ptr<const sslam::StereoCamera> cam,
    const std::vector<cv::Point2f>& pts,
    const std::vector<float>& depths,
    const std::vector<cv::Mat>& descs,
    const Eigen::Matrix4d& T_cw = Eigen::Matrix4d::Identity()) {

    sslam::Frame f;
    f.left   = cv::Mat::zeros(cam->height, cam->width, CV_8UC1);
    f.right  = cv::Mat::zeros(cam->height, cam->width, CV_8UC1);
    f.camera = cam;
    f.T_cw   = T_cw;

    for (std::size_t i = 0; i < pts.size(); ++i) {
        cv::KeyPoint kp;
        kp.pt     = pts[i];
        kp.octave = 0;
        f.keypoints_left.push_back(kp);
        f.descriptors_left.push_back(descs[i]);
        f.depth.push_back(depths[i]);
        // right_u is not used by FeatureMatcher, but fill it for completeness.
        const float disp = depths[i] > 0.0f
            ? static_cast<float>(cam->fx * cam->baseline / depths[i]) : -1.0f;
        f.right_u.push_back(depths[i] > 0.0f ? pts[i].x - disp : -1.0f);
    }
    return f;
}

// Generate N distinct 32-byte ORB-like descriptors (seeded).
std::vector<cv::Mat> make_descriptors(int n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<cv::Mat> descs;
    descs.reserve(n);
    for (int i = 0; i < n; ++i) {
        cv::Mat d(1, 32, CV_8U);
        for (int b = 0; b < 32; ++b)
            d.at<uint8_t>(0, b) = static_cast<uint8_t>(byte_dist(rng));
        descs.push_back(d);
    }
    return descs;
}

}  // namespace

// --------------------------------------------------------------------------
// Test 1: identity transform → every depth-mapped keypoint in prev should
// project to (nearly) the same pixel in curr and be uniquely matched.
// --------------------------------------------------------------------------
TEST(FeatureMatcher, IdentityTransformRecoversAllMatches) {
    auto cam = make_cam();
    sslam::FeatureMatcher matcher(cam);

    constexpr int N = 50;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> du(100.0f, 1100.0f);
    std::uniform_real_distribution<float> dv(50.0f, 320.0f);
    std::uniform_real_distribution<float> dd(5.0f, 50.0f);

    std::vector<cv::Point2f> pts(N);
    std::vector<float>       depths(N);
    for (int i = 0; i < N; ++i) {
        pts[i]    = {du(rng), dv(rng)};
        depths[i] = dd(rng);
    }
    auto descs = make_descriptors(N);

    // prev and curr are identical frames at identity pose.
    auto prev = make_frame(cam, pts, depths, descs);
    auto curr = make_frame(cam, pts, depths, descs);

    const auto matches = matcher.match_by_projection(
        prev, curr, Eigen::Matrix4d::Identity());

    // With unique descriptors and zero displacement every point must match.
    EXPECT_EQ(static_cast<int>(matches.size()), N);

    // Verify each match is the identity correspondence (prev_i ↔ curr_i).
    for (const auto& [pi, ci] : matches) {
        EXPECT_EQ(pi, ci) << "prev[" << pi << "] matched to curr[" << ci << "]";
    }
}

// --------------------------------------------------------------------------
// Test 2: small forward translation → projected positions shift by at most
// a few pixels; all N features should still be found within the default
// search radius.
// --------------------------------------------------------------------------
TEST(FeatureMatcher, SmallTranslationRecoversMatches) {
    auto cam = make_cam();
    sslam::FeatureMatcher matcher(cam);

    constexpr int N = 50;
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> du(150.0f, 1050.0f);
    std::uniform_real_distribution<float> dv(60.0f, 300.0f);
    std::uniform_real_distribution<float> dd(10.0f, 50.0f);

    std::vector<cv::Point2f> prev_pts(N);
    std::vector<float>       depths(N);
    for (int i = 0; i < N; ++i) {
        prev_pts[i] = {du(rng), dv(rng)};
        depths[i]   = dd(rng);
    }
    auto descs = make_descriptors(N, 99);

    // Camera moves 0.05 m in X: T_curr_cw has t.x = +0.05
    // (world origin at prev; after moving 0.05m rightward, points shift left).
    Eigen::Matrix4d T_curr_cw = Eigen::Matrix4d::Identity();
    constexpr double tx = 0.05;
    T_curr_cw(0, 3) = tx;

    // Compute ground-truth projected positions in curr for each 3-D point.
    std::vector<cv::Point2f> curr_pts(N);
    for (int i = 0; i < N; ++i) {
        const double xc = (prev_pts[i].x - cam->cx) * depths[i] / cam->fx + tx;
        const double yc = (prev_pts[i].y - cam->cy) * depths[i] / cam->fy;
        const double zc = static_cast<double>(depths[i]);
        curr_pts[i] = {
            static_cast<float>(cam->fx * xc / zc + cam->cx),
            static_cast<float>(cam->fy * yc / zc + cam->cy)};
    }

    auto prev = make_frame(cam, prev_pts, depths, descs);
    auto curr = make_frame(cam, curr_pts, depths, descs, T_curr_cw);

    const auto matches = matcher.match_by_projection(prev, curr, T_curr_cw);

    // All 50 depth-mapped points should match (shift ≪ 10 px default radius).
    EXPECT_GE(static_cast<int>(matches.size()), N * 9 / 10)
        << "got only " << matches.size() << "/" << N << " matches";
}

// --------------------------------------------------------------------------
// Test 3: keypoints without depth must be skipped.
// --------------------------------------------------------------------------
TEST(FeatureMatcher, SkipsKeypointsWithoutDepth) {
    auto cam = make_cam();
    sslam::FeatureMatcher matcher(cam);

    std::vector<cv::Point2f> pts  = {{300.0f, 150.0f}, {600.0f, 200.0f}};
    std::vector<float>       deps = {-1.0f, 15.0f};  // first has no depth
    auto descs = make_descriptors(2);

    auto prev = make_frame(cam, pts, deps, descs);
    auto curr = make_frame(cam, pts, deps, descs);

    const auto matches = matcher.match_by_projection(
        prev, curr, Eigen::Matrix4d::Identity());

    // Only the second keypoint (with depth=15) should produce a match.
    EXPECT_EQ(static_cast<int>(matches.size()), 1);
    EXPECT_EQ(matches[0].first,  1);
    EXPECT_EQ(matches[0].second, 1);
}
