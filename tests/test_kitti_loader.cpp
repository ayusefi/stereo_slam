// Smoke tests for the KITTI loader. The real loader test runs only when
// SSLAM_KITTI_SEQ env var points to a real sequence directory.

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/dataset/kitti_loader.hpp"

#include <gtest/gtest.h>
#include <cstdlib>

TEST(StereoCamera, BackprojectRoundTrip) {
    sslam::StereoCamera cam;
    cam.fx = 700; cam.fy = 700; cam.cx = 600; cam.cy = 180;
    cam.baseline = 0.54;

    // Pixel (700, 280) with disparity 14 => Z = 700 * 0.54 / 14 = 27 m
    auto X = cam.backproject(700, 280, 14);
    EXPECT_NEAR(X.z(), 27.0, 1e-9);
    // u - cx = 100 => X = 100 * 27 / 700
    EXPECT_NEAR(X.x(), 100.0 * 27.0 / 700.0, 1e-9);
    // v - cy = 100 => Y = 100 * 27 / 700
    EXPECT_NEAR(X.y(), 100.0 * 27.0 / 700.0, 1e-9);
}

TEST(StereoCamera, BackprojectRejectsBadDisparity) {
    sslam::StereoCamera cam;
    cam.fx = cam.fy = 500; cam.cx = 320; cam.cy = 240; cam.baseline = 0.1;
    EXPECT_THROW(cam.backproject(0, 0, 0.0),  std::invalid_argument);
    EXPECT_THROW(cam.backproject(0, 0, -1.0), std::invalid_argument);
}

TEST(KittiLoader, LoadsRealSequenceIfAvailable) {
    const char* env = std::getenv("SSLAM_KITTI_SEQ");
    if (!env) GTEST_SKIP() << "set SSLAM_KITTI_SEQ to a sequence dir to enable";

    sslam::KittiLoader loader(env);
    ASSERT_GT(loader.size(), 0u);
    EXPECT_GT(loader.camera().fx, 0.0);
    EXPECT_GT(loader.camera().baseline, 0.0);

    auto f0 = loader.load(0);
    EXPECT_FALSE(f0.left.empty());
    EXPECT_FALSE(f0.right.empty());
    EXPECT_EQ(f0.left.size(), f0.right.size());
}
