#include "sslam/frontend/stereo_matcher.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace sslam {
namespace {

cv::Mat descriptor_row(uint8_t value) {
    cv::Mat desc(1, 32, CV_8U);
    desc.setTo(value);
    return desc;
}

}  // namespace

TEST(StereoMatcher, IgnoresInvalidDisparityBeforeChoosingBestDescriptor) {
    auto cam = std::make_shared<StereoCamera>();
    cam->fx = 100.0;
    cam->fy = 100.0;
    cam->cx = 80.0;
    cam->cy = 40.0;
    cam->baseline = 0.5;
    cam->width = 160;
    cam->height = 80;

    StereoMatcher::Params params;
    params.min_disparity = 1.0f;
    params.max_disparity = 30.0f;
    StereoMatcher matcher(cam, params);

    Frame frame(0, 0.0,
                cv::Mat::zeros(cam->height, cam->width, CV_8U),
                cv::Mat::zeros(cam->height, cam->width, CV_8U),
                cam);
    frame.keypoints_left.emplace_back(cv::Point2f(100.0f, 40.0f), 1.0f);
    frame.keypoints_left.back().octave = 0;
    frame.descriptors_left = descriptor_row(0);

    std::vector<cv::KeyPoint> right_kps;
    right_kps.emplace_back(cv::Point2f(10.0f, 40.0f), 1.0f);   // invalid: disparity 90 > max
    right_kps.back().octave = 0;
    right_kps.emplace_back(cv::Point2f(80.0f, 40.0f), 1.0f);   // valid: disparity 20
    right_kps.back().octave = 0;

    cv::Mat right_descs(2, 32, CV_8U);
    right_descs.row(0).setTo(0);  // would win without disparity pre-filtering
    right_descs.row(1).setTo(1);

    matcher.match(frame, right_kps, right_descs);

    ASSERT_EQ(frame.depth.size(), 1u);
    ASSERT_EQ(frame.right_u.size(), 1u);
    EXPECT_GT(frame.depth[0], 0.0f);
    EXPECT_NEAR(frame.right_u[0], 80.0f, 1.0f);
    EXPECT_NEAR(frame.depth[0], 2.5f, 0.2f);
}

TEST(StereoMatcher, IgnoresRightKeypointsAtIncompatibleOctaves) {
    auto cam = std::make_shared<StereoCamera>();
    cam->fx = 100.0;
    cam->fy = 100.0;
    cam->cx = 80.0;
    cam->cy = 40.0;
    cam->baseline = 0.5;
    cam->width = 160;
    cam->height = 80;

    StereoMatcher matcher(cam);

    Frame frame(0, 0.0,
                cv::Mat::zeros(cam->height, cam->width, CV_8U),
                cv::Mat::zeros(cam->height, cam->width, CV_8U),
                cam);
    frame.keypoints_left.emplace_back(cv::Point2f(100.0f, 40.0f), 1.0f);
    frame.keypoints_left.back().octave = 0;
    frame.descriptors_left = descriptor_row(0);

    std::vector<cv::KeyPoint> right_kps;
    right_kps.emplace_back(cv::Point2f(80.0f, 40.0f), 1.0f);
    right_kps.back().octave = 4;
    cv::Mat right_descs = descriptor_row(0);

    matcher.match(frame, right_kps, right_descs);

    ASSERT_EQ(frame.depth.size(), 1u);
    EXPECT_LT(frame.depth[0], 0.0f);
}

}  // namespace sslam
