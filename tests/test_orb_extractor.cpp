#include "sslam/frontend/orb_extractor.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <random>

namespace {

cv::Mat make_test_image(int w = 640, int h = 480) {
    // Random Gaussian noise — gives FAST plenty of corners everywhere, which
    // is exactly the case where the quadtree distribution matters.
    cv::Mat img(h, w, CV_8UC1);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> u(0, 255);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img.at<uchar>(y, x) = static_cast<uchar>(u(rng));
    return img;
}

}  // namespace

TEST(ORBExtractor, ProducesCloseToRequestedFeatureCount) {
    sslam::ORBExtractor::Params p;
    p.num_features = 1000;
    sslam::ORBExtractor ext(p);

    auto img = make_test_image();
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    ext.detect(img, kps, desc);

    EXPECT_GT(kps.size(), 600u);    // we tolerate some shortfall
    EXPECT_LT(kps.size(), 1500u);   // and some overshoot from quadtree expansion
    EXPECT_EQ(static_cast<int>(kps.size()), desc.rows);
    EXPECT_EQ(desc.cols, 32);
    EXPECT_EQ(desc.type(), CV_8U);
}

TEST(ORBExtractor, KeypointsAreSpatiallyDistributed) {
    // The quadtree is supposed to spread keypoints over the image. We check
    // that *every quadrant* of the image gets a meaningful share — naive
    // FAST on noise tends to clump heavily.
    sslam::ORBExtractor::Params p;
    p.num_features = 1000;
    sslam::ORBExtractor ext(p);

    auto img = make_test_image(640, 480);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    ext.detect(img, kps, desc);
    ASSERT_FALSE(kps.empty());

    int q[4] = {0, 0, 0, 0};
    for (const auto& kp : kps) {
        const int qi = (kp.pt.x >= 320 ? 1 : 0) + (kp.pt.y >= 240 ? 2 : 0);
        ++q[qi];
    }
    const auto frac = [&](int idx) { return static_cast<double>(q[idx]) / kps.size(); };
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(frac(i), 0.10) << "quadrant " << i << " under-represented";
    }
}

TEST(ORBExtractor, AssignsOctaveAcrossPyramidLevels) {
    sslam::ORBExtractor::Params p;
    p.num_features = 1000;
    p.num_levels = 8;
    sslam::ORBExtractor ext(p);

    auto img = make_test_image();
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    ext.detect(img, kps, desc);
    ASSERT_FALSE(kps.empty());

    int max_octave = 0;
    for (const auto& kp : kps) max_octave = std::max(max_octave, kp.octave);
    EXPECT_GE(max_octave, 2) << "all features came from base level";
}
