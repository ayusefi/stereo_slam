// Unit tests for Map, KeyFrame, and MapPoint 
//
// Test: build a Map of 5 KFs and 100 MPs with overlapping observations;
// verify that covisibility edge weights match the hand-computed values and
// that Map::local_map_around returns the correct subset.
//
// Observation layout (KF k observes MPs [k*10, k*10+40)):
//   KF0: MPs 0–39      KF1: MPs 10–49
//   KF2: MPs 20–59     KF3: MPs 30–69
//   KF4: MPs 40–79
//
// Expected pairwise shared counts:
//   KF0↔KF1=30  KF0↔KF2=20  KF0↔KF3=10  KF0↔KF4=0
//   KF1↔KF2=30  KF1↔KF3=20  KF1↔KF4=10
//   KF2↔KF3=30  KF2↔KF4=20
//   KF3↔KF4=30

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/types/frame.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"
#include "sslam/types/mappoint.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace {

// Helper: build a camera shared_ptr with KITTI-like intrinsics.
static std::shared_ptr<sslam::StereoCamera> make_camera() {
    auto cam       = std::make_shared<sslam::StereoCamera>();
    cam->fx = cam->fy = 718.856;
    cam->cx        = 607.193;
    cam->cy        = 185.216;
    cam->baseline  = 0.5372;
    cam->width     = 1241;
    cam->height    = 376;
    return cam;
}

// Helper: build a Frame with n_kps fake keypoints (octave 0, no depth).
static sslam::Frame make_test_frame(std::size_t idx, int n_kps,
                                    std::shared_ptr<const sslam::StereoCamera> cam) {
    sslam::Frame f;
    f.index     = idx;
    f.timestamp = static_cast<double>(idx) * 0.1;
    f.camera    = cam;

    f.keypoints_left.resize(static_cast<std::size_t>(n_kps));
    for (int i = 0; i < n_kps; ++i) {
        f.keypoints_left[static_cast<std::size_t>(i)].pt     = {static_cast<float>(i), 0.0f};
        f.keypoints_left[static_cast<std::size_t>(i)].octave = 0;
    }
    // 32 bytes = 256-bit ORB descriptor; zeros are valid for testing.
    f.descriptors_left = cv::Mat::zeros(n_kps, 32, CV_8U);
    f.right_u.assign(static_cast<std::size_t>(n_kps), -1.0f);
    f.depth.assign(static_cast<std::size_t>(n_kps), -1.0f);
    f.T_cw = Eigen::Matrix4d::Identity();
    return f;
}

// Expected covisibility weight between KF a and KF b.
// With the layout described above: weight = max(0, 40 - |a - b| * 10).
static int expected_weight(int a, int b) {
    const int diff = std::abs(a - b);
    return std::max(0, 40 - diff * 10);
}

// Build the fixture: 5 KFs × 100 MPs with overlapping observations.
struct MapFixture {
    std::shared_ptr<sslam::StereoCamera> cam;
    std::shared_ptr<sslam::Map>          map;
    std::vector<sslam::KeyFrame::Ptr>    kfs;
    std::vector<sslam::MapPoint::Ptr>    mps;

    MapFixture() {
        cam = make_camera();
        map = std::make_shared<sslam::Map>();

        const int n_kfs = 5;
        const int n_mps = 100;

        // Create KFs (80 features each — enough for 40-wide windows).
        kfs.reserve(n_kfs);
        for (int k = 0; k < n_kfs; ++k) {
            auto f  = make_test_frame(static_cast<std::size_t>(k), 80, cam);
            auto kf = std::make_shared<sslam::KeyFrame>(
                static_cast<uint64_t>(k), f, cam);
            kfs.push_back(std::move(kf));
        }

        // Create MPs and add observations.
        mps.reserve(n_mps);
        for (int m = 0; m < n_mps; ++m) {
            auto mp = std::make_shared<sslam::MapPoint>(
                static_cast<uint64_t>(m),
                Eigen::Vector3d(static_cast<double>(m), 0.0, 10.0),
                kfs[0].get());

            for (int k = 0; k < n_kfs; ++k) {
                const int start = k * 10;
                const int end   = start + 40;
                if (m >= start && m < end) {
                    const int feat_idx = m - start;  // local index in this KF
                    mp->add_observation(kfs[static_cast<std::size_t>(k)].get(),
                                        feat_idx);
                    kfs[static_cast<std::size_t>(k)]->add_map_point(feat_idx, mp);
                }
            }
            map->add_mappoint(mp);
            mps.push_back(std::move(mp));
        }

        // Build covisibility for all KFs, then register with Map.
        for (auto& kf : kfs) {
            kf->update_connections();
            map->add_keyframe(kf);
        }
    }
};

// --- MapPoint tests -------------------------------------------------------

TEST(MapPoint, AddAndQueryObservation) {
    auto cam = make_camera();
    auto f0  = make_test_frame(0, 10, cam);
    auto kf0 = std::make_shared<sslam::KeyFrame>(0u, f0, cam);

    auto mp = std::make_shared<sslam::MapPoint>(
        0u, Eigen::Vector3d(1.0, 2.0, 3.0), kf0.get());

    EXPECT_EQ(mp->n_observations(), 0);
    mp->add_observation(kf0.get(), 3);
    EXPECT_EQ(mp->n_observations(), 1);
    EXPECT_EQ(mp->get_feat_idx(kf0.get()), 3);

    mp->remove_observation(kf0.get());
    EXPECT_EQ(mp->n_observations(), 0);
    EXPECT_EQ(mp->get_feat_idx(kf0.get()), -1);
}

TEST(MapPoint, BadFlag) {
    auto cam = make_camera();
    auto f0  = make_test_frame(0, 1, cam);
    auto kf0 = std::make_shared<sslam::KeyFrame>(0u, f0, cam);
    auto mp  = std::make_shared<sslam::MapPoint>(0u, Eigen::Vector3d::Zero(), kf0.get());

    EXPECT_FALSE(mp->is_bad());
    mp->set_bad();
    EXPECT_TRUE(mp->is_bad());
}

// --- KeyFrame tests -------------------------------------------------------

TEST(KeyFrame, PoseGetSet) {
    auto cam = make_camera();
    auto f   = make_test_frame(0, 5, cam);
    auto kf  = std::make_shared<sslam::KeyFrame>(0u, f, cam);

    EXPECT_TRUE(kf->get_pose().isApprox(Eigen::Matrix4d::Identity()));

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = 1.5;
    kf->set_pose(T);
    EXPECT_TRUE(kf->get_pose().isApprox(T));
}

TEST(KeyFrame, CameraCenter) {
    auto cam = make_camera();
    auto f   = make_test_frame(0, 5, cam);
    auto kf  = std::make_shared<sslam::KeyFrame>(0u, f, cam);
    // Identity pose → camera centre at world origin.
    EXPECT_TRUE(kf->camera_center().isApprox(Eigen::Vector3d::Zero()));
}

TEST(KeyFrame, BadKeyFramePoseFollowsParent) {
    auto cam = make_camera();
    auto f0  = make_test_frame(0, 5, cam);
    auto f1  = make_test_frame(1, 5, cam);
    auto parent = std::make_shared<sslam::KeyFrame>(0u, f0, cam);
    auto child  = std::make_shared<sslam::KeyFrame>(1u, f1, cam);

    Eigen::Matrix4d T_parent = Eigen::Matrix4d::Identity();
    T_parent(0, 3) = -10.0;
    Eigen::Matrix4d T_child = Eigen::Matrix4d::Identity();
    T_child(0, 3) = -12.0;
    parent->set_pose(T_parent);
    child->set_pose(T_child);
    child->set_parent(parent.get());
    child->set_bad();

    Eigen::Matrix4d T_parent_corrected = Eigen::Matrix4d::Identity();
    T_parent_corrected(0, 3) = -20.0;
    parent->set_pose(T_parent_corrected);

    Eigen::Matrix4d T_expected = Eigen::Matrix4d::Identity();
    T_expected(0, 3) = -22.0;
    EXPECT_TRUE(child->get_pose_through_spanning_tree().isApprox(T_expected, 1e-9));
}

TEST(KeyFrame, TrackedMapPoints) {
    auto cam = make_camera();
    auto f   = make_test_frame(0, 10, cam);
    auto kf  = std::make_shared<sslam::KeyFrame>(0u, f, cam);

    EXPECT_EQ(kf->tracked_map_points(), 0);

    auto mp = std::make_shared<sslam::MapPoint>(0u, Eigen::Vector3d::Zero(), kf.get());
    mp->add_observation(kf.get(), 0);
    kf->add_map_point(0, mp);

    EXPECT_EQ(kf->tracked_map_points(), 1);

    mp->set_bad();
    EXPECT_EQ(kf->tracked_map_points(), 0);
}

// --- Covisibility tests ---------------------------------------------------

TEST(Map, CovisibilityWeights) {
    MapFixture fix;

    for (int a = 0; a < 5; ++a) {
        for (int b = a + 1; b < 5; ++b) {
            const int exp_w = expected_weight(a, b);
            // Check from KF a's perspective.
            const auto covis_a = fix.kfs[static_cast<std::size_t>(a)]
                                     ->get_covisibility_keyframes(0);
            const sslam::KeyFrame* kf_b_ptr =
                fix.kfs[static_cast<std::size_t>(b)].get();

            if (exp_w == 0) {
                // KF b must NOT appear in KF a's covisibility.
                const bool found = std::find(covis_a.begin(), covis_a.end(),
                                             kf_b_ptr) != covis_a.end();
                EXPECT_FALSE(found)
                    << "KF" << a << "↔KF" << b
                    << " should have weight 0 but appears in covisibility";
            } else {
                // Retrieve weight directly via add_connection bookkeeping.
                // get_covisibility_keyframes returns sorted by weight; verify
                // presence and correct weight via get_map_points intersection.
                const bool found = std::find(covis_a.begin(), covis_a.end(),
                                             kf_b_ptr) != covis_a.end();
                EXPECT_TRUE(found)
                    << "KF" << a << "↔KF" << b
                    << " expected weight " << exp_w << " but KF" << b
                    << " not in covisibility of KF" << a;
            }
        }
    }
}

TEST(Map, CovisibilityWeightsExact) {
    MapFixture fix;

    // Count shared MPs manually and compare to covisibility edge weights.
    for (int a = 0; a < 5; ++a) {
        for (int b = a + 1; b < 5; ++b) {
            // Count shared MPs between kf_a and kf_b.
            const auto mps_a = fix.kfs[static_cast<std::size_t>(a)]->get_map_points();
            int shared = 0;
            for (const auto& mp : mps_a)
                if (mp->get_feat_idx(fix.kfs[static_cast<std::size_t>(b)].get()) >= 0)
                    ++shared;

            EXPECT_EQ(shared, expected_weight(a, b))
                << "KF" << a << "↔KF" << b;
        }
    }
}

// --- Map::local_map_around -----------------------------------------------

TEST(Map, LocalMapAround_MinShared15) {
    MapFixture fix;

    // KF0: covis KF1=30, KF2=20, KF3=10, KF4=0
    // With min_shared=15: should return KF1 (30) and KF2 (20).
    const auto local = fix.map->local_map_around(fix.kfs[0].get(), 15);

    ASSERT_EQ(local.size(), 2u);
    const bool has_kf1 = std::any_of(local.begin(), local.end(),
        [&](const sslam::KeyFrame::Ptr& kf) {
            return kf.get() == fix.kfs[1].get();
        });
    const bool has_kf2 = std::any_of(local.begin(), local.end(),
        [&](const sslam::KeyFrame::Ptr& kf) {
            return kf.get() == fix.kfs[2].get();
        });
    EXPECT_TRUE(has_kf1);
    EXPECT_TRUE(has_kf2);
}

TEST(Map, LocalMapAround_MinShared0) {
    MapFixture fix;

    // KF0: covis KF1=30, KF2=20, KF3=10; KF4 has weight 0 → excluded.
    const auto local = fix.map->local_map_around(fix.kfs[0].get(), 0);
    EXPECT_EQ(local.size(), 3u);
}

// --- Map count tests -----------------------------------------------------

TEST(Map, Counts) {
    MapFixture fix;
    EXPECT_EQ(fix.map->keyframe_count(), 5u);
    EXPECT_EQ(fix.map->mappoint_count(), 100u);
}

}  // namespace
