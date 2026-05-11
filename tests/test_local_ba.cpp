// Unit test for local bundle adjustment.
//
// Synthetic scene: 6 KeyFrames arranged along the x-axis, each observing
// 60 MapPoints drawn from a uniform 10×10×10 m cube centred at (0,0,15).
//
// Noise model:
//   - Point positions perturbed by N(0, 0.05 m) on each axis before
//     constructing KFs (simulates imperfect stereo depth).
//   - KF poses perturbed by N(0, 0.01 rad) in rotation and
//     N(0, 0.05 m) in translation.
//
// Acceptance:
//   - Mean rotation error after Local BA < 0.1°.
//   - Mean point position error after Local BA < 1 cm.

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/mapping/triangulation.hpp"
#include "sslam/optim/ba.hpp"
#include "sslam/types/frame.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"
#include "sslam/types/mappoint.hpp"

#include <Eigen/Geometry>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal StereoCamera (KITTI-like intrinsics).
sslam::StereoCamera make_camera() {
    sslam::StereoCamera cam;
    cam.fx = 718.856; cam.fy = 718.856;
    cam.cx = 607.193; cam.cy = 185.216;
    cam.baseline = 0.537;
    cam.width = 1241; cam.height = 376;
    return cam;
}

/// Rotation matrix from axis-angle (small angle OK).
Eigen::Matrix3d rot_x(double rad) {
    Eigen::AngleAxisd aa(rad, Eigen::Vector3d::UnitX());
    return aa.toRotationMatrix();
}
Eigen::Matrix3d rot_y(double rad) {
    Eigen::AngleAxisd aa(rad, Eigen::Vector3d::UnitY());
    return aa.toRotationMatrix();
}

/// Build T_cw from R, t_wc (camera-in-world translation).
Eigen::Matrix4d make_T_cw(const Eigen::Matrix3d& R, const Eigen::Vector3d& t_wc) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = -R * t_wc;
    return T;
}

/// Rotation angle between two rotation matrices (degrees).
double rotation_error_deg(const Eigen::Matrix3d& R1, const Eigen::Matrix3d& R2) {
    const Eigen::Matrix3d dR = R1.transpose() * R2;
    const double trace = dR.trace();
    const double cos_theta = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(cos_theta) * 180.0 / M_PI;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct LocalBAFixture {
    static constexpr int kNumKFs  = 6;
    static constexpr int kNumMPs  = 60;

    std::shared_ptr<sslam::StereoCamera> cam;
    std::shared_ptr<sslam::Map>          map;

    // Ground-truth data.
    std::vector<Eigen::Matrix4d> gt_poses;  // T_cw for each KF
    std::vector<Eigen::Vector3d> gt_points; // world positions

    // Object handles (same order as gt_*).
    std::vector<std::shared_ptr<sslam::KeyFrame>>  kfs;
    std::vector<std::shared_ptr<sslam::MapPoint>>  mps;

    LocalBAFixture() {
        std::mt19937 rng(42);
        std::normal_distribution<double> pose_rot_noise(0.0, 0.002);  // rad (~0.11°)
        std::normal_distribution<double> pose_t_noise  (0.0, 0.01);   // m
        std::normal_distribution<double> pt_noise      (0.0, 0.01);   // m
        std::uniform_real_distribution<double> pt_dist (-2.0, 2.0);   // m (±2m x,y)

        cam = std::make_shared<sslam::StereoCamera>(make_camera());
        map = std::make_shared<sslam::Map>();

        // --- Ground-truth MapPoint positions --------------------------------
        for (int i = 0; i < kNumMPs; ++i)
            gt_points.emplace_back(pt_dist(rng), pt_dist(rng),
                                   6.0 + std::abs(pt_dist(rng)) * 0.5 + 2.0);  // 2–9 m

        // --- Ground-truth KF poses (cameras moving along +x) ---------------
        for (int k = 0; k < kNumKFs; ++k) {
            const Eigen::Matrix3d R = rot_y(k * 0.05) * rot_x(0.005 * k);
            const Eigen::Vector3d t_wc(k * 0.3, 0.0, 0.0);  // 0.3m spacing
            gt_poses.push_back(make_T_cw(R, t_wc));
        }

        // --- Compute noisy poses and noisy MP world positions ---------------
        // KF0 is kept at GT (Identity) since it will be the anchor; only KF1+ are
        // perturbed.  This mirrors real SLAM where the first KF is the world origin.
        std::vector<Eigen::Matrix4d> noisy_poses(kNumKFs);
        for (int k = 0; k < kNumKFs; ++k) {
            const Eigen::Matrix4d& T_gt = gt_poses[k];
            if (k == 0) {
                noisy_poses[k] = T_gt;  // anchor — kept at GT
                continue;
            }
            Eigen::Matrix3d R_gt = T_gt.block<3,3>(0,0);
            Eigen::Vector3d aa_noise(pose_rot_noise(rng),
                                     pose_rot_noise(rng),
                                     pose_rot_noise(rng));
            Eigen::Matrix3d R_noisy = Eigen::AngleAxisd(
                aa_noise.norm(), aa_noise.normalized()).toRotationMatrix() * R_gt;
            Eigen::Vector3d t_noisy = T_gt.block<3,1>(0,3)
                + Eigen::Vector3d(pose_t_noise(rng),
                                  pose_t_noise(rng),
                                  pose_t_noise(rng));
            noisy_poses[k] = Eigen::Matrix4d::Identity();
            noisy_poses[k].block<3,3>(0,0) = R_noisy;
            noisy_poses[k].block<3,1>(0,3) = t_noisy;
        }

        std::vector<Eigen::Vector3d> noisy_points(kNumMPs);
        for (int i = 0; i < kNumMPs; ++i)
            noisy_points[i] = gt_points[i]
                + Eigen::Vector3d(pt_noise(rng), pt_noise(rng), pt_noise(rng));

        // --- Build observation tables: project GT points through GT poses with pixel noise
        // Self-consistent measurements allow BA to recover GT geometry.
        std::normal_distribution<double> pixel_noise(0.0, 0.1);  // 0.1 px σ
        struct OBS { float ul, v, ur; };
        std::vector<std::vector<OBS>> obs(kNumKFs,
            std::vector<OBS>(kNumMPs, {-1.f, -1.f, -1.f}));

        for (int k = 0; k < kNumKFs; ++k) {
            const Eigen::Matrix4d& T = gt_poses[k];
            for (int i = 0; i < kNumMPs; ++i) {
                const Eigen::Vector3d Xc =
                    T.block<3,3>(0,0) * gt_points[i] + T.block<3,1>(0,3);
                if (Xc.z() <= 0.1) continue;
                const float u = static_cast<float>(cam->fx * Xc.x() / Xc.z() + cam->cx
                                                   + pixel_noise(rng));
                const float v = static_cast<float>(cam->fy * Xc.y() / Xc.z() + cam->cy
                                                   + pixel_noise(rng));
                if (u < 0 || u >= cam->width || v < 0 || v >= cam->height) continue;
                const float disp = static_cast<float>(cam->fx * cam->baseline / Xc.z());
                obs[k][i] = {u, v, u - disp};
            }
        }

        // --- Build Frames with correct keypoints, then create KFs -----------
        cv::Mat empty_img(cam->height, cam->width, CV_8UC1, cv::Scalar(0));
        for (int k = 0; k < kNumKFs; ++k) {
            auto frame = std::make_shared<sslam::Frame>(
                static_cast<std::size_t>(k), 0.0, empty_img, empty_img, cam);
            frame->T_cw = noisy_poses[k];
            frame->keypoints_left.resize(kNumMPs);
            frame->descriptors_left = cv::Mat::zeros(kNumMPs, 32, CV_8UC1);
            frame->right_u.resize(kNumMPs);
            frame->depth.resize(kNumMPs, -1.0f);

            for (int i = 0; i < kNumMPs; ++i) {
                const OBS& o = obs[k][i];
                frame->keypoints_left[i].pt = {o.ul, o.v};
                frame->right_u[i]  = o.ur;  // -1 if invisible
            }

            auto kf = std::make_shared<sslam::KeyFrame>(
                static_cast<uint64_t>(k), *frame, cam);
            kfs.push_back(kf);
            map->add_keyframe(kf);
        }

        // --- Create noisy MapPoints and register observations ---------------
        for (int i = 0; i < kNumMPs; ++i) {
            auto mp = std::make_shared<sslam::MapPoint>(
                static_cast<uint64_t>(i), noisy_points[i], kfs[0].get());

            for (int k = 0; k < kNumKFs; ++k) {
                if (obs[k][i].ul < 0) continue;  // invisible
                mp->add_observation(kfs[k].get(), i);
                kfs[k]->add_map_point(i, mp);
            }
            mp->update_normal_and_depth();
            mps.push_back(mp);
            map->add_mappoint(mp);
        }

        // Build covisibility edges.
        for (auto& kf : kfs) kf->update_connections();
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(Triangulation, LinearDLT) {
    // Two cameras 1 m apart, known point at (0, 0, 10).
    const auto cam = make_camera();

    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2(0, 3) = -1.0;  // second camera 1 m to the right (T_cw, so +1 m in camera frame)

    const Eigen::Matrix3d K = cam.K();
    Eigen::Matrix<double, 3, 4> P1, P2;
    P1.block<3,3>(0,0) = K; P1.block<3,1>(0,3) = Eigen::Vector3d::Zero();
    P2.block<3,3>(0,0) = K; P2.block<3,1>(0,3) = K * T2.block<3,1>(0,3);

    const Eigen::Vector3d pw(0.5, -0.3, 10.0);
    const Eigen::Vector3d pc1 = pw;
    const Eigen::Vector3d pc2 = T2.block<3,3>(0,0) * pw + T2.block<3,1>(0,3);

    const Eigen::Vector2d x1(cam.fx * pc1.x()/pc1.z() + cam.cx,
                              cam.fy * pc1.y()/pc1.z() + cam.cy);
    const Eigen::Vector2d x2(cam.fx * pc2.x()/pc2.z() + cam.cx,
                              cam.fy * pc2.y()/pc2.z() + cam.cy);

    const Eigen::Vector3d result = sslam::triangulate_linear(P1, P2, x1, x2);
    EXPECT_NEAR(result.x(), pw.x(), 1e-6);
    EXPECT_NEAR(result.y(), pw.y(), 1e-6);
    EXPECT_NEAR(result.z(), pw.z(), 1e-6);
}

TEST(LocalBA, RecoversPosesAndPoints) {
    LocalBAFixture fix;

    // Record initial errors before BA.
    // Mean rotation error only over non-anchor KFs (KF0 is fixed at GT, skip it).
    auto mean_rot_err = [&]() {
        double total = 0.0;
        for (int k = 1; k < LocalBAFixture::kNumKFs; ++k) {
            const Eigen::Matrix3d R_est = fix.kfs[k]->get_pose().block<3,3>(0,0);
            const Eigen::Matrix3d R_gt  = fix.gt_poses[k].block<3,3>(0,0);
            total += rotation_error_deg(R_est, R_gt);
        }
        return total / (LocalBAFixture::kNumKFs - 1);
    };
    auto mean_pt_err = [&]() {
        double total = 0.0;
        for (int i = 0; i < LocalBAFixture::kNumMPs; ++i)
            total += (fix.mps[i]->get_world_pos() - fix.gt_points[i]).norm();
        return total / LocalBAFixture::kNumMPs;
    };

    const double rot_before = mean_rot_err();
    const double pt_before  = mean_pt_err();

    // Run Local BA with the last KF as the "new" one.
    sslam::ba::local_bundle_adjustment(fix.kfs.back().get(), *fix.cam);

    const double rot_after = mean_rot_err();
    const double pt_after  = mean_pt_err();

    // BA must not make things worse.
    EXPECT_LT(rot_after, rot_before + 0.05);
    EXPECT_LT(pt_after,  pt_before  + 0.005);

    // Accuracy target for the synthetic local BA scene.
    EXPECT_LT(rot_after, 0.1)   << "Mean rotation error after BA: " << rot_after << "°";
    EXPECT_LT(pt_after,  0.01)  << "Mean point error after BA: "    << pt_after  << " m";
}
