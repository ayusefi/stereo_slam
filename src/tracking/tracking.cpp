// Frame-to-frame stereo visual odometry tracker.
//
// Pipeline per frame:
//   1. ORB extraction on left + right images.
//   2. Stereo L↔R matching (fills frame.depth[]).
//   3. Constant-velocity pose prediction from the motion model.
//   4. Frame-to-frame projection matching against the previous frame.
//   5. PnP RANSAC (EPNP initial, iterative refinement on inliers) → T_cw.
//   6. Motion model update; previous frame replaced.
//
// Reference: ORB-SLAM2 Tracking.cc::TrackWithMotionModel.

#include "sslam/tracking/tracking.hpp"
#include "sslam/optim/ba.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <stdexcept>

namespace sslam {

// ---------------------------------------------------------------------------
// ConstantVelocityMotionModel
// ---------------------------------------------------------------------------

void ConstantVelocityMotionModel::update(const Eigen::Matrix4d& T_curr_cw,
                                         const Eigen::Matrix4d& T_prev_cw) {
    // velocity = T_curr_cw * T_prev_wc
    // SE(3) inverse: T_wc = [R^T | -R^T*t]
    const Eigen::Matrix3d R_prev_T = T_prev_cw.topLeftCorner<3, 3>().transpose();
    Eigen::Matrix4d T_prev_wc      = Eigen::Matrix4d::Identity();
    T_prev_wc.topLeftCorner<3, 3>()  = R_prev_T;
    T_prev_wc.topRightCorner<3, 1>() = -R_prev_T * T_prev_cw.topRightCorner<3, 1>();
    velocity_ = T_curr_cw * T_prev_wc;
    valid_    = true;
}

Eigen::Matrix4d ConstantVelocityMotionModel::predict(
    const Eigen::Matrix4d& T_curr_cw) const {
    if (!valid_) return T_curr_cw;
    return velocity_ * T_curr_cw;
}

// ---------------------------------------------------------------------------
// Tracking
// ---------------------------------------------------------------------------

Tracking::Tracking(std::shared_ptr<const StereoCamera> cam)
    : Tracking(std::move(cam), Params{}) {}

Tracking::Tracking(std::shared_ptr<const StereoCamera> cam, const Params& p)
    : cam_(cam),
      params_(p),
      orb_extractor_(p.orb),
      stereo_matcher_(cam, p.stereo),
      feature_matcher_(cam, p.feature) {
    if (!cam_) throw std::runtime_error("Tracking: null camera pointer");
}

Tracking::FrameResult Tracking::process_frame(std::size_t idx, double ts,
                                               const cv::Mat& left,
                                               const cv::Mat& right) {
    auto frame = std::make_shared<Frame>(idx, ts, left, right, cam_);

    // --- Step 1-2: extract ORB and stereo-match ----------------------------
    std::vector<cv::KeyPoint> right_kps;
    cv::Mat right_descs;
    orb_extractor_.detect(frame->left, frame->keypoints_left, frame->descriptors_left);
    orb_extractor_.detect(frame->right, right_kps, right_descs);
    stereo_matcher_.match(*frame, right_kps, right_descs);

    const int n_stereo = static_cast<int>(
        std::count_if(frame->depth.begin(), frame->depth.end(),
                      [](float d) { return d > 0.0f; }));

    // --- Initialization: accept the first frame at the world origin --------
    if (!prev_frame_) {
        frame->T_cw = Eigen::Matrix4d::Identity();
        state_      = TrackingState::OK;
        prev_frame_ = frame;
        return {frame, state_, n_stereo, 0, 0};
    }

    // --- Step 3: predict pose with constant-velocity model -----------------
    const Eigen::Matrix4d T_pred = motion_model_.predict(prev_frame_->T_cw);

    // --- Step 4: frame-to-frame projection matching ------------------------
    // Use the wide search radius unconditionally: KITTI's 10 Hz capture means
    // even modest turns can shift features 30–50 px between frames, well
    // outside the default 10 px radius.  When the previous frame was LOST,
    // the prediction is unreliable, so escalate to a still-wider search.
    const float radius_scale =
        (state_ == TrackingState::LOST) ? params_.wide_radius_scale * 2.0f
                                        : params_.wide_radius_scale;
    auto matches = feature_matcher_.match_by_projection(
        *prev_frame_, *frame, T_pred, radius_scale);

    const int n_matches = static_cast<int>(matches.size());

    if (n_matches < 4) {
        frame->T_cw = prev_frame_->T_cw;  // hold last known pose; T_pred may be wildly off
        motion_model_.reset();            // stale velocity caused the failure; drop it
        prev_frame_ = frame;
        state_ = TrackingState::LOST;
        return {frame, state_, n_stereo, n_matches, 0};
    }

    // --- Step 5: build 3-D / 2-D correspondences for PnP ------------------
    const double fx = cam_->fx, fy = cam_->fy;
    const double cx = cam_->cx, cy = cam_->cy;

    // SE(3) inverse of prev_frame_->T_cw
    const Eigen::Matrix3d R_prev_T = prev_frame_->T_cw.topLeftCorner<3, 3>().transpose();
    const Eigen::Vector3d t_prev_wc =
        -R_prev_T * prev_frame_->T_cw.topRightCorner<3, 1>();

    std::vector<cv::Point3f>    pts3d;
    std::vector<cv::Point2f>    pts2d;
    std::vector<Eigen::Vector3d> obs_stereo;  // (u_l, v, u_r) for BA
    pts3d.reserve(n_matches);
    pts2d.reserve(n_matches);
    obs_stereo.reserve(n_matches);

    const double bf = cam_->fx * cam_->baseline;

    for (const auto& [pi, ci] : matches) {
        const float d = prev_frame_->depth[pi];
        if (d <= 0.0f) continue;

        const cv::KeyPoint& kp_prev = prev_frame_->keypoints_left[pi];
        const Eigen::Vector3d p_c(
            (static_cast<double>(kp_prev.pt.x) - cx) * d / fx,
            (static_cast<double>(kp_prev.pt.y) - cy) * d / fy,
            static_cast<double>(d));
        const Eigen::Vector3d p_w = R_prev_T * p_c + t_prev_wc;

        pts3d.push_back({static_cast<float>(p_w.x()),
                         static_cast<float>(p_w.y()),
                         static_cast<float>(p_w.z())});

        const cv::KeyPoint& kp_curr = frame->keypoints_left[ci];
        pts2d.push_back(kp_curr.pt);

        // Stereo observation for BA: u_r = u_l - bf/depth.
        // depth here is the stereo depth from the *previous* frame, which
        // gives the 3-D point.  For the current frame we only have the left
        // pixel; derive the virtual u_r from the reconstructed depth.
        const double z_c = p_c.z();
        const double u_r = static_cast<double>(kp_curr.pt.x) - bf / z_c;
        obs_stereo.push_back({static_cast<double>(kp_curr.pt.x),
                               static_cast<double>(kp_curr.pt.y),
                               u_r});
    }

    if (static_cast<int>(pts3d.size()) < 4) {
        frame->T_cw = prev_frame_->T_cw;
        motion_model_.reset();
        prev_frame_ = frame;
        state_ = TrackingState::LOST;
        return {frame, state_, n_stereo, n_matches, 0};
    }

    // Camera matrix (OpenCV convention)
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);

    // --- Step 5a: EPnP RANSAC ----------------------------------------------
    cv::Mat rvec, tvec, inliers_mat;
    const bool ok = cv::solvePnPRansac(
        pts3d, pts2d, K, cv::noArray(),
        rvec, tvec,
        /*useExtrinsicGuess=*/false,
        params_.pnp_max_iterations,
        params_.pnp_reprojection_error,
        static_cast<double>(params_.pnp_confidence),
        inliers_mat,
        cv::SOLVEPNP_EPNP);

    const int n_inliers = ok ? inliers_mat.rows : 0;

    if (!ok || n_inliers < params_.min_inliers_pnp) {
        frame->T_cw = prev_frame_->T_cw;  // hold last known pose; PnP result is unreliable
        motion_model_.reset();            // drop stale velocity so next frame predicts identity
        prev_frame_ = frame;
        state_ = TrackingState::LOST;
        return {frame, state_, n_stereo, n_matches, n_inliers};
    }

    // --- Step 5b: iterative refinement on inliers only ---------------------
    std::vector<cv::Point3f>    pts3d_in;
    std::vector<cv::Point2f>    pts2d_in;
    std::vector<Eigen::Vector3d> obs_stereo_in;
    pts3d_in.reserve(n_inliers);
    pts2d_in.reserve(n_inliers);
    obs_stereo_in.reserve(n_inliers);
    for (int k = 0; k < n_inliers; ++k) {
        const int idx = inliers_mat.at<int>(k);
        pts3d_in.push_back(pts3d[static_cast<std::size_t>(idx)]);
        pts2d_in.push_back(pts2d[static_cast<std::size_t>(idx)]);
        obs_stereo_in.push_back(obs_stereo[static_cast<std::size_t>(idx)]);
    }
    cv::solvePnP(pts3d_in, pts2d_in, K, cv::noArray(),
                 rvec, tvec, /*useExtrinsicGuess=*/true,
                 cv::SOLVEPNP_ITERATIVE);

    // --- Convert rvec/tvec → Eigen::Matrix4d T_cw --------------------------
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);

    Eigen::Matrix3d R_eig;
    cv::cv2eigen(R_cv, R_eig);

    frame->T_cw = Eigen::Matrix4d::Identity();
    frame->T_cw.topLeftCorner<3, 3>()  = R_eig;
    frame->T_cw(0, 3) = tvec.at<double>(0);
    frame->T_cw(1, 3) = tvec.at<double>(1);
    frame->T_cw(2, 3) = tvec.at<double>(2);

    // --- Step 5c: motion-only BA on the inlier set -------------------------
    // Build world-point list from inlier pts3d_in.
    std::vector<Eigen::Vector3d> pts3d_eig;
    pts3d_eig.reserve(static_cast<std::size_t>(n_inliers));
    for (const auto& p : pts3d_in) {
        pts3d_eig.push_back({p.x, p.y, p.z});
    }

    const auto ba_result = ba::optimize_pose(
        frame->T_cw, pts3d_eig, obs_stereo_in, *cam_);

    if (ba_result.n_inliers >= params_.min_inliers_pnp) {
        frame->T_cw = ba_result.T_cw;
    }
    // If BA produced fewer inliers than the threshold the PnP result is kept;
    // the frame is still marked OK since PnP already passed.

    // --- Step 6: update motion model and advance prev frame ----------------
    motion_model_.update(frame->T_cw, prev_frame_->T_cw);
    prev_frame_ = frame;
    state_      = TrackingState::OK;

    return {frame, state_, n_stereo, n_matches, n_inliers};
}

}  // namespace sslam
