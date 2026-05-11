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
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"
#include "sslam/types/mappoint.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
      feature_matcher_(cam, p.feature),
      map_(std::make_shared<Map>()),
      local_mapping_(std::make_shared<LocalMapping>(map_, cam)) {
    if (!cam_) throw std::runtime_error("Tracking: null camera pointer");
    local_mapping_->start();
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
        // Insert the first KeyFrame with no predecessor matches.
        const std::vector<std::pair<int, int>> no_matches;
        maybe_insert_keyframe(*frame, no_matches, 0);
        anchor_and_record_(*frame);
        prev_frame_ = frame;
        return {frame, state_, n_stereo, 0, 0};
    }

    // --- Step 2.5: corrected-frame prediction anchor for match_local_map_ --
    // T_prev_corrected is used only by match_local_map_ (currently disabled).
    // It must NOT replace prev_frame_->T_cw; that would inject BA
    // discontinuities into the motion model and break tracking.
    Eigen::Matrix4d T_prev_corrected = prev_frame_->T_cw;
    if (prev_frame_->ref_kf && !prev_frame_->ref_kf->is_bad())
        T_prev_corrected = prev_frame_->T_ref * prev_frame_->ref_kf->get_pose();
    (void)T_prev_corrected;  // unused until match_local_map_ is re-enabled

    // --- Step 3: predict pose with constant-velocity model -----------------
    const Eigen::Matrix4d T_pred = motion_model_.predict(prev_frame_->T_cw);

    // Camera intrinsics used in PnP and motion-only BA.
    const double fx = cam_->fx, fy = cam_->fy;
    const double cx = cam_->cx, cy = cam_->cy;

    // --- Step 4: build 3-D/2-D correspondences for PnP --------------------
    // Frame-to-frame projection matching against prev_frame_ in the raw VO frame.
    // match_local_map_ (local-map projection) is implemented but disabled until
    // the two-stage tracker is properly integrated.

    std::vector<cv::Point3f>     pts3d;
    std::vector<cv::Point2f>     pts2d;
    std::vector<Eigen::Vector3d> obs_stereo;
    int n_matches = 0;
    std::vector<std::pair<int,int>> frame_matches;  // populated only in fallback path

    // --- Stage 1: frame-to-frame projection matching ----------------------
    {
        // --- Frame-to-frame projection matching ----------------------------
        const float radius_scale =
            (state_ == TrackingState::LOST) ? params_.wide_radius_scale * 2.0f
                                            : params_.wide_radius_scale;
        frame_matches = feature_matcher_.match_by_projection(
            *prev_frame_, *frame, T_pred, radius_scale);
        n_matches = static_cast<int>(frame_matches.size());

        if (n_matches < 4) {
            frame->T_cw = prev_frame_->T_cw;
            motion_model_.reset();
            anchor_and_record_(*frame);
            prev_frame_ = frame;
            state_ = TrackingState::LOST;
            return {frame, state_, n_stereo, n_matches, 0};
        }

        // Build PnP correspondences from frame-to-frame matches.
        // Both prev_frame_->T_cw and the resulting frame->T_cw are in the
        // raw VO world frame; BA corrections reach the output via resolved_trajectory().
        const double bf = cam_->fx * cam_->baseline;

        const Eigen::Matrix3d R_prev_T =
            prev_frame_->T_cw.topLeftCorner<3, 3>().transpose();
        const Eigen::Vector3d t_prev_wc =
            -R_prev_T * prev_frame_->T_cw.topRightCorner<3, 1>();

        pts3d.reserve(n_matches);
        pts2d.reserve(n_matches);
        obs_stereo.reserve(n_matches);

        for (const auto& [pi, ci] : frame_matches) {
            const float d = prev_frame_->depth[pi];
            if (d <= 0.0f) continue;
            const cv::KeyPoint& kp_prev = prev_frame_->keypoints_left[pi];
            const cv::KeyPoint& kp_curr = frame->keypoints_left[ci];
            const Eigen::Vector3d p_c(
                (static_cast<double>(kp_prev.pt.x) - cx) * d / fx,
                (static_cast<double>(kp_prev.pt.y) - cy) * d / fy,
                static_cast<double>(d));
            const Eigen::Vector3d p_w = R_prev_T * p_c + t_prev_wc;
            pts3d.push_back({static_cast<float>(p_w.x()),
                             static_cast<float>(p_w.y()),
                             static_cast<float>(p_w.z())});
            pts2d.push_back(kp_curr.pt);
            const double u_r =
                static_cast<double>(kp_curr.pt.x) - static_cast<double>(bf) / d;
            obs_stereo.push_back({static_cast<double>(kp_curr.pt.x),
                                  static_cast<double>(kp_curr.pt.y),
                                  u_r});
        }

        if (static_cast<int>(pts3d.size()) < 4) {
            frame->T_cw = prev_frame_->T_cw;
            motion_model_.reset();
            anchor_and_record_(*frame);
            prev_frame_ = frame;
            state_ = TrackingState::LOST;
            return {frame, state_, n_stereo, n_matches, 0};
        }
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
        anchor_and_record_(*frame);
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

    // --- Step 5d: sanity check — reject catastrophic pose outliers ---------
    // PnP+RANSAC can pick the "wrong" one of two mirror solutions when the
    // scene is nearly planar.  Guard against this by checking the camera-centre
    // jump.  Both poses are in the raw VO frame so the comparison is consistent.
    {
        const Eigen::Matrix3d R_c  = frame->T_cw.topLeftCorner<3, 3>();
        const Eigen::Vector3d c_c  = -R_c.transpose() * frame->T_cw.topRightCorner<3, 1>();
        const Eigen::Matrix3d R_p  = prev_frame_->T_cw.topLeftCorner<3, 3>();
        const Eigen::Vector3d c_p  = -R_p.transpose() * prev_frame_->T_cw.topRightCorner<3, 1>();
        const double jump = (c_c - c_p).norm();
        if (jump > static_cast<double>(params_.max_frame_translation)) {
            frame->T_cw = prev_frame_->T_cw;
            motion_model_.reset();
            anchor_and_record_(*frame);
            prev_frame_ = frame;
            state_ = TrackingState::LOST;
            return {frame, state_, n_stereo, n_matches, n_inliers};
        }
    }

    // --- Step 7: KeyFrame insertion check ---------------------------------
    maybe_insert_keyframe(*frame, frame_matches, n_inliers);

    // --- Step 6: update motion model and advance prev frame ---------------
    motion_model_.update(frame->T_cw, prev_frame_->T_cw);
    anchor_and_record_(*frame);
    prev_frame_ = frame;
    state_      = TrackingState::OK;

    return {frame, state_, n_stereo, n_matches, n_inliers};
}

// ---------------------------------------------------------------------------
// Anchor the frame to the latest KeyFrame and append a trajectory entry.
// ---------------------------------------------------------------------------

void Tracking::anchor_and_record_(Frame& frame) {
    if (last_kf_ && !last_kf_->is_bad()) {
        const Eigen::Matrix4d T_kw = last_kf_->get_pose();
        // SE(3) inverse: T_wk = [R^T | -R^T t]
        const Eigen::Matrix3d R_T = T_kw.topLeftCorner<3, 3>().transpose();
        Eigen::Matrix4d T_wk      = Eigen::Matrix4d::Identity();
        T_wk.topLeftCorner<3, 3>() = R_T;
        T_wk.topRightCorner<3, 1>() = -R_T * T_kw.topRightCorner<3, 1>();
        frame.ref_kf = last_kf_.get();
        frame.T_ref  = frame.T_cw * T_wk;
        trajectory_entries_.push_back({last_kf_.get(), frame.T_ref, frame.T_cw});
    } else {
        frame.ref_kf = nullptr;
        frame.T_ref  = Eigen::Matrix4d::Identity();
        trajectory_entries_.push_back({nullptr, Eigen::Matrix4d::Identity(), frame.T_cw});
    }
}

std::vector<Eigen::Matrix4d> Tracking::resolved_trajectory() const {
    std::vector<Eigen::Matrix4d> out;
    out.reserve(trajectory_entries_.size());
    for (const auto& e : trajectory_entries_) {
        if (e.ref_kf && !e.ref_kf->is_bad()) {
            out.push_back(e.T_ref * e.ref_kf->get_pose());
        } else {
            out.push_back(e.T_cw_fallback);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// match_local_map_
// ---------------------------------------------------------------------------
// Projects MapPoints from last_kf_ into the current frame using T_pred
// (the current pose estimate in the BA-corrected world frame).  Returns true
// iff match count >= params_.min_local_map_matches.
//
// Disabled until a proper two-stage tracker (TrackWithMotionModel then
// TrackLocalMap) is implemented.
// ---------------------------------------------------------------------------
bool Tracking::match_local_map_(
    const Frame&                  frame,
    const Eigen::Matrix4d&        T_pred,
    std::vector<cv::Point3f>&     pts3d,
    std::vector<cv::Point2f>&     pts2d,
    std::vector<Eigen::Vector3d>& obs_stereo)
{
    if (!last_kf_ || last_kf_->is_bad()) return false;

    // Candidate set: last_kf_ only.
    // The covisibility neighbourhood was tried but brought too many wrong
    // associations even with Stage 1 providing an accurate initial pose:
    // dense urban scenes have many MPs projecting within the 35 px radius,
    // and RANSAC can accept a wrong-consensus set that accidentally has low
    // reprojection error.
    std::vector<KeyFrame*> local_kfs{last_kf_.get()};

    // 2. Collect unique non-bad local MPs.
    std::unordered_set<MapPoint*> seen;
    std::vector<MapPoint::Ptr>    local_mps;
    for (KeyFrame* kf : local_kfs) {
        for (const auto& mp : kf->get_map_points()) {
            if (!mp || mp->is_bad()) continue;
            if (seen.insert(mp.get()).second)
                local_mps.push_back(mp);
        }
    }
    if (local_mps.empty()) return false;

    // Precompute projection constants.
    const double fx = cam_->fx, fy = cam_->fy;
    const double cx = cam_->cx, cy = cam_->cy;
    const double bf = cam_->fx * cam_->baseline;
    const Eigen::Matrix3d R_pred = T_pred.topLeftCorner<3, 3>();
    const Eigen::Vector3d t_pred = T_pred.topRightCorner<3, 1>();

    const auto& kps   = frame.keypoints_left;
    const auto& descs = frame.descriptors_left;
    const int   n_kp  = static_cast<int>(kps.size());

    std::vector<bool> matched(static_cast<std::size_t>(n_kp), false);

    constexpr int   kTH_HIGH       = 100;
    constexpr float kNarrowRadius2 = 1225.0f;  // 35 px — normal tracking
    constexpr float kWideRadius2   = 4900.0f;  // 70 px — after LOST or first frames
    constexpr float kLoweRatio     = 0.70f;

    // After a LOST recovery the velocity model is reset to identity, so the
    // prediction error can be large.  Use a wider radius in that case.
    const float kRadius2 = (state_ == TrackingState::LOST)
                           ? kWideRadius2 : kNarrowRadius2;

    for (const auto& mp : local_mps) {
        // Project MP into the current frame using the predicted pose.
        const Eigen::Vector3d X_c = R_pred * mp->get_world_pos() + t_pred;
        if (X_c.z() <= 0.0) continue;

        const float u_proj = static_cast<float>(fx * X_c.x() / X_c.z() + cx);
        const float v_proj = static_cast<float>(fy * X_c.y() / X_c.z() + cy);
        if (u_proj < 0.0f || u_proj >= static_cast<float>(cam_->width) ||
            v_proj < 0.0f || v_proj >= static_cast<float>(cam_->height))
            continue;

        const cv::Mat mp_desc = mp->get_descriptor();
        if (mp_desc.empty()) continue;

        // 2-best search for Lowe ratio filtering.
        int best_dist = kTH_HIGH + 1, second_dist = kTH_HIGH + 1, best_j = -1;
        for (int j = 0; j < n_kp; ++j) {
            if (matched[static_cast<std::size_t>(j)]) continue;
            const float du = kps[static_cast<std::size_t>(j)].pt.x - u_proj;
            const float dv = kps[static_cast<std::size_t>(j)].pt.y - v_proj;
            if (du * du + dv * dv > kRadius2) continue;
            const int dist = cv::norm(mp_desc, descs.row(j), cv::NORM_HAMMING);
            if (dist < best_dist) {
                second_dist = best_dist;
                best_dist   = dist;
                best_j      = j;
            } else if (dist < second_dist) {
                second_dist = dist;
            }
        }

        if (best_j < 0 || best_dist >= kTH_HIGH) continue;
        if (second_dist < kTH_HIGH &&
            best_dist > static_cast<int>(kLoweRatio * second_dist))
            continue;

        // Record correspondence.
        matched[static_cast<std::size_t>(best_j)] = true;
        const Eigen::Vector3d p_w = mp->get_world_pos();
        pts3d.push_back({static_cast<float>(p_w.x()),
                         static_cast<float>(p_w.y()),
                         static_cast<float>(p_w.z())});
        pts2d.push_back(kps[static_cast<std::size_t>(best_j)].pt);

        // Right-camera observation: prefer the actual stereo match stored on
        // the frame over the depth derived from the predicted projection.
        // X_c.z() is the predicted depth (consistent with T_pred, not the
        // final optimized pose), so using it for u_r would bias BA.
        const double u_l = static_cast<double>(kps[static_cast<std::size_t>(best_j)].pt.x);
        const double v_l = static_cast<double>(kps[static_cast<std::size_t>(best_j)].pt.y);
        const float  d_actual = frame.depth[static_cast<std::size_t>(best_j)];
        const double u_r = (d_actual > 0.0f)
            ? u_l - bf / static_cast<double>(d_actual)   // real stereo observation
            : u_l - bf / X_c.z();                         // fallback: projected depth
        obs_stereo.push_back({u_l, v_l, u_r});
    }

    return static_cast<int>(pts3d.size()) >= params_.min_local_map_matches;
}

// ---------------------------------------------------------------------------
// KeyFrame insertion helpers
// ---------------------------------------------------------------------------

void Tracking::maybe_insert_keyframe(
    const Frame& curr_frame,
    const std::vector<std::pair<int, int>>& /*matches*/,
    int n_inliers) {

    ++nframes_since_kf_;

    // Criterion A: always insert the very first KeyFrame.
    const bool must_insert = !last_kf_;

    // Criterion B: time-based cadence.
    const bool time_crit = (nframes_since_kf_ > params_.kf_max_frames_since);

    // Criterion C: minimum tracking quality.
    // Guard n_inliers > 0 so the forced-zero initialisation call doesn't fire it.
    const bool min_crit = (n_inliers > 0 &&
                           n_inliers < params_.kf_min_tracked_points);

    if (!must_insert && !time_crit && !min_crit) return;

    // --- Compute the BA-corrected pose for this frame ---------------------
    // curr_frame.T_cw is the raw VO pose from PnP.  Rebase via the reference
    // KF so that fresh MPs created here are in the same world frame as the
    // rest of the map.
    Eigen::Matrix4d T_cw_corrected = curr_frame.T_cw;
    if (last_kf_ && !last_kf_->is_bad()) {
        const Eigen::Matrix4d T_kw = last_kf_->get_pose();
        const Eigen::Matrix3d R_T  = T_kw.topLeftCorner<3,3>().transpose();
        Eigen::Matrix4d T_wk = Eigen::Matrix4d::Identity();
        T_wk.topLeftCorner<3,3>()  = R_T;
        T_wk.topRightCorner<3,1>() = -R_T * T_kw.topRightCorner<3,1>();
        // T_ref_curr = curr_frame.T_cw * T_wk (relative to last_kf_ at PnP time)
        const Eigen::Matrix4d T_ref_curr = curr_frame.T_cw * T_wk;
        T_cw_corrected = T_ref_curr * T_kw;
    }

    // --- Build new KeyFrame -----------------------------------------------
    auto kf = std::make_shared<KeyFrame>(next_kf_id_++, curr_frame, cam_);
    kf->set_pose(T_cw_corrected);

    // --- Propagate MPs from last_kf_ via projection + descriptor matching --
    // For each MP in last_kf_, project into curr_frame using curr_frame.T_cw,
    // then scan curr_frame features within a 20 px radius for the best Hamming
    // match. This propagates MPs across the inter-KF gap without requiring a
    // continuous frame chain, at O(N_MPs × N_feats_in_radius) cost per KF
    // insertion (acceptable since KFs are rare — every kf_max_frames_since frames).
    std::vector<bool> curr_matched(curr_frame.num_features(), false);

    if (last_kf_) {
        const Eigen::Matrix3d R_cw = T_cw_corrected.topLeftCorner<3, 3>();
        const Eigen::Vector3d t_cw = T_cw_corrected.topRightCorner<3, 1>();

        const auto& kps_curr   = curr_frame.keypoints_left;
        const auto& desc_curr  = curr_frame.descriptors_left;
        const int   n_curr     = static_cast<int>(kps_curr.size());
        const float search_r2  = 50.0f * 50.0f;  // 50 px radius (squared)
        constexpr int kHammingTh = 100;           // TH_HIGH for ORB

        for (int i = 0; i < static_cast<int>(last_kf_->num_features()); ++i) {
            auto mp = last_kf_->get_map_point(i);
            if (!mp || mp->is_bad()) continue;

            // Project into curr_frame.
            const Eigen::Vector3d X_c = R_cw * mp->get_world_pos() + t_cw;
            if (X_c.z() <= 0.0) continue;
            const float u_proj = static_cast<float>(
                cam_->fx * X_c.x() / X_c.z() + cam_->cx);
            const float v_proj = static_cast<float>(
                cam_->fy * X_c.y() / X_c.z() + cam_->cy);
            if (u_proj < 0.0f || u_proj >= cam_->width ||
                v_proj < 0.0f || v_proj >= cam_->height) continue;

            // Find best unmatched feature in curr_frame within search radius.
            const cv::Mat desc_i = last_kf_->descriptors_left().row(i);
            int best_dist  = kHammingTh + 1;
            int best_j     = -1;
            for (int j = 0; j < n_curr; ++j) {
                if (curr_matched[static_cast<std::size_t>(j)]) continue;
                const float du = kps_curr[static_cast<std::size_t>(j)].pt.x - u_proj;
                const float dv = kps_curr[static_cast<std::size_t>(j)].pt.y - v_proj;
                if (du * du + dv * dv > search_r2) continue;
                const int dist = cv::norm(
                    desc_i, desc_curr.row(j), cv::NORM_HAMMING);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_j    = j;
                }
            }

            if (best_j >= 0) {
                kf->add_map_point(best_j, mp);
                mp->add_observation(kf.get(), best_j);
                curr_matched[static_cast<std::size_t>(best_j)] = true;
            }
        }
    }

    // --- Create fresh MPs for unmatched stereo-depth features ---------------
    const Eigen::Matrix3d R_wc =
        T_cw_corrected.topLeftCorner<3, 3>().transpose();
    const Eigen::Vector3d t_wc =
        -R_wc * T_cw_corrected.topRightCorner<3, 1>();

    for (std::size_t i = 0; i < curr_frame.num_features(); ++i) {
        if (curr_matched[i]) continue;
        const float d = curr_frame.depth[i];
        if (d <= 0.0f || !std::isfinite(d)) continue;

        const auto&   kp   = curr_frame.keypoints_left[i];
        const double  disp = cam_->fx * cam_->baseline / d;
        const Eigen::Vector3d X_c = cam_->backproject(kp.pt.x, kp.pt.y, disp);
        const Eigen::Vector3d X_w = R_wc * X_c + t_wc;

        auto mp = std::make_shared<MapPoint>(map_->allocate_mappoint_id(), X_w, kf.get());
        mp->add_observation(kf.get(), static_cast<int>(i));
        mp->compute_descriptor();        // single observation: copies this KF's descriptor
        mp->update_normal_and_depth();   // initialise scale-invariance range for TrackLocalMap
        kf->add_map_point(static_cast<int>(i), mp);
        map_->add_mappoint(mp);
    }

    kf->update_connections();
    map_->add_keyframe(kf);
    local_mapping_->enqueue_keyframe(kf);

    last_kf_           = kf;
    last_kf_frame_idx_ = curr_frame.index;
    nframes_since_kf_  = 0;
}

}  // namespace sslam
