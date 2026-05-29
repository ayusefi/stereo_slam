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

#include <DBoW2/BowVector.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <algorithm>
#include <cstdlib>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace sslam {

namespace {

Eigen::Matrix4d inverse_se3(const Eigen::Matrix4d& T_cw) {
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R_wc = T_cw.topLeftCorner<3, 3>().transpose();
    T_wc.topLeftCorner<3, 3>()  = R_wc;
    T_wc.topRightCorner<3, 1>() = -R_wc * T_cw.topRightCorner<3, 1>();
    return T_wc;
}

struct LocalMapMatchCandidate {
    std::size_t mp_idx;
    int         feat_idx;
    int         distance;
    double      pred_depth;
};

}  // namespace

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

    std::shared_lock<std::shared_mutex> map_update_lk(map_->update_mutex_);

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

    // --- Step 2.5: refresh previous frame pose after Local BA --------------
    // LocalMapping may have optimized the reference KF after prev_frame_ was
    // recorded.  ORB-SLAM stores frame poses relative to the reference KF and
    // resolves them through the current KF pose; do the same before using the
    // previous frame for motion prediction and frame-to-frame projection.
    if (prev_frame_->ref_kf) {
        prev_frame_->T_cw =
            prev_frame_->T_ref * prev_frame_->ref_kf->get_pose_through_spanning_tree();
    }

    // --- Step 3: predict pose with constant-velocity model -----------------
    const Eigen::Matrix4d T_pred = motion_model_.predict(prev_frame_->T_cw);

    // Camera intrinsics used in PnP and motion-only BA.
    const double fx = cam_->fx, fy = cam_->fy;
    const double cx = cam_->cx, cy = cam_->cy;

    // --- Step 4: build 3-D/2-D correspondences for PnP --------------------
    // Frame-to-frame projection matching against prev_frame_ in the shared
    // map/world frame.
    std::vector<cv::Point3f>     pts3d;
    std::vector<cv::Point2f>     pts2d;
    std::vector<Eigen::Vector3d> obs_stereo;
    std::vector<int>             octaves_curr;  // per-observation octave (current-frame KP)
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
        // shared map/world frame.
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
            // Use current-frame stereo depth for the right-image observation;
            // fall back to the previous-frame depth when the current feature
            // has no stereo match.
            const float d_curr = frame->depth[static_cast<std::size_t>(ci)];
            const double z_curr = (d_curr > 0.0f)
                ? static_cast<double>(d_curr)
                : static_cast<double>(d);
            const double u_r = static_cast<double>(kp_curr.pt.x) - bf / z_curr;
            obs_stereo.push_back({static_cast<double>(kp_curr.pt.x),
                                  static_cast<double>(kp_curr.pt.y),
                                  u_r});
            octaves_curr.push_back(kp_curr.octave);
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
        // Attempt BoW-based relocalization before giving up.
        if (relocalize_(*frame)) {
            motion_model_.reset();
            anchor_and_record_(*frame);
            prev_frame_ = frame;
            state_ = TrackingState::OK;
            return {frame, state_, n_stereo, n_matches, 0};
        }
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
    std::vector<int>             octaves_in;
    pts3d_in.reserve(n_inliers);
    pts2d_in.reserve(n_inliers);
    obs_stereo_in.reserve(n_inliers);
    octaves_in.reserve(n_inliers);
    for (int k = 0; k < n_inliers; ++k) {
        const int idx = inliers_mat.at<int>(k);
        pts3d_in.push_back(pts3d[static_cast<std::size_t>(idx)]);
        pts2d_in.push_back(pts2d[static_cast<std::size_t>(idx)]);
        obs_stereo_in.push_back(obs_stereo[static_cast<std::size_t>(idx)]);
        if (static_cast<std::size_t>(idx) < octaves_curr.size())
            octaves_in.push_back(octaves_curr[static_cast<std::size_t>(idx)]);
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
        frame->T_cw, pts3d_eig, obs_stereo_in, *cam_, octaves_in);

    if (ba_result.n_inliers >= params_.min_inliers_pnp) {
        frame->T_cw = ba_result.T_cw;
    }
    // If BA produced fewer inliers than the threshold the PnP result is kept;
    // the frame is still marked OK since PnP already passed.
    int n_tracking_inliers = (ba_result.n_inliers >= params_.min_inliers_pnp)
                             ? ba_result.n_inliers : n_inliers;

    // --- Step 5d: sanity check — reject catastrophic pose outliers ---------
    // PnP+RANSAC can pick the "wrong" one of two mirror solutions when the
    // scene is nearly planar.  Guard against this by checking the camera-centre
    // jump.  Both poses are in the shared map/world frame so the comparison is
    // consistent.
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

    // --- Step 6: Stage 2 — TrackLocalMap (refine pose with local map MPs) ---
    // ORB-SLAM-style single world frame: Stage 1 already estimates T_cw in
    // the map/world frame because prev_frame_->T_cw is kept in that frame.
    // Local-map BA therefore writes its optimized pose directly back to the
    // current frame; no raw/corrected bridge is used.

    if (last_kf_ && !last_kf_->is_bad()) {
        const Eigen::Matrix4d T_cw_pred = frame->T_cw;

        std::vector<cv::Point3f>                    pts3d_lm;
        std::vector<cv::Point2f>                    pts2d_lm;
        std::vector<Eigen::Vector3d>                obs_stereo_lm;
        std::vector<std::pair<int, MapPoint::Ptr>>  mp_feat_lm;
        std::vector<int>                            octaves_lm;

        if (match_local_map_(*frame, T_cw_pred,
                             pts3d_lm, pts2d_lm, obs_stereo_lm, mp_feat_lm, octaves_lm)) {
            // Stage 2: motion-only BA directly on the local-map matches.
            // We already have a good map-frame pose from Stage 1, so EPnP
            // RANSAC is redundant — just refine with BA.
            std::vector<Eigen::Vector3d> pts3d_eig2;
            pts3d_eig2.reserve(pts3d_lm.size());
            for (const auto& p : pts3d_lm)
                pts3d_eig2.push_back({p.x, p.y, p.z});

            const auto ba2 = ba::optimize_pose(
                T_cw_pred, pts3d_eig2, obs_stereo_lm, *cam_, octaves_lm);

            if (ba2.n_inliers >= params_.min_inliers_pnp) {
                // Sanity check: reject catastrophic jumps.
                const Eigen::Matrix3d R_s2 = ba2.T_cw.topLeftCorner<3, 3>();
                const Eigen::Vector3d c_s2 =
                    -R_s2.transpose() * ba2.T_cw.topRightCorner<3, 1>();
                const Eigen::Matrix3d R_pp = prev_frame_->T_cw.topLeftCorner<3, 3>();
                const Eigen::Vector3d c_pp =
                    -R_pp.transpose() * prev_frame_->T_cw.topRightCorner<3, 1>();
                if ((c_s2 - c_pp).norm() <=
                    static_cast<double>(params_.max_frame_translation)) {
                    frame->T_cw = ba2.T_cw;
                    n_matches   = static_cast<int>(pts3d_lm.size());
                    n_tracking_inliers = ba2.n_inliers;

                    // Populate frame MP associations for KF insertion seeding.
                    // Use the BA inlier mask to associate only reliable matches.
                    frame->map_points.assign(frame->num_features(), nullptr);
                    const auto n_pairs =
                        static_cast<int>(mp_feat_lm.size());
                    for (int k = 0; k < n_pairs; ++k) {
                        if (static_cast<std::size_t>(k) < ba2.inlier_mask.size()
                            && !ba2.inlier_mask[static_cast<std::size_t>(k)]) continue;
                        const auto& [feat_j, mp_ptr] = mp_feat_lm[static_cast<std::size_t>(k)];
                        if (feat_j >= 0 &&
                            static_cast<std::size_t>(feat_j) < frame->map_points.size())
                            frame->map_points[static_cast<std::size_t>(feat_j)] = mp_ptr;
                    }
                }
            }
        }
    }

    // --- Step 7: KeyFrame insertion check ---------------------------------
    maybe_insert_keyframe(*frame, frame_matches, n_tracking_inliers);

    // --- Step 8: update motion model and advance prev frame ---------------
    motion_model_.update(frame->T_cw, prev_frame_->T_cw);
    anchor_and_record_(*frame);
    prev_frame_ = frame;
    state_      = TrackingState::OK;

    return {frame, state_, n_stereo, n_matches, n_tracking_inliers};
}

// ---------------------------------------------------------------------------
// Anchor the frame to the latest KeyFrame and append a trajectory entry.
// ---------------------------------------------------------------------------

void Tracking::anchor_and_record_(Frame& frame) {
    if (last_kf_ && !last_kf_->is_bad()) {
        frame.ref_kf = last_kf_.get();
        frame.T_ref  = frame.T_cw * inverse_se3(last_kf_->get_pose());
        trajectory_entries_.push_back({last_kf_, frame.T_ref, frame.T_cw});
    } else {
        frame.ref_kf = nullptr;
        frame.T_ref  = Eigen::Matrix4d::Identity();
        trajectory_entries_.push_back({KeyFrame::Ptr{}, Eigen::Matrix4d::Identity(), frame.T_cw});
    }
}

std::vector<Eigen::Matrix4d> Tracking::resolved_trajectory() const {
    std::vector<Eigen::Matrix4d> out;
    out.reserve(trajectory_entries_.size());
    for (const auto& e : trajectory_entries_) {
        if (e.ref_kf) {
            out.push_back(e.T_ref * e.ref_kf->get_pose_through_spanning_tree());
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
// ---------------------------------------------------------------------------
bool Tracking::match_local_map_(
    const Frame&                  frame,
    const Eigen::Matrix4d&        T_pred,
    std::vector<cv::Point3f>&     pts3d,
    std::vector<cv::Point2f>&     pts2d,
    std::vector<Eigen::Vector3d>& obs_stereo,
    std::vector<std::pair<int, MapPoint::Ptr>>& mp_feat_pairs,
    std::vector<int>&             octaves)
{
    if (!last_kf_ || last_kf_->is_bad()) return false;

    // Candidate set: last KF + up to N covisible neighbours.
    // Viewing-angle, predicted-scale, and orientation-consistency guards
    // are applied below to suppress aliases.
    static const int kMaxLocalKFs = []() {
        if (const char* e = std::getenv("SSLAM_LOCAL_KFS")) {
            const int v = std::atoi(e);
            if (v >= 0) return v;
        }
        return 10;
    }();
    std::vector<KeyFrame*> local_kfs{last_kf_.get()};
    {
        auto covis = last_kf_->get_covisibility_keyframes(0);
        const int n = std::min(static_cast<int>(covis.size()), kMaxLocalKFs);
        for (int i = 0; i < n; ++i) {
            if (covis[i] && !covis[i]->is_bad())
                local_kfs.push_back(covis[i]);
        }
    }

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

    std::vector<LocalMapMatchCandidate> candidates;

    constexpr int   kTH_HIGH       = 100;
    constexpr float kNarrowRadius2 = 1225.0f;  // 35 px — normal tracking
    constexpr float kWideRadius2   = 4900.0f;  // 70 px — after LOST or first frames
    constexpr float kLoweRatio     = 0.70f;

    // After a LOST recovery the velocity model is reset to identity, so the
    // prediction error can be large.  Use a wider radius in that case.
    const float kRadius2 = (state_ == TrackingState::LOST)
                           ? kWideRadius2 : kNarrowRadius2;

    const Eigen::Vector3d cam_center = -R_pred.transpose() * t_pred;

    for (std::size_t mp_idx = 0; mp_idx < local_mps.size(); ++mp_idx) {
        const auto& mp = local_mps[mp_idx];

        // Project MP into the current frame using the predicted pose.
        const Eigen::Vector3d p_w = mp->get_world_pos();
        const Eigen::Vector3d X_c = R_pred * p_w + t_pred;
        if (X_c.z() <= 0.0) continue;

        const double dist_to_camera = (p_w - cam_center).norm();
        const float min_dist = mp->min_distance();
        const float max_dist = mp->max_distance();
        if (min_dist > 0.0f && dist_to_camera < 0.8 * static_cast<double>(min_dist))
            continue;
        if (max_dist > 0.0f && dist_to_camera > 1.2 * static_cast<double>(max_dist))
            continue;

        const Eigen::Vector3d normal = mp->mean_normal();
        if (normal.squaredNorm() > 1e-12) {
            const Eigen::Vector3d view_dir = (p_w - cam_center).normalized();
            if (normal.dot(view_dir) < 0.5) continue;
        }

        const float u_proj = static_cast<float>(fx * X_c.x() / X_c.z() + cx);
        const float v_proj = static_cast<float>(fy * X_c.y() / X_c.z() + cy);
        if (u_proj < 0.0f || u_proj >= static_cast<float>(cam_->width) ||
            v_proj < 0.0f || v_proj >= static_cast<float>(cam_->height))
            continue;

        const cv::Mat mp_desc = mp->get_descriptor();
        if (mp_desc.empty()) continue;

        // Count projection as a visibility event.
        mp->inc_visible();

        // 2-best search for Lowe ratio filtering.
        int best_dist = kTH_HIGH + 1, second_dist = kTH_HIGH + 1, best_j = -1;
        for (int j = 0; j < n_kp; ++j) {
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

        candidates.push_back({mp_idx, best_j, best_dist, X_c.z()});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const LocalMapMatchCandidate& a,
                 const LocalMapMatchCandidate& b) {
                  return a.distance < b.distance;
              });

    std::vector<bool> matched_feat(static_cast<std::size_t>(n_kp), false);
    std::vector<bool> matched_mp(local_mps.size(), false);

    for (const LocalMapMatchCandidate& c : candidates) {
        if (matched_mp[c.mp_idx] ||
            matched_feat[static_cast<std::size_t>(c.feat_idx)]) {
            continue;
        }

        const auto& mp = local_mps[c.mp_idx];
        matched_mp[c.mp_idx] = true;
        matched_feat[static_cast<std::size_t>(c.feat_idx)] = true;

        // Record correspondence.
        mp->inc_found();  // actually matched
        const Eigen::Vector3d p_w = mp->get_world_pos();
        pts3d.push_back({static_cast<float>(p_w.x()),
                         static_cast<float>(p_w.y()),
                         static_cast<float>(p_w.z())});
        pts2d.push_back(kps[static_cast<std::size_t>(c.feat_idx)].pt);
        mp_feat_pairs.push_back({c.feat_idx, mp});

        // Right-camera observation: prefer the actual stereo match stored on
        // the frame over the depth derived from the predicted projection.
        // X_c.z() is the predicted depth (consistent with T_pred, not the
        // final optimized pose), so using it for u_r would bias BA.
        const double u_l = static_cast<double>(kps[static_cast<std::size_t>(c.feat_idx)].pt.x);
        const double v_l = static_cast<double>(kps[static_cast<std::size_t>(c.feat_idx)].pt.y);
        const float  d_actual = frame.depth[static_cast<std::size_t>(c.feat_idx)];
        const double u_r = (d_actual > 0.0f)
            ? u_l - bf / static_cast<double>(d_actual)   // real stereo observation
            : u_l - bf / c.pred_depth;                    // fallback: projected depth
        obs_stereo.push_back({u_l, v_l, u_r});
        octaves.push_back(kps[static_cast<std::size_t>(c.feat_idx)].octave);
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
    const bool min_spacing = (nframes_since_kf_ >= params_.kf_min_frames_since);
    const bool max_spacing = (nframes_since_kf_ >= params_.kf_max_frames_since);

    // Criterion C: minimum tracking quality.
    // Guard n_inliers > 0 so the forced-zero initialisation call doesn't fire it.
    const bool min_crit = (n_inliers > 0 &&
                           n_inliers < params_.kf_min_tracked_points);

    // Criterion D: tracked ratio vs reference KF.
    // ORB-SLAM2 NeedNewKeyFrame compares against reference MPs with enough
    // observations, not every depth point ever created by the reference KF.
    // Counting all MPs makes nRefMatches huge and causes a keyframe explosion.
    bool severe_ratio_crit = false;
    int n_ref_mps = 0;
    if (last_kf_ && !last_kf_->is_bad() && n_inliers > 0) {
        n_ref_mps = last_kf_->tracked_map_points(3);
        if (n_ref_mps > 0 &&
            n_inliers < static_cast<int>(0.25f * static_cast<float>(n_ref_mps)))
            severe_ratio_crit = true;
    }

    const bool max_time_insert = max_spacing;
    const bool early_quality_insert = min_spacing && (min_crit || severe_ratio_crit);

    if (!must_insert && !max_time_insert && !early_quality_insert)
        return;

    // curr_frame.T_cw is already in the single map/world frame.  This mirrors
    // ORB-SLAM: tracking, local mapping, and loop closing all operate on the
    // same pose convention rather than maintaining a separate raw VO frame.
    const Eigen::Matrix4d T_cw_corrected = curr_frame.T_cw;

    // --- Build new KeyFrame -----------------------------------------------
    auto kf = std::make_shared<KeyFrame>(next_kf_id_++, curr_frame, cam_);
    kf->set_pose(T_cw_corrected);
    kf->set_scale_factors(orb_extractor_.scale_factors());
    if (last_kf_ && !last_kf_->is_bad())
        kf->set_parent(last_kf_.get());

    // --- Propagate MPs from last_kf_ via projection + descriptor matching --
    // For each MP in last_kf_, project into curr_frame using curr_frame.T_cw,
    // then scan curr_frame features within a 20 px radius for the best Hamming
    // match. This propagates MPs across the inter-KF gap without requiring a
    // continuous frame chain, at O(N_MPs × N_feats_in_radius) cost per KF
    // insertion (acceptable since KFs are rare — every kf_max_frames_since frames).
    std::vector<bool> curr_matched(curr_frame.num_features(), false);

    // Seed curr_matched from Stage 2 map_points if available — this avoids
    // duplicate associations and skips the projection search for features that
    // TrackLocalMap has already matched to known MPs.
    if (!curr_frame.map_points.empty()) {
        const std::size_t n_mp =
            std::min(curr_frame.map_points.size(), curr_frame.num_features());
        for (std::size_t i = 0; i < n_mp; ++i) {
            const auto& mp = curr_frame.map_points[i];
            if (!mp || mp->is_bad()) continue;
            kf->add_map_point(static_cast<int>(i), mp);
            mp->add_observation(kf.get(), static_cast<int>(i));
            curr_matched[i] = true;
        }
    }

    if (last_kf_) {
        const Eigen::Matrix3d R_cw = T_cw_corrected.topLeftCorner<3, 3>();
        const Eigen::Vector3d t_cw = T_cw_corrected.topRightCorner<3, 1>();

        const auto& kps_curr   = curr_frame.keypoints_left;
        const auto& desc_curr  = curr_frame.descriptors_left;
        const int   n_curr     = static_cast<int>(kps_curr.size());
        constexpr float kBaseRadius = 15.0f;      // base search radius in pixels
        constexpr int kHammingTh = 100;           // TH_HIGH for ORB
        const auto& sf_ref = last_kf_->scale_factors();  // octave scale factors

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
            // Use a scale-aware radius: base_r * scale_factor[octave].
            const int ref_feat_i = i;  // last_kf_ feature index
            const auto& ref_kps  = last_kf_->keypoints_left();
            const int ref_oct = (static_cast<std::size_t>(ref_feat_i) < ref_kps.size())
                                ? ref_kps[static_cast<std::size_t>(ref_feat_i)].octave : 0;
            const float scale_fac = (!sf_ref.empty() && ref_oct < static_cast<int>(sf_ref.size()))
                                    ? sf_ref[static_cast<std::size_t>(ref_oct)] : 1.0f;
            const float search_r2 = (kBaseRadius * scale_fac) * (kBaseRadius * scale_fac);
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

// ---------------------------------------------------------------------------
// Relocalization: BoW query → PnP RANSAC against candidate KFs.
// ---------------------------------------------------------------------------

bool Tracking::relocalize_(Frame& frame) {
    if (!vocab_ || !kf_db_reloc_) return false;

    // Compute BoW for the current frame descriptors.
    DBoW2::BowVector   bow;
    DBoW2::FeatureVector fvec;
    {
        std::vector<cv::Mat> desc_vec;
        desc_vec.reserve(static_cast<std::size_t>(frame.descriptors_left.rows));
        for (int r = 0; r < frame.descriptors_left.rows; ++r)
            desc_vec.push_back(frame.descriptors_left.row(r));
        vocab_->transform(desc_vec, bow, fvec, 4);
    }

    const auto candidates = kf_db_reloc_->query_relocalization_candidates(bow, 0.01);
    if (candidates.empty()) return false;

    const double fx = cam_->fx, fy = cam_->fy, cx = cam_->cx, cy = cam_->cy;
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    constexpr int kHammingTh = 100;
    constexpr int kMinInliers = 10;

    for (const KeyFrame* cand_kf : candidates) {
        if (!cand_kf || cand_kf->is_bad()) continue;

        // Collect all non-bad MPs from this candidate KF.
        std::vector<cv::Point3f>    pts3d;
        std::vector<cv::Point2f>    pts2d;
        std::vector<Eigen::Vector3d> obs_stereo;
        std::vector<int>             octaves_r;

        const auto& cand_kps   = cand_kf->keypoints_left();
        const auto& cand_descs = cand_kf->descriptors_left();
        const auto& frame_kps  = frame.keypoints_left;
        const int n_frame      = static_cast<int>(frame_kps.size());

        const double bf = cam_->fx * cam_->baseline;

        for (int ci = 0; ci < static_cast<int>(cand_kps.size()); ++ci) {
            auto mp = cand_kf->get_map_point(ci);
            if (!mp || mp->is_bad()) continue;

            // Match candidate KF descriptor against current frame features.
            const cv::Mat d_c = cand_descs.row(ci);
            int best_dist = kHammingTh + 1, second_dist = kHammingTh + 1, best_j = -1;
            for (int j = 0; j < n_frame; ++j) {
                const int dist = cv::norm(d_c, frame.descriptors_left.row(j), cv::NORM_HAMMING);
                if (dist < best_dist) { second_dist = best_dist; best_dist = dist; best_j = j; }
                else if (dist < second_dist) { second_dist = dist; }
            }
            if (best_j < 0 || best_dist > kHammingTh) continue;
            if (second_dist <= kHammingTh && best_dist > static_cast<int>(0.75f * second_dist)) continue;

            const Eigen::Vector3d pw = mp->get_world_pos();
            pts3d.push_back({static_cast<float>(pw.x()), static_cast<float>(pw.y()), static_cast<float>(pw.z())});
            const cv::KeyPoint& fkp = frame_kps[static_cast<std::size_t>(best_j)];
            pts2d.push_back(fkp.pt);
            const float d = frame.depth[static_cast<std::size_t>(best_j)];
            const double u_l = fkp.pt.x, v_l = fkp.pt.y;
            const double u_r = (d > 0.0f) ? u_l - bf / d : u_l - bf / 1.0;
            obs_stereo.push_back({u_l, v_l, u_r});
            octaves_r.push_back(fkp.octave);
        }

        if (static_cast<int>(pts3d.size()) < kMinInliers) continue;

        cv::Mat rvec, tvec, inliers_mat;
        const bool ok = cv::solvePnPRansac(
            pts3d, pts2d, K, cv::noArray(),
            rvec, tvec, false,
            params_.pnp_max_iterations,
            params_.pnp_reprojection_error,
            static_cast<double>(params_.pnp_confidence),
            inliers_mat,
            cv::SOLVEPNP_EPNP);

        if (!ok || inliers_mat.rows < kMinInliers) continue;

        // Iterative refinement on inliers.
        std::vector<cv::Point3f>     pts3d_in;
        std::vector<cv::Point2f>     pts2d_in;
        std::vector<Eigen::Vector3d> obs_in;
        std::vector<int>             oct_in;
        pts3d_in.reserve(inliers_mat.rows);
        pts2d_in.reserve(inliers_mat.rows);
        obs_in.reserve(inliers_mat.rows);
        oct_in.reserve(inliers_mat.rows);
        for (int k = 0; k < inliers_mat.rows; ++k) {
            const int idx = inliers_mat.at<int>(k);
            pts3d_in.push_back(pts3d[static_cast<std::size_t>(idx)]);
            pts2d_in.push_back(pts2d[static_cast<std::size_t>(idx)]);
            obs_in.push_back(obs_stereo[static_cast<std::size_t>(idx)]);
            if (static_cast<std::size_t>(idx) < octaves_r.size())
                oct_in.push_back(octaves_r[static_cast<std::size_t>(idx)]);
        }
        cv::solvePnP(pts3d_in, pts2d_in, K, cv::noArray(), rvec, tvec,
                     true, cv::SOLVEPNP_ITERATIVE);

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R_eig;
        cv::cv2eigen(R_cv, R_eig);
        Eigen::Matrix4d T_cw_reloc = Eigen::Matrix4d::Identity();
        T_cw_reloc.topLeftCorner<3,3>() = R_eig;
        T_cw_reloc(0,3) = tvec.at<double>(0);
        T_cw_reloc(1,3) = tvec.at<double>(1);
        T_cw_reloc(2,3) = tvec.at<double>(2);

        // Motion-only BA to refine.
        std::vector<Eigen::Vector3d> pts3d_eig;
        pts3d_eig.reserve(pts3d_in.size());
        for (const auto& p : pts3d_in) pts3d_eig.push_back({p.x, p.y, p.z});
        const auto ba_res = ba::optimize_pose(T_cw_reloc, pts3d_eig, obs_in, *cam_, oct_in);
        if (ba_res.n_inliers < kMinInliers) continue;

        frame.T_cw = ba_res.T_cw;

        // Re-anchor onto the candidate KF.  The relocalized pose is already
        // in the map/world frame, so subsequent tracking can continue in the
        // same frame just like ORB-SLAM.
        for (const auto& kf_ptr : map_->get_all_keyframes()) {
            if (kf_ptr.get() == cand_kf) {
                last_kf_           = kf_ptr;
                last_kf_frame_idx_ = frame.index;
                nframes_since_kf_  = 0;
                break;
            }
        }
        return true;
    }
    return false;
}

}  // namespace sslam
