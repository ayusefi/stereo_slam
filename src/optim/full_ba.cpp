#include "sslam/optim/full_ba.hpp"

#include "sslam/optim/ba.hpp"
#include "sslam/optim/g2o_types.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/mappoint.hpp"

#include <g2o/types/sba/types_sba.h>

#include <Eigen/Geometry>

#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sslam {

namespace {

Eigen::Matrix4d inverse_se3(const Eigen::Matrix4d& T_cw) {
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R_wc = T_cw.topLeftCorner<3, 3>().transpose();
    T_wc.topLeftCorner<3, 3>()  = R_wc;
    T_wc.topRightCorner<3, 1>() = -R_wc * T_cw.topRightCorner<3, 1>();
    return T_wc;
}

}  // namespace

FullBA::FullBA(Map::Ptr map) : map_(std::move(map)) {}

// ---------------------------------------------------------------------------

void FullBA::trigger() {
    std::scoped_lock lk(mutex_);
    // Cancel any in-flight run.
    cancel_.store(true);
    if (future_.valid()) future_.wait();
    cancel_.store(false);
    future_ = std::async(std::launch::async, &FullBA::run, this);
}

void FullBA::wait() {
    std::scoped_lock lk(mutex_);
    if (future_.valid()) future_.wait();
}

// ---------------------------------------------------------------------------

void FullBA::run() {
    // Full BA over all non-bad KFs and MPs.
    // Huber delta for a 3-DoF stereo residual: sqrt(chi2_3dof at 95th pct).
    constexpr double kHuberDelta = 2.7955;  // sqrt(7.815)
    constexpr double kChi2Th    = 7.815;
    // The final pass runs on the reclassified inlier set.  A deterministic
    // sweep found 20 iterations to be the best loop-sequence tradeoff; keep an
    // env hook for diagnostics without changing normal defaults.
    static const int kFinalIterations = []() {
        if (const char* env = std::getenv("SSLAM_FULL_BA_FINAL_ITERS")) {
            const int parsed = std::atoi(env);
            if (parsed > 0) return parsed;
        }
        return 20;
    }();
    // Uses the same g2o setup as local BA but with all KFs as vertices,
    // KF id=0 fixed as the gauge anchor.
    //
    // Implemented as an outer-loop with cancellation checks between passes.

    using BlockSolverType  = g2o::BlockSolver_6_3;
    using LinearSolverType =
        g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear = std::make_unique<LinearSolverType>();
    auto block  = std::make_unique<BlockSolverType>(std::move(linear));
    auto algo   = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    g2o::SparseOptimizer opt;
    opt.setAlgorithm(algo);
    opt.setVerbose(false);

    const std::vector<KeyFrame::Ptr> all_kfs = map_->get_all_keyframes();
    const std::vector<MapPoint::Ptr> all_mps = map_->get_all_mappoints();

    // Snapshot the highest KF id before BA starts so we can identify KFs/MPs
    // added by Tracking/LocalMapping while the optimiser is running.
    uint64_t max_kf_id_at_start = 0;
    for (const KeyFrame::Ptr& kf : all_kfs)
        max_kf_id_at_start = std::max(max_kf_id_at_start, kf->id());

    // Snapshot pre-BA poses of all KFs included in the optimisation (needed
    // later to compute the correction for post-BA KFs).
    std::unordered_map<uint64_t, Eigen::Matrix4d> pose_before;
    pose_before.reserve(all_kfs.size());
    for (const KeyFrame::Ptr& kf : all_kfs)
        if (!kf->is_bad()) pose_before[kf->id()] = kf->get_pose();

    if (cancel_.load()) return;

    // --- Vertices: KF poses ---------------------------------------------------
    std::unordered_map<uint64_t, int> kf_vid;  // kf_id → g2o vertex id
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad()) continue;
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(static_cast<int>(kf->id()));
        const Eigen::Matrix3d R = kf->get_pose().topLeftCorner<3,3>();
        const Eigen::Vector3d t = kf->get_pose().topRightCorner<3,1>();
        v->setEstimate(g2o::SE3Quat(R, t));
        v->setFixed(kf->id() == 0);
        opt.addVertex(v);
        kf_vid[kf->id()] = static_cast<int>(kf->id());
    }

    if (cancel_.load()) return;

    // --- Vertices: MP positions -----------------------------------------------
    constexpr int kMpIdOffset = 1 << 20;  // shift MP vertex ids above KF ids
    std::unordered_map<uint64_t, g2o::VertexPointXYZ*> mp_v;
    for (const MapPoint::Ptr& mp : all_mps) {
        if (mp->is_bad()) continue;
        auto* v = new g2o::VertexPointXYZ();
        const int vid = kMpIdOffset + static_cast<int>(mp->id());
        v->setId(vid);
        v->setEstimate(mp->get_world_pos());
        v->setMarginalized(true);
        opt.addVertex(v);
        mp_v[mp->id()] = v;
    }

    if (cancel_.load()) return;

    // --- Edges ----------------------------------------------------------------
    const StereoCamera* cam = nullptr;
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (!kf->is_bad()) { cam = kf->camera().get(); break; }
    }
    if (!cam) return;

    int edge_id = kMpIdOffset * 2;
    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> edges;

    for (const MapPoint::Ptr& mp : all_mps) {
        if (mp->is_bad() || !mp_v.count(mp->id())) continue;
        for (const auto& [kf_raw, feat_idx] : mp->get_observations()) {
            if (kf_raw->is_bad() || !kf_vid.count(kf_raw->id())) continue;

            const cv::KeyPoint& kp = kf_raw->keypoints_left()[feat_idx];
            const float u_r = kf_raw->right_u()[feat_idx];
            if (u_r < 0.0f) continue;  // no stereo

            auto* e = new g2o::EdgeStereoSE3ProjectXYZ();
            e->setId(++edge_id);
            e->setVertex(0, mp_v.at(mp->id()));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                opt.vertex(kf_vid.at(kf_raw->id()))));
            e->setMeasurement(Eigen::Vector3d(kp.pt.x, kp.pt.y, u_r));
            // Information matrix: I / sigma2[octave], sigma2 = scale_factor^(2*octave)
            // with scale_factor = 1.2.  Coarse-pyramid observations are noisier and
            // must be down-weighted, exactly as in local_bundle_adjustment().
            constexpr double kScaleSq = 1.2 * 1.2;
            const double sigma2 = std::pow(kScaleSq, kp.octave);
            e->setInformation(Eigen::Matrix3d::Identity() * (1.0 / sigma2));
            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(kHuberDelta);
            e->setRobustKernel(rk);
            e->fx = cam->fx; e->fy = cam->fy;
            e->cx = cam->cx; e->cy = cam->cy;
            e->bf = cam->fx * cam->baseline;
            opt.addEdge(e);
            edges.push_back(e);
        }
        if (cancel_.load()) return;
    }

    // --- Two-pass optimisation with cancellation checks ----------------------
    for (int outer = 0; outer < 2; ++outer) {
        if (cancel_.load()) return;
        opt.initializeOptimization(0);
        opt.optimize(outer == 0 ? 10 : kFinalIterations);
        if (cancel_.load()) return;
        for (auto* e : edges) {
            if (e->level() == 0 && e->chi2() > kChi2Th)
                e->setLevel(1);
            else if (e->level() == 1 && e->chi2() <= kChi2Th)
                e->setLevel(0);
        }
    }

    if (cancel_.load()) return;

    // --- Write back -----------------------------------------------------------
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad() || !kf_vid.count(kf->id())) continue;
        auto* v = dynamic_cast<g2o::VertexSE3Expmap*>(opt.vertex(kf_vid.at(kf->id())));
        if (!v) continue;
        const g2o::SE3Quat& est = v->estimate();
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner<3,3>()  = est.rotation().toRotationMatrix();
        T.topRightCorner<3,1>() = est.translation();
        kf->set_pose(T);
    }
    for (const MapPoint::Ptr& mp : all_mps) {
        if (mp->is_bad() || !mp_v.count(mp->id())) continue;
        if (!cancel_.load())
            mp->set_world_pos(mp_v.at(mp->id())->estimate());
    }

    // --- BFS correction propagation for KFs/MPs added during BA --------------
    // Fetch the current live map state (may include new entries since BA start).
    const std::vector<KeyFrame::Ptr> live_kfs = map_->get_all_keyframes();
    const std::vector<MapPoint::Ptr> live_mps = map_->get_all_mappoints();

    // Build lookup: id → corrected T_cw for every KF that was in the BA.
    std::unordered_map<uint64_t, Eigen::Matrix4d> pose_after;
    pose_after.reserve(all_kfs.size());
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad() || !kf_vid.count(kf->id())) continue;
        pose_after[kf->id()] = kf->get_pose();  // already written back above
    }

    // Collect post-BA KFs (added while optimiser was running).
    std::vector<KeyFrame*> post_kfs;
    std::unordered_map<uint64_t, Eigen::Matrix4d> old_pose_by_id = pose_before;
    for (const KeyFrame::Ptr& kf : live_kfs) {
        if (!kf->is_bad() && kf->id() > max_kf_id_at_start) {
            post_kfs.push_back(kf.get());
            old_pose_by_id[kf->id()] = kf->get_pose();
        }
    }

    if (!post_kfs.empty()) {
        // Build children map over the spanning tree restricted to post_kfs
        // plus the last shared KF as the BFS root.
        std::unordered_map<uint64_t, std::vector<KeyFrame*>> children;
        for (KeyFrame* kf : post_kfs) {
            KeyFrame* par = kf->parent();
            if (par) children[par->id()].push_back(kf);
        }

        std::unordered_map<uint64_t, Eigen::Matrix4d> new_pose_by_id = pose_after;
        if (!new_pose_by_id.count(max_kf_id_at_start) ||
            !old_pose_by_id.count(max_kf_id_at_start)) {
            return;
        }

        std::queue<uint64_t> bfs;
        bfs.push(max_kf_id_at_start);
        while (!bfs.empty()) {
            const uint64_t pid = bfs.front(); bfs.pop();
            if (!children.count(pid)) continue;
            for (KeyFrame* child : children.at(pid)) {
                const Eigen::Matrix4d T_old_child = child->get_pose();
                const Eigen::Matrix4d T_new_child =
                    T_old_child * inverse_se3(old_pose_by_id.at(pid))
                    * new_pose_by_id.at(pid);
                child->set_pose(T_new_child);
                new_pose_by_id[child->id()] = T_new_child;
                bfs.push(child->id());
            }
        }

        // Move MPs whose reference KF is one of the post-BA KFs.
        for (const MapPoint::Ptr& mp : live_mps) {
            if (mp->is_bad()) continue;
            const uint64_t ref_id = mp->created_kf_id();
            if (ref_id <= max_kf_id_at_start) continue;  // already corrected above
            if (!old_pose_by_id.count(ref_id) || !new_pose_by_id.count(ref_id))
                continue;
            const Eigen::Matrix4d T_wk_new = inverse_se3(new_pose_by_id.at(ref_id));
            const Eigen::Vector3d X_local =
                old_pose_by_id.at(ref_id).topLeftCorner<3,3>() * mp->get_world_pos()
                + old_pose_by_id.at(ref_id).topRightCorner<3,1>();
            const Eigen::Vector3d X_new =
                T_wk_new.topLeftCorner<3,3>() * X_local
                + T_wk_new.topRightCorner<3,1>();
            mp->set_world_pos(X_new);
        }
    }
}

}  // namespace sslam
