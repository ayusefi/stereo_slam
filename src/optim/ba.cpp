// Motion-only and local bundle adjustment.
//
// optimize_pose:
//   Optimises a single camera pose (SE3Expmap vertex) against a fixed set of
//   3-D world points observed as stereo correspondences (u_l, v, u_r).
//
// local_bundle_adjustment:
//   Jointly optimises a window of KeyFrame poses (SE3Expmap vertices) and
//   their shared MapPoint positions (PointXYZ vertices).  Fixed-KF poses
//   anchor the gauge without participating in the update.
//
// Both optimisers run `outer_iterations` Levenberg-Marquardt rounds.  After
// each round every edge is reclassified as inlier/outlier by comparing its
// chi-squared error to `chi2_threshold`.  Outlier edges have their level set
// to 1 (excluded from subsequent LM steps) while inlier edges remain at
// level 0.
//
// The Huber robust kernel is attached only to inlier-level edges; outlier
// edges are detached before the next iteration to avoid pulling the estimate
// toward wrong measurements.
//
// Reference: ORB-SLAM2 Optimizer.cc::PoseOptimization.

#include "sslam/optim/ba.hpp"
#include "sslam/optim/g2o_types.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/mappoint.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sslam {
namespace ba {

namespace {

/// Convert a 4×4 SE(3) matrix (our T_cw convention) to a g2o SE3Quat.
g2o::SE3Quat to_se3quat(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T.topRightCorner<3, 1>();
    return g2o::SE3Quat(R, t);
}

/// Convert a g2o SE3Quat back to a 4×4 SE(3) matrix.
Eigen::Matrix4d from_se3quat(const g2o::SE3Quat& q) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>()  = q.rotation().toRotationMatrix();
    T.topRightCorner<3, 1>() = q.translation();
    return T;
}

}  // namespace

PoseOptResult optimize_pose(
    const Eigen::Matrix4d&              T_cw_init,
    const std::vector<Eigen::Vector3d>& pts3d,
    const std::vector<Eigen::Vector3d>& obs_stereo,
    const StereoCamera&                 cam,
    const std::vector<int>&             octaves,
    const Params&                       p) {

    assert(pts3d.size() == obs_stereo.size());
    const int n = static_cast<int>(pts3d.size());

    // --- Build optimizer ---------------------------------------------------
    using BlockSolver  = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverDense<BlockSolver::PoseMatrixType>;

    auto optimizer = std::make_unique<g2o::SparseOptimizer>();
    optimizer->setVerbose(false);

    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolver>(std::make_unique<LinearSolver>()));
    optimizer->setAlgorithm(algorithm);

    // --- Pose vertex (the only free variable) ------------------------------
    auto* v = new g2o::VertexSE3Expmap();
    v->setId(0);
    v->setFixed(false);
    v->setEstimate(to_se3quat(T_cw_init));
    optimizer->addVertex(v);

    // --- Build edges -------------------------------------------------------
    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> edges;
    edges.reserve(static_cast<std::size_t>(n));

    const double bf = cam.fx * cam.baseline;
    constexpr double kScaleSq = 1.2 * 1.2;

    for (int i = 0; i < n; ++i) {
        auto* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

        e->setVertex(0, v);
        e->setMeasurement(obs_stereo[static_cast<std::size_t>(i)]);

        // Information matrix: I / sigma2[octave].
        // sigma2[octave] = (scale_factor)^(2*octave), scale_factor = 1.2.
        const int oct = (!octaves.empty() && i < static_cast<int>(octaves.size()))
                        ? octaves[static_cast<std::size_t>(i)] : 0;
        const double sigma2 = std::pow(kScaleSq, oct);
        e->setInformation(Eigen::Matrix3d::Identity() * (1.0 / sigma2));

        // g2o takes ownership of the kernel pointer; allocate one per edge.
        auto* rk = new g2o::RobustKernelHuber();
        rk->setDelta(p.huber_delta);
        e->setRobustKernel(rk);

        e->Xw = pts3d[static_cast<std::size_t>(i)];
        e->fx  = cam.fx;
        e->fy  = cam.fy;
        e->cx  = cam.cx;
        e->cy  = cam.cy;
        e->bf  = bf;

        optimizer->addEdge(e);
        edges.push_back(e);
    }

    // --- 4 outer × 10 inner iterations with inlier reclassification --------
    // Set the initial estimate once, before any outer iteration.
    v->setEstimate(to_se3quat(T_cw_init));

    int n_inliers = n;

    for (int outer = 0; outer < p.outer_iterations; ++outer) {
        optimizer->initializeOptimization(0);  // optimise level-0 edges
        optimizer->optimize(p.inner_iterations);

        // Reclassify edges; outliers → level 1 (excluded next iter)
        n_inliers = 0;
        for (auto* e : edges) {
            e->computeError();
            const double chi2 = e->chi2();
            if (chi2 < p.chi2_threshold) {
                e->setLevel(0);   // inlier
                ++n_inliers;
            } else {
                e->setLevel(1);   // outlier — excluded from next LM step
            }
            // Drop the Huber kernel on the penultimate outer so the final
            // pass is a clean Gauss-Newton step on the inlier set.
            // setRobustKernel(nullptr) frees the heap-allocated kernel.
            if (outer == p.outer_iterations - 2) {
                e->setRobustKernel(nullptr);
            }
        }

        if (n_inliers < 4) break;
    }

    // --- Extract result ----------------------------------------------------
    PoseOptResult result;
    result.T_cw      = from_se3quat(v->estimate());
    result.n_inliers = n_inliers;
    result.inlier_mask.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        result.inlier_mask[static_cast<std::size_t>(i)] = (edges[static_cast<std::size_t>(i)]->level() == 0);
    return result;
}

// ---------------------------------------------------------------------------
// Local Bundle Adjustment
// ---------------------------------------------------------------------------

void local_bundle_adjustment(KeyFrame* kf,
                             const StereoCamera& cam,
                             const Params& p) {
    // -----------------------------------------------------------------------
    // 1. Collect local KFs (current KF + covisibility neighbours).
    // -----------------------------------------------------------------------
    std::unordered_set<KeyFrame*> local_kf_set;
    local_kf_set.insert(kf);
    {
        auto covis = kf->get_covisibility_keyframes(0);  // sorted by weight desc
        if (static_cast<int>(covis.size()) > p.max_local_kfs)
            covis.resize(static_cast<std::size_t>(p.max_local_kfs));
        for (KeyFrame* nb : covis)
            if (nb && !nb->is_bad())
                local_kf_set.insert(nb);
    }

    // -----------------------------------------------------------------------
    // 2. Collect all MPs observed by local KFs (capped to keep g2o tractable).
    // -----------------------------------------------------------------------
    std::unordered_set<MapPoint*> local_mp_set;
    for (KeyFrame* lkf : local_kf_set)
        for (const auto& mp : lkf->get_map_points())
            if (mp && !mp->is_bad() && mp->n_observations() >= 2)
                local_mp_set.insert(mp.get());

    // Cap the MP set deterministically so the g2o problem stays bounded.
    if (static_cast<int>(local_mp_set.size()) > p.max_local_mps) {
        std::vector<MapPoint*> sorted_mps(local_mp_set.begin(), local_mp_set.end());
        std::sort(sorted_mps.begin(), sorted_mps.end(),
                  [](const MapPoint* a, const MapPoint* b) {
                      return a->id() < b->id();
                  });
        std::unordered_set<MapPoint*> trimmed;
        trimmed.reserve(static_cast<std::size_t>(p.max_local_mps));
        for (MapPoint* mp : sorted_mps) {
            trimmed.insert(mp);
            if (static_cast<int>(trimmed.size()) >= p.max_local_mps) break;
        }
        local_mp_set = std::move(trimmed);
    }

    // -----------------------------------------------------------------------
    // 3. Fixed KFs: observe local MPs but are not local themselves.
    //    If none exist (map is small), fix the oldest local KF to remove
    //    gauge freedom.
    // -----------------------------------------------------------------------
    std::unordered_set<KeyFrame*> fixed_kf_set;
    for (MapPoint* mp : local_mp_set) {
        for (const auto& [obs_kf, feat_idx] : mp->get_observations()) {
            if (obs_kf && !obs_kf->is_bad() && local_kf_set.count(obs_kf) == 0)
                fixed_kf_set.insert(obs_kf);
        }
    }

    // Anchor: if still no fixed KF, fix the local KF with the smallest id.
    KeyFrame* anchor_kf = nullptr;
    if (fixed_kf_set.empty() && local_kf_set.size() > 1) {
        for (KeyFrame* lkf : local_kf_set) {
            if (!anchor_kf || lkf->id() < anchor_kf->id())
                anchor_kf = lkf;
        }
    }

    // -----------------------------------------------------------------------
    // 4. Build g2o graph.
    //    Schur-complement (sparse Eigen) solver for KF × MP block structure.
    // -----------------------------------------------------------------------
    using BlockSolver  = g2o::BlockSolver_6_3;
    using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

    auto optimizer = std::make_unique<g2o::SparseOptimizer>();
    optimizer->setVerbose(false);
    auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolver>(std::make_unique<LinearSolver>()));
    optimizer->setAlgorithm(algorithm);

    // Vertex IDs:
    //   0 … N_local-1  : local KF poses
    //   N_local … N_local+N_fixed-1 : fixed KF poses
    //   N_local+N_fixed … : MP positions

    int next_id = 0;
    std::unordered_map<KeyFrame*, int>  kf_to_id;
    std::unordered_map<MapPoint*, int>  mp_to_id;

    // Local KF vertices (free, except the anchor if set)
    for (KeyFrame* lkf : local_kf_set) {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(next_id);
        v->setFixed(lkf == anchor_kf);
        v->setEstimate(to_se3quat(lkf->get_pose()));
        optimizer->addVertex(v);
        kf_to_id[lkf] = next_id++;
    }

    // Fixed KF vertices
    for (KeyFrame* fkf : fixed_kf_set) {
        auto* v = new g2o::VertexSE3Expmap();
        v->setId(next_id);
        v->setFixed(true);
        v->setEstimate(to_se3quat(fkf->get_pose()));
        optimizer->addVertex(v);
        kf_to_id[fkf] = next_id++;
    }

    // MP vertices
    for (MapPoint* mp : local_mp_set) {
        auto* v = new g2o::VertexPointXYZ();
        v->setId(next_id);
        v->setFixed(false);
        v->setMarginalized(true);  // enables Schur complement
        const Eigen::Vector3d pos = mp->get_world_pos();
        v->setEstimate(pos);
        optimizer->addVertex(v);
        mp_to_id[mp] = next_id++;
    }

    // Edges
    struct EdgeInfo {
        g2o::EdgeStereoSE3ProjectXYZ* edge;
        KeyFrame*  kf;
        MapPoint*  mp;
        int        feat_idx;
    };
    std::vector<EdgeInfo> edge_infos;

    const double bf = cam.fx * cam.baseline;

    for (MapPoint* mp : local_mp_set) {
        const int mp_id = mp_to_id.at(mp);
        for (const auto& [obs_kf, feat_idx] : mp->get_observations()) {
            if (!obs_kf || obs_kf->is_bad()) continue;
            if (kf_to_id.count(obs_kf) == 0) continue;

            const auto& kps  = obs_kf->keypoints_left();
            const auto& ru   = obs_kf->right_u();
            if (feat_idx < 0 ||
                static_cast<std::size_t>(feat_idx) >= kps.size()) continue;
            const float right_u = ru[static_cast<std::size_t>(feat_idx)];
            if (right_u < 0.0f) continue;  // no stereo observation

            auto* e = new g2o::EdgeStereoSE3ProjectXYZ();
            e->setVertex(0, optimizer->vertex(mp_id));
            e->setVertex(1, optimizer->vertex(kf_to_id.at(obs_kf)));

            const cv::KeyPoint& kp = kps[static_cast<std::size_t>(feat_idx)];
            e->setMeasurement(
                Eigen::Vector3d(kp.pt.x, kp.pt.y, right_u));

            // Per-octave information matrix: I / sigma2[octave].
            {
                const auto& sf = obs_kf->scale_factors();
                const int oct  = kp.octave;
                double sigma2  = 1.0;
                if (!sf.empty() && oct < static_cast<int>(sf.size())) {
                    sigma2 = static_cast<double>(sf[static_cast<std::size_t>(oct)]) *
                             static_cast<double>(sf[static_cast<std::size_t>(oct)]);
                } else {
                    sigma2 = std::pow(1.2 * 1.2, oct);
                }
                e->setInformation(Eigen::Matrix3d::Identity() * (1.0 / sigma2));
            }

            auto* rk = new g2o::RobustKernelHuber();
            rk->setDelta(p.huber_delta);
            e->setRobustKernel(rk);

            e->fx = cam.fx; e->fy = cam.fy;
            e->cx = cam.cx; e->cy = cam.cy;
            e->bf = bf;

            optimizer->addEdge(e);
            edge_infos.push_back({e, obs_kf, mp, feat_idx});
        }
    }

    if (edge_infos.empty()) return;

    // -----------------------------------------------------------------------
    // 5. Two-pass optimisation with outlier reclassification.
    // -----------------------------------------------------------------------
    for (int pass = 0; pass < 2; ++pass) {
        // Only reset estimates on the first pass; pass 2 starts from pass 1's
        // result (already stored in the g2o vertices).
        if (pass == 0) {
            for (KeyFrame* lkf : local_kf_set)
                static_cast<g2o::VertexSE3Expmap*>(
                    optimizer->vertex(kf_to_id.at(lkf)))
                    ->setEstimate(to_se3quat(lkf->get_pose()));
        }

        optimizer->initializeOptimization(0);
        optimizer->optimize(pass == 0 ? p.inner_iterations
                                      : p.inner_iterations * 2);

        // Reclassify after pass 1 only (pass 2 is the final clean solve).
        if (pass == 0) {
            for (auto& ei : edge_infos) {
                ei.edge->computeError();
                if (ei.edge->chi2() > p.chi2_threshold) {
                    ei.edge->setLevel(1);  // exclude from pass 2
                    ei.edge->setRobustKernel(nullptr);
                } else {
                    ei.edge->setLevel(0);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. Write back optimised KF poses and MP positions.
    //
    //    No magnitude guard: the local window has fixed cameras (KFs that
    //    observe local MPs but aren't being optimised) plus a fallback
    //    smallest-id anchor when the map is small, so gauge is constrained
    //    and large corrections are by construction legitimate.  Matches
    //    ORB-SLAM2 LocalBundleAdjustment, which writes back unconditionally.
    // -----------------------------------------------------------------------
    // 8. Write back optimised poses and point positions, remove outlier obs.
    // -----------------------------------------------------------------------
    for (KeyFrame* lkf : local_kf_set) {
        const auto* v = static_cast<const g2o::VertexSE3Expmap*>(
            optimizer->vertex(kf_to_id.at(lkf)));
        lkf->set_pose(from_se3quat(v->estimate()));
    }

    for (MapPoint* mp : local_mp_set) {
        const auto* v = static_cast<const g2o::VertexPointXYZ*>(
            optimizer->vertex(mp_to_id.at(mp)));
        mp->set_world_pos(v->estimate());
        mp->update_normal_and_depth();
    }

    // Remove observations whose final edge is an outlier.
    for (const auto& ei : edge_infos) {
        if (ei.edge->level() == 1 || ei.edge->chi2() > p.chi2_threshold) {
            ei.kf->erase_map_point(ei.feat_idx);
            ei.mp->remove_observation(ei.kf);
        }
    }

    for (KeyFrame* lkf : local_kf_set)
        lkf->update_connections();
    for (KeyFrame* fkf : fixed_kf_set)
        fkf->update_connections();
}

}  // namespace ba
}  // namespace sslam
