// Motion-only bundle adjustment.
//
// Optimises a single camera pose (SE3Expmap vertex) against a fixed set of
// 3-D world points observed as stereo correspondences (u_l, v, u_r).
//
// The optimiser runs `outer_iterations` Levenberg-Marquardt rounds.  After
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

#include <Eigen/Geometry>

#include <cassert>
#include <memory>

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

    for (int i = 0; i < n; ++i) {
        auto* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

        e->setVertex(0, v);
        e->setMeasurement(obs_stereo[static_cast<std::size_t>(i)]);
        e->setInformation(Eigen::Matrix3d::Identity());

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
    result.T_cw    = from_se3quat(v->estimate());
    result.n_inliers = n_inliers;
    return result;
}

}  // namespace ba
}  // namespace sslam
