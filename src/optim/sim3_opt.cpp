// g2o Sim3 refinement (ORB-SLAM2 OptimizeSim3 equivalent).
//
// Sets up a single VertexSim3Expmap (query-cam → match-cam) with fixed
// VertexPointXYZ observations in each camera frame:
//   EdgeSim3ProjectXYZ        : query-cam point → project into match cam.
//   EdgeInverseSim3ProjectXYZ : match-cam point → project into query cam.
//
// Two optimisation rounds (5 + 5 LM iters), Huber kernel delta = sqrt(10).

#include "sslam/optim/sim3_opt.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <cmath>
#include <memory>

namespace sslam {
namespace optim {

namespace {

// chi2 threshold at α=0.01, 2-DOF (ORB-SLAM2 uses 9.210 for monocular;
// same value is appropriate here since each edge has 2-DOF residual).
constexpr double kChi2Th = 9.210;
constexpr double kHuberDelta = 3.162;  // sqrt(10)

g2o::Sim3 make_sim3(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, double s) {
    return g2o::Sim3(Eigen::Quaterniond(R).normalized(), t, s);
}

}  // namespace

Sim3OptResult optimize_sim3(
    const std::vector<Eigen::Vector3d>& pts_q_w,
    const std::vector<Eigen::Vector3d>& pts_m_w,
    const std::vector<Eigen::Vector2d>& obs_q,
    const std::vector<Eigen::Vector2d>& obs_m,
    const std::vector<double>& sigma2_q,
    const std::vector<double>& sigma2_m,
    const std::vector<bool>& init_mask,
    const Eigen::Matrix4d& T_cw_q,
    const Eigen::Matrix4d& T_cw_m,
    double s, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const StereoCamera& cam,
    bool fix_scale)
{
    const int n = static_cast<int>(pts_q_w.size());

    // -----------------------------------------------------------------------
    // Build the camera-relative initial Sim3: S_cm_cq (query-cam → match-cam)
    //   S_cm_cq * p_cam_q = p_cam_m
    //   Since p_cam_q = T_cw_q * p_world_q  and  p_world_m = s*R*p_world_q+t
    //   and   p_cam_m = T_cw_m * p_world_m
    //   => S_cm_cq is encoded in T_cw_m * SE3(s,R,t) * T_wc_q
    // -----------------------------------------------------------------------
    const Eigen::Matrix4d T_wc_q = T_cw_q.inverse();
    const Eigen::Matrix4d T_wc_m = T_cw_m.inverse();

    // SE3 from RANSAC (s=1 for stereo, so we treat it as pure SE3)
    Eigen::Matrix4d T_qm_world = Eigen::Matrix4d::Identity();
    T_qm_world.topLeftCorner<3,3>() = R;
    T_qm_world.topRightCorner<3,1>() = t;

    // Combined: maps query-cam → match-cam
    const Eigen::Matrix4d T_init_cam = T_cw_m * T_qm_world * T_wc_q;
    const Eigen::Matrix3d R_init = T_init_cam.topLeftCorner<3,3>();
    const Eigen::Vector3d t_init = T_init_cam.topRightCorner<3,1>();

    // -----------------------------------------------------------------------
    // g2o optimizer
    // -----------------------------------------------------------------------
    using BlockSolverType  = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto opt = std::make_unique<g2o::SparseOptimizer>();
    opt->setVerbose(false);
    opt->setAlgorithm(
        new g2o::OptimizationAlgorithmLevenberg(
            std::make_unique<BlockSolverType>(
                std::make_unique<LinearSolverType>())));

    // -----------------------------------------------------------------------
    // Sim3 vertex (the single free variable)
    // -----------------------------------------------------------------------
    auto* v_sim3 = new g2o::VertexSim3Expmap();
    v_sim3->setId(0);
    v_sim3->setFixed(false);
    v_sim3->_fix_scale = fix_scale;
    v_sim3->setEstimate(make_sim3(R_init, t_init, fix_scale ? 1.0 : s));
    v_sim3->_principle_point1 = {cam.cx, cam.cy};
    v_sim3->_focal_length1    = {cam.fx, cam.fy};
    v_sim3->_principle_point2 = {cam.cx, cam.cy};
    v_sim3->_focal_length2    = {cam.fx, cam.fy};
    opt->addVertex(v_sim3);

    // -----------------------------------------------------------------------
    // Point vertices and edges — one forward + one backward per inlier.
    // Point vertices are FIXED (we're only optimising the pose).
    // Points stored in camera frame (not world frame).
    // -----------------------------------------------------------------------
    std::vector<g2o::VertexPointXYZ*> vpts_q(static_cast<std::size_t>(n), nullptr);
    std::vector<g2o::VertexPointXYZ*> vpts_m(static_cast<std::size_t>(n), nullptr);
    std::vector<g2o::EdgeSim3ProjectXYZ*>        efwd(static_cast<std::size_t>(n), nullptr);
    std::vector<g2o::EdgeInverseSim3ProjectXYZ*> ebwd(static_cast<std::size_t>(n), nullptr);

    // Base vertex id for points: interleave query (2*i+1) and match (2*i+2)
    int pt_id = 1;
    for (int i = 0; i < n; ++i) {
        if (!init_mask[static_cast<std::size_t>(i)]) continue;

        // Query-cam point (for forward edge)
        const Eigen::Vector3d p_cq = T_cw_q.topLeftCorner<3,3>() * pts_q_w[static_cast<std::size_t>(i)]
                                   + T_cw_q.topRightCorner<3,1>();
        if (p_cq.z() <= 0.0) continue;

        // Match-cam point (for backward edge)
        const Eigen::Vector3d p_cm = T_cw_m.topLeftCorner<3,3>() * pts_m_w[static_cast<std::size_t>(i)]
                                   + T_cw_m.topRightCorner<3,1>();
        if (p_cm.z() <= 0.0) continue;

        // Query-cam VertexPointXYZ
        auto* vpq = new g2o::VertexPointXYZ();
        vpq->setId(pt_id++);
        vpq->setEstimate(p_cq);
        vpq->setFixed(true);
        opt->addVertex(vpq);
        vpts_q[static_cast<std::size_t>(i)] = vpq;

        // Match-cam VertexPointXYZ
        auto* vpm = new g2o::VertexPointXYZ();
        vpm->setId(pt_id++);
        vpm->setEstimate(p_cm);
        vpm->setFixed(true);
        opt->addVertex(vpm);
        vpts_m[static_cast<std::size_t>(i)] = vpm;

        // Forward edge: query-cam point → project into match cam via S_cm_cq
        auto* ef = new g2o::EdgeSim3ProjectXYZ();
        ef->setVertex(0, vpq);
        ef->setVertex(1, v_sim3);
        ef->setMeasurement(obs_m[static_cast<std::size_t>(i)]);
        const double w_m = 1.0 / sigma2_m[static_cast<std::size_t>(i)];
        ef->setInformation(Eigen::Matrix2d::Identity() * w_m);
        auto* rk_f = new g2o::RobustKernelHuber();
        rk_f->setDelta(kHuberDelta);
        ef->setRobustKernel(rk_f);
        opt->addEdge(ef);
        efwd[static_cast<std::size_t>(i)] = ef;

        // Backward edge: match-cam point → project into query cam via S_cm_cq^{-1}
        auto* eb = new g2o::EdgeInverseSim3ProjectXYZ();
        eb->setVertex(0, vpm);
        eb->setVertex(1, v_sim3);
        eb->setMeasurement(obs_q[static_cast<std::size_t>(i)]);
        const double w_q = 1.0 / sigma2_q[static_cast<std::size_t>(i)];
        eb->setInformation(Eigen::Matrix2d::Identity() * w_q);
        auto* rk_b = new g2o::RobustKernelHuber();
        rk_b->setDelta(kHuberDelta);
        eb->setRobustKernel(rk_b);
        opt->addEdge(eb);
        ebwd[static_cast<std::size_t>(i)] = eb;
    }

    // -----------------------------------------------------------------------
    // Round 1: 5 LM iterations with Huber kernels.
    // -----------------------------------------------------------------------
    opt->initializeOptimization();
    opt->optimize(5);

    // -----------------------------------------------------------------------
    // Classify inliers, deactivate outlier edges for round 2.
    // -----------------------------------------------------------------------
    std::vector<bool> cur_mask(static_cast<std::size_t>(n), false);
    int n_inliers = 0;
    for (int i = 0; i < n; ++i) {
        auto* ef = efwd[static_cast<std::size_t>(i)];
        auto* eb = ebwd[static_cast<std::size_t>(i)];
        if (!ef || !eb) continue;

        const double chi2_f = ef->chi2();
        const double chi2_b = eb->chi2();
        const double th_f   = kChi2Th * sigma2_m[static_cast<std::size_t>(i)];
        const double th_b   = kChi2Th * sigma2_q[static_cast<std::size_t>(i)];

        if (chi2_f <= th_f && chi2_b <= th_b) {
            cur_mask[static_cast<std::size_t>(i)] = true;
            ++n_inliers;
            // Remove robust kernel for the refinement round.
            ef->setRobustKernel(nullptr);
            eb->setRobustKernel(nullptr);
        } else {
            // Deactivate (level=1 means excluded by g2o).
            ef->setLevel(1);
            eb->setLevel(1);
        }
    }

    // -----------------------------------------------------------------------
    // Round 2: 5 more iterations on inliers only, no robust kernel.
    // -----------------------------------------------------------------------
    if (n_inliers >= 3) {
        opt->initializeOptimization(0);  // level 0 only
        opt->optimize(5);

        // Final inlier re-count.
        n_inliers = 0;
        cur_mask.assign(static_cast<std::size_t>(n), false);
        for (int i = 0; i < n; ++i) {
            auto* ef = efwd[static_cast<std::size_t>(i)];
            auto* eb = ebwd[static_cast<std::size_t>(i)];
            if (!ef || !eb || ef->level() == 1) continue;
            const double chi2_f = ef->chi2();
            const double chi2_b = eb->chi2();
            if (chi2_f <= kChi2Th * sigma2_m[static_cast<std::size_t>(i)] &&
                chi2_b <= kChi2Th * sigma2_q[static_cast<std::size_t>(i)]) {
                cur_mask[static_cast<std::size_t>(i)] = true;
                ++n_inliers;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Extract optimised Sim3 and convert back to world-space.
    // S_cm_cq_opt maps query-cam → match-cam.
    // Recover world-space: T_qm = T_wc_m * T_opt_cam * T_cw_q
    // -----------------------------------------------------------------------
    const g2o::Sim3 S_opt = v_sim3->estimate();
    const double s_opt = fix_scale ? 1.0 : S_opt.scale();
    Eigen::Matrix3d R_opt_cam = S_opt.rotation().toRotationMatrix();
    Eigen::Vector3d t_opt_cam = S_opt.translation() / S_opt.scale();

    Eigen::Matrix4d T_opt_cam = Eigen::Matrix4d::Identity();
    T_opt_cam.topLeftCorner<3,3>()  = R_opt_cam;
    T_opt_cam.topRightCorner<3,1>() = t_opt_cam;

    const Eigen::Matrix4d T_qm_world_opt = T_wc_m * T_opt_cam * T_cw_q;

    Sim3OptResult res;
    res.scale      = s_opt;
    res.R          = T_qm_world_opt.topLeftCorner<3,3>();
    res.t          = T_qm_world_opt.topRightCorner<3,1>();
    res.n_inliers  = n_inliers;
    res.inlier_mask = cur_mask;
    return res;
}

}  // namespace optim
}  // namespace sslam
