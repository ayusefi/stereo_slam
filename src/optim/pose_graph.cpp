#include "sslam/optim/pose_graph.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace sslam {
namespace pose_graph {

namespace {

/// Minimum covisibility weight to include an edge in the essential graph.
constexpr int kMinCovisWeight = 100;

g2o::Sim3 se3_to_sim3(const Eigen::Matrix4d& T_cw) {
    const Eigen::Matrix3d R = T_cw.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_cw.topRightCorner<3, 1>();
    return g2o::Sim3(Eigen::Quaterniond(R).normalized(), t, 1.0);
}

Eigen::Matrix4d sim3_to_se3(const g2o::Sim3& s) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>()  = s.rotation().toRotationMatrix();
    T.topRightCorner<3, 1>() = s.translation() / s.scale();
    return T;
}

}  // namespace

// ---------------------------------------------------------------------------

void optimize(Map& map,
              KeyFrame* query_kf,
              KeyFrame* match_kf,
              double s_qm,
              const Eigen::Matrix3d& R_qm,
              const Eigen::Vector3d& t_qm,
              int n_iters)
{
    using BlockSolverType  = g2o::BlockSolverX;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;

    auto linear = std::make_unique<LinearSolverType>();
    auto block  = std::make_unique<BlockSolverType>(std::move(linear));
    auto algo   = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    g2o::SparseOptimizer opt;
    opt.setAlgorithm(algo);
    opt.setVerbose(false);

    const std::vector<KeyFrame::Ptr> all_kfs = map.get_all_keyframes();

    // Pass 1: save old poses for MP correction after optimisation.
    std::unordered_map<KeyFrame*, Eigen::Matrix4d> old_T_cw;
    std::unordered_map<uint64_t, KeyFrame*> kf_by_id;
    old_T_cw.reserve(all_kfs.size());
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (!kf->is_bad()) {
            old_T_cw[kf.get()] = kf->get_pose();
            kf_by_id[kf->id()] = kf.get();
        }
    }

    // --- Build vertices -------------------------------------------------------
    std::unordered_map<KeyFrame*, g2o::VertexSim3Expmap*> v_map;
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad()) continue;
        auto* v = new g2o::VertexSim3Expmap();
        v->setId(static_cast<int>(kf->id()));
        v->setEstimate(se3_to_sim3(kf->get_pose()));
        v->_fix_scale = true;
        v->setFixed(kf->id() == 0);
        opt.addVertex(v);
        v_map[kf.get()] = v;
    }
    // Stereo SLAM: metric scale is fixed by the baseline — all vertices hold
    // unit scale throughout the optimisation.

    // --- Edges ----------------------------------------------------------------
    // g2o EdgeSim3 error = C * v0 * v1^{-1}, so for zero residual at the
    // current estimates we need C = v1 * v0^{-1}  (i.e. S_vertex1 * S_vertex0^{-1}).
    int edge_id = static_cast<int>(all_kfs.size()) + 1000;
    auto add_edge = [&](KeyFrame* ka, KeyFrame* kb, const g2o::Sim3& meas) {
        if (!v_map.count(ka) || !v_map.count(kb)) return;
        auto* e = new g2o::EdgeSim3();
        e->setId(++edge_id);
        e->setVertex(0, v_map.at(ka));
        e->setVertex(1, v_map.at(kb));
        e->setMeasurement(meas);
        e->setInformation(Eigen::Matrix<double, 7, 7>::Identity());
        opt.addEdge(e);
    };

    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad()) continue;
        if (kf->parent() && !kf->parent()->is_bad()) {
            // C = S_parent * S_kf^{-1}  (vertex0=kf, vertex1=parent)
            const g2o::Sim3 s_pk = se3_to_sim3(kf->parent()->get_pose())
                                  * se3_to_sim3(kf->get_pose()).inverse();
            add_edge(kf.get(), kf->parent(), s_pk);
        }
        for (KeyFrame* cov : kf->get_covisibility_keyframes(kMinCovisWeight)) {
            if (cov->id() >= kf->id() || cov->is_bad()) continue;
            // C = S_cov * S_kf^{-1}  (vertex0=kf, vertex1=cov)
            const g2o::Sim3 s_ck = se3_to_sim3(cov->get_pose())
                                  * se3_to_sim3(kf->get_pose()).inverse();
            add_edge(kf.get(), cov, s_ck);
        }
    }

    // Defensive odometry-chain edges.  The parent spanning tree is the primary
    // essential-graph backbone, but older maps or culled-parent gaps can leave
    // it incomplete.  Consecutive active KFs keep the graph connected enough for
    // a loop edge to distribute correction instead of moving one component.
    std::vector<KeyFrame*> ordered_kfs;
    ordered_kfs.reserve(all_kfs.size());
    for (const KeyFrame::Ptr& kf : all_kfs)
        if (kf && !kf->is_bad()) ordered_kfs.push_back(kf.get());
    std::sort(ordered_kfs.begin(), ordered_kfs.end(),
              [](const KeyFrame* a, const KeyFrame* b) { return a->id() < b->id(); });
    for (std::size_t i = 1; i < ordered_kfs.size(); ++i) {
        KeyFrame* prev = ordered_kfs[i - 1];
        KeyFrame* curr = ordered_kfs[i];
        const g2o::Sim3 s_pc = se3_to_sim3(prev->get_pose())
                              * se3_to_sim3(curr->get_pose()).inverse();
        add_edge(curr, prev, s_pc);
    }

    // Loop-closure edge.
    // The world-frame Sim3 {s_qm, R_qm, t_qm} maps query-world → match-world:
    //   X_m_world ≈ s_qm * R_qm * X_q_world + t_qm
    // Corrected query camera pose (in match-world frame):
    //   S_q_corr = se3_to_sim3(T_cw_query) * S_w^{-1}
    // Measurement for edge (v0=match, v1=query) = S_q_corr * S_match^{-1}
    {
        const g2o::Sim3 S_w(Eigen::Quaterniond(R_qm).normalized(), t_qm, s_qm);
        const g2o::Sim3 S_q_old = se3_to_sim3(query_kf->get_pose());
        const g2o::Sim3 S_m     = se3_to_sim3(match_kf->get_pose());
        const g2o::Sim3 S_q_corr = S_q_old * S_w.inverse();
        // Pre-set query vertex to corrected estimate so the PGO starts near
        // the correct answer (mirrors ORB-SLAM2's CorrectLoop() approach).
        if (v_map.count(query_kf))
            v_map.at(query_kf)->setEstimate(S_q_corr);
        add_edge(match_kf, query_kf, S_q_corr * S_m.inverse());
    }

    // --- Optimise -------------------------------------------------------------
    opt.initializeOptimization();
    opt.optimize(n_iters);

    // --- Write back KF poses --------------------------------------------------
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad() || !v_map.count(kf.get())) continue;
        kf->set_pose(sim3_to_se3(v_map.at(kf.get())->estimate()));
    }

    // --- Move MapPoints with their reference KF correction -------------------
    // X_new = T_wk_new * T_wk_old^{-1} * X_old
    //       = T_cw_new^{-1} * T_cw_old * X_old
    const std::vector<MapPoint::Ptr> all_mps = map.get_all_mappoints();
    for (const MapPoint::Ptr& mp : all_mps) {
        if (mp->is_bad()) continue;
        const auto obs = mp->get_observations();
        if (obs.empty()) continue;
        KeyFrame* ref = nullptr;
        if (kf_by_id.count(mp->created_kf_id())) {
            ref = kf_by_id.at(mp->created_kf_id());
        } else {
            for (const auto& [obs_kf, feat_idx] : obs) {
                if (!obs_kf || !old_T_cw.count(obs_kf)) continue;
                if (!ref || obs_kf->id() < ref->id()) ref = obs_kf;
            }
        }
        if (!ref) continue;
        if (!old_T_cw.count(ref)) continue;

        const Eigen::Matrix4d& T_old = old_T_cw.at(ref);
        const Eigen::Matrix4d  T_new = ref->get_pose();  // already updated above

        // T_wk_new = T_new^{-1}, T_wk_old = T_old^{-1}
        // correction = T_wk_new * T_wk_old^{-1} = T_new^{-1} * T_old
        const Eigen::Matrix4d T_wk_new =
            Eigen::Matrix4d(T_new).inverse();  // 4×4 inversion
        const Eigen::Vector3d X_old = mp->get_world_pos();
        // X_local = T_cw_old * X_old (in camera frame of ref)
        const Eigen::Vector3d X_local =
            T_old.topLeftCorner<3, 3>() * X_old + T_old.topRightCorner<3, 1>();
        // X_new = T_wk_new * X_local
        const Eigen::Vector3d X_new =
            T_wk_new.topLeftCorner<3, 3>() * X_local
            + T_wk_new.topRightCorner<3, 1>();
        mp->set_world_pos(X_new);
    }
}

}  // namespace pose_graph
}  // namespace sslam

