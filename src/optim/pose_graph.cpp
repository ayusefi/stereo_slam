#include "sslam/optim/pose_graph.hpp"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <numeric>

#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>
#include <unordered_set>
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

Eigen::Vector3d camera_center(const Eigen::Matrix4d& T_cw) {
    return -T_cw.topLeftCorner<3, 3>().transpose() *
            T_cw.topRightCorner<3, 1>();
}

CorrectionStats optimize_impl(Map& map,
                              KeyFrame* query_kf,
                              KeyFrame* match_kf,
                              double s_qm,
                              const Eigen::Matrix3d& R_qm,
                              const Eigen::Vector3d& t_qm,
                              int loop_inliers,
                              int n_iters,
                              bool write_back) {
    CorrectionStats stats;

    // Sparse essential-graph optimisation.  Sim3 vertices have 7 DOF, so use
    // the 7_3 block solver with an Eigen sparse linear solver.  A dense solver
    // here is O((7N)^3) per iteration and stalls for minutes once the map has
    // hundreds of keyframes; the sparse Cholesky factorisation exploits the
    // graph sparsity (each KF connects to a handful of covisible neighbours)
    // and runs in well under a second, matching ORB-SLAM2's OptimizeEssentialGraph.
    using BlockSolverType  = g2o::BlockSolver_7_3;
    using LinearSolverType = g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType>;

    auto linear = std::make_unique<LinearSolverType>();
    auto block  = std::make_unique<BlockSolverType>(std::move(linear));
    auto algo   = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    g2o::SparseOptimizer opt;
    opt.setAlgorithm(algo);
    opt.setVerbose(false);

    const std::vector<KeyFrame::Ptr> all_kfs = map.get_all_keyframes();

    std::unordered_map<KeyFrame*, Eigen::Matrix4d> old_T_cw;
    std::unordered_map<uint64_t, KeyFrame*> kf_by_id;
    old_T_cw.reserve(all_kfs.size());
    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (!kf->is_bad()) {
            old_T_cw[kf.get()] = kf->get_pose();
            kf_by_id[kf->id()] = kf.get();
        }
    }
    if (!old_T_cw.count(query_kf) || !old_T_cw.count(match_kf)) return stats;

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

    int edge_id = static_cast<int>(all_kfs.size()) + 1000;
    std::vector<std::pair<KeyFrame*, KeyFrame*>> edge_endpoints;
    auto add_edge = [&](KeyFrame* ka, KeyFrame* kb, const g2o::Sim3& meas,
                        double info_scale = 1.0) {
        if (!v_map.count(ka) || !v_map.count(kb)) return;
        auto* e = new g2o::EdgeSim3();
        e->setId(++edge_id);
        e->setVertex(0, v_map.at(ka));
        e->setVertex(1, v_map.at(kb));
        e->setMeasurement(meas);
        e->setInformation(info_scale * Eigen::Matrix<double, 7, 7>::Identity());
        opt.addEdge(e);
        edge_endpoints.emplace_back(ka, kb);
    };

    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad()) continue;
        // Spanning-tree edge: kf → parent (info_scale = 1.0)
        if (kf->parent() && !kf->parent()->is_bad()) {
            if (!old_T_cw.count(kf->parent())) continue; // shouldn't happen
            const g2o::Sim3 s_pk = se3_to_sim3(old_T_cw.at(kf->parent()))
                                  * se3_to_sim3(old_T_cw.at(kf.get())).inverse();
            add_edge(kf.get(), kf->parent(), s_pk, 1.0);
        }
        // Covisibility edges: scale by covisibility weight (capped at 3×)
        for (KeyFrame* cov : kf->get_covisibility_keyframes(kMinCovisWeight)) {
            if (cov->id() >= kf->id() || cov->is_bad()) continue;
            if (!old_T_cw.count(cov)) continue;
            const int w = kf->get_covisibility_weight(cov);
            const double info_scale = std::min(3.0, w / 100.0);
            const g2o::Sim3 s_ck = se3_to_sim3(old_T_cw.at(cov))
                                  * se3_to_sim3(old_T_cw.at(kf.get())).inverse();
            add_edge(kf.get(), cov, s_ck, info_scale);
        }
    }

    // Loop-closure constraint edge: scale by inlier quality
    {
        const double loop_info_scale = std::max(1.0, loop_inliers / 30.0);
        const g2o::Sim3 S_w(Eigen::Quaterniond(R_qm).normalized(), t_qm, s_qm);
        const g2o::Sim3 S_q_old = se3_to_sim3(old_T_cw.at(query_kf));
        const g2o::Sim3 S_m     = se3_to_sim3(old_T_cw.at(match_kf));
        const g2o::Sim3 S_q_corr = S_q_old * S_w.inverse();
        if (v_map.count(query_kf))
            v_map.at(query_kf)->setEstimate(S_q_corr);
        add_edge(match_kf, query_kf, S_q_corr * S_m.inverse(), loop_info_scale);
    }

    // --- Connectivity check via union-find --------------------------------
    // Build adjacency from collected edge endpoints over the vertex set.
    {
        std::unordered_map<KeyFrame*, KeyFrame*> uf_parent;
        uf_parent.reserve(v_map.size());
        for (const auto& [kf, v] : v_map) uf_parent[kf] = kf;

        std::function<KeyFrame*(KeyFrame*)> uf_find = [&](KeyFrame* x) -> KeyFrame* {
            while (uf_parent.at(x) != x) {
                uf_parent[x] = uf_parent.at(uf_parent.at(x));  // path compression
                x = uf_parent.at(x);
            }
            return x;
        };

        for (const auto& [a, b] : edge_endpoints) {
            if (!uf_parent.count(a) || !uf_parent.count(b)) continue;
            KeyFrame* ra = uf_find(a);
            KeyFrame* rb = uf_find(b);
            if (ra != rb) uf_parent[ra] = rb;
        }

        std::unordered_set<KeyFrame*> roots;
        for (auto& [kf, _parent] : uf_parent) roots.insert(uf_find(kf));
        stats.graph_components = static_cast<int>(roots.size());
        stats.graph_vertices   = static_cast<int>(v_map.size());
        stats.graph_edges      = static_cast<int>(edge_endpoints.size());

        // Only reject if the loop endpoints themselves are in different
        // components — orphan KFs elsewhere in the map (e.g. from KF-culling
        // re-parenting failures) are tolerated; their poses simply remain
        // unchanged by PGO.  ORB-SLAM2 makes no global-connectivity demand.
        if (uf_parent.count(query_kf) && uf_parent.count(match_kf)) {
            KeyFrame* rq = uf_find(query_kf);
            KeyFrame* rm = uf_find(match_kf);
            if (rq != rm) {
                std::cerr << "[PGO] reject: loop endpoints in different components ("
                          << stats.graph_components << " components total, "
                          << stats.graph_vertices << " vertices, "
                          << stats.graph_edges << " edges)\n";
                return stats;  // stats.valid remains false
            }
        }
    }

    opt.initializeOptimization();
    opt.optimize(n_iters);

    stats.valid = true;
    struct CenterById {
        uint64_t id;
        Eigen::Vector3d center;
    };
    std::vector<CenterById> centers;
    centers.reserve(old_T_cw.size());
    for (const auto& [kf, old_pose] : old_T_cw) {
        if (!v_map.count(kf)) continue;
        const Eigen::Matrix4d new_pose = sim3_to_se3(v_map.at(kf)->estimate());
        const Eigen::Vector3d new_center = camera_center(new_pose);
        centers.push_back({kf->id(), new_center});
        const double correction =
            (new_center - camera_center(old_pose)).norm();
        if (correction > stats.max_center_correction_m) {
            stats.max_center_correction_m = correction;
            stats.max_center_kf_id = kf->id();
        }
        if (kf == query_kf) stats.query_center_correction_m = correction;
        if (kf == match_kf) stats.match_center_correction_m = correction;

        // Rotation correction angle.
        const Eigen::Matrix3d R_corr =
            new_pose.topLeftCorner<3, 3>() *
            old_pose.topLeftCorner<3, 3>().transpose();
        const double angle_deg =
            Eigen::AngleAxisd(R_corr).angle() * (180.0 / M_PI);
        if (angle_deg > stats.max_rotation_correction_deg)
            stats.max_rotation_correction_deg = angle_deg;
    }
    std::sort(centers.begin(), centers.end(),
              [](const CenterById& a, const CenterById& b) {
                  return a.id < b.id;
              });
    for (std::size_t i = 1; i < centers.size(); ++i) {
        const double step = (centers[i].center - centers[i - 1].center).norm();
        if (step > stats.max_adjacent_step_m) {
            stats.max_adjacent_step_m = step;
            stats.max_adjacent_step_kf_id = centers[i - 1].id;
        }
    }

    if (!write_back) return stats;

    for (const KeyFrame::Ptr& kf : all_kfs) {
        if (kf->is_bad() || !v_map.count(kf.get())) continue;
        kf->set_pose(sim3_to_se3(v_map.at(kf.get())->estimate()));
    }

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
                (void)feat_idx;
                if (!obs_kf || !old_T_cw.count(obs_kf)) continue;
                if (!ref || obs_kf->id() < ref->id()) ref = obs_kf;
            }
        }
        if (!ref || !old_T_cw.count(ref)) continue;

        const Eigen::Matrix4d& T_old = old_T_cw.at(ref);
        const Eigen::Matrix4d  T_new = ref->get_pose();
        const Eigen::Matrix4d T_wk_new = Eigen::Matrix4d(T_new).inverse();
        const Eigen::Vector3d X_old = mp->get_world_pos();
        const Eigen::Vector3d X_local =
            T_old.topLeftCorner<3, 3>() * X_old + T_old.topRightCorner<3, 1>();
        const Eigen::Vector3d X_new =
            T_wk_new.topLeftCorner<3, 3>() * X_local
            + T_wk_new.topRightCorner<3, 1>();
        mp->set_world_pos(X_new);
    }

    return stats;
}

}  // namespace

// ---------------------------------------------------------------------------

void optimize(Map& map,
              KeyFrame* query_kf,
              KeyFrame* match_kf,
              double s_qm,
              const Eigen::Matrix3d& R_qm,
              const Eigen::Vector3d& t_qm,
              int loop_inliers,
              int n_iters)
{
    (void)optimize_impl(map, query_kf, match_kf, s_qm, R_qm, t_qm,
                        loop_inliers, n_iters, true);
}

CorrectionStats preview(Map& map,
                        KeyFrame* query_kf,
                        KeyFrame* match_kf,
                        double s_qm,
                        const Eigen::Matrix3d& R_qm,
                        const Eigen::Vector3d& t_qm,
                        int loop_inliers,
                        int n_iters) {
    return optimize_impl(map, query_kf, match_kf, s_qm, R_qm, t_qm,
                         loop_inliers, n_iters, false);
}

}  // namespace pose_graph
}  // namespace sslam

