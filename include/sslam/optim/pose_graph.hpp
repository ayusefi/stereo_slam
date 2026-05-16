#pragma once

#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"

#include <Eigen/Core>

namespace sslam {
namespace pose_graph {

struct CorrectionStats {
    bool     valid{false};

    // Translation-based correction metrics.
    double   max_center_correction_m{0.0};
    uint64_t max_center_kf_id{0};
    double   max_adjacent_step_m{0.0};
    uint64_t max_adjacent_step_kf_id{0};
    double   query_center_correction_m{0.0};
    double   match_center_correction_m{0.0};

    // Rotation-based correction metric.
    double   max_rotation_correction_deg{0.0};

    // Pose-graph topology.
    int      graph_vertices{0};
    int      graph_edges{0};
    int      graph_components{1};  ///< 1 = connected (healthy)
};

/// Essential-graph Sim3 pose-graph optimisation.
///
/// Vertices: one VertexSim3Expmap per non-bad KeyFrame (scale fixed = 1
///           for all existing KFs; the loop match vertex has scale free).
/// Edges (measurement = relative Sim3 from current estimates):
///   - Spanning-tree parent edges          (info scale = 1.0).
///   - Strong covisibility edges           (info scale ∝ covis weight, capped at 3).
///   - The single loop-closure edge        (info scale ∝ loop_inliers / 30).
///
/// After optimisation the corrected Sim3 poses are written back to every
/// KF, and every MapPoint is moved with its reference KF.
///
/// @param map           The map to optimise in-place.
/// @param query_kf      The query KeyFrame on the loop.
/// @param match_kf      The matched KeyFrame on the loop.
/// @param S_qm          Sim3 from Sim3Solver that maps query-side world points
///                      into match-side world points: p_m = s*R*p_q + t.
/// @param loop_inliers  Number of Sim3 RANSAC inliers for the loop edge.
///                      Used to scale the loop edge's information matrix
///                      relative to structural edges (default 30 = minimum).
/// @param n_iters       LM iterations (default 20).
void optimize(Map& map,
              KeyFrame* query_kf,
              KeyFrame* match_kf,
              double s_qm,
              const Eigen::Matrix3d& R_qm,
              const Eigen::Vector3d& t_qm,
              int loop_inliers = 30,
              int n_iters = 20);

/// Run the same essential-graph optimisation without writing poses or
/// MapPoints back.  Used by loop closing to reject implausible corrections
/// before fusing duplicate MapPoints or mutating the map.
CorrectionStats preview(Map& map,
                        KeyFrame* query_kf,
                        KeyFrame* match_kf,
                        double s_qm,
                        const Eigen::Matrix3d& R_qm,
                        const Eigen::Vector3d& t_qm,
                        int loop_inliers = 30,
                        int n_iters = 20);

}  // namespace pose_graph
}  // namespace sslam
