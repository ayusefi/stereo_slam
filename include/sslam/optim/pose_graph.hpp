#pragma once

#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"

#include <Eigen/Core>

namespace sslam {
namespace pose_graph {

/// Essential-graph Sim3 pose-graph optimisation.
///
/// Vertices: one VertexSim3Expmap per non-bad KeyFrame (scale fixed = 1
///           for all existing KFs; the loop match vertex has scale free).
/// Edges (measurement = relative Sim3 from current estimates):
///   - Spanning-tree parent edges.
///   - Strong covisibility edges (weight >= 100).
///   - The single loop-closure edge (query → match, with measured Sim3).
///
/// After optimisation the corrected Sim3 poses are written back to every
/// KF, and every MapPoint is moved with its reference KF.
///
/// @param map        The map to optimise in-place.
/// @param query_kf   The query KeyFrame on the loop.
/// @param match_kf   The matched KeyFrame on the loop.
/// @param S_qm       Sim3 that maps match-side into query-side world frame
///                   (from Sim3Solver): p_q = s*R*p_m + t.
///                   The edge measurement is S_mq = S_qm^{-1}.
/// @param n_iters    LM iterations (default 20).
void optimize(Map& map,
              KeyFrame* query_kf,
              KeyFrame* match_kf,
              double s_qm,
              const Eigen::Matrix3d& R_qm,
              const Eigen::Vector3d& t_qm,
              int n_iters = 20);

}  // namespace pose_graph
}  // namespace sslam
