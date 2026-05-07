#pragma once

// Thin re-export of the g2o types used by the motion-only BA.
//
// We use g2o's built-in stereo-only-pose edge rather than defining a custom
// one.  This header centralises all g2o includes so the rest of the codebase
// never includes g2o headers directly.

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/edge_project_stereo_xyz_onlypose.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
