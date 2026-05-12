#pragma once

#include <Eigen/Core>
#include <vector>

namespace sslam {

/// Parameters for Sim3Solver.
struct Sim3SolverParams {
    int    max_iterations{200};
    double inlier_threshold_m{0.2};  ///< 3-D inlier distance (metres)
    int    min_inliers{20};           ///< Minimum inliers for acceptance
    int    refine_iterations{5};      ///< Horn iterations on inlier refinement
};

/// Result returned by Sim3Solver::solve().
struct Sim3SolverResult {
    bool            found{false};
    double          scale{1.0};
    Eigen::Matrix3d R;               ///< Rotation:    p2 = s*R*p1 + t
    Eigen::Vector3d t;               ///< Translation
    int             n_inliers{0};
    std::vector<bool> inlier_mask;   ///< One entry per input correspondence
};

/// Sim3 (similarity transform, 7 DOF: rotation, translation, scale) solver.
///
/// Given 3D-3D point correspondences between two coordinate frames, recovers
/// the Sim3 that maps points in frame 1 into frame 2:
///   p2 = s * R * p1 + t
///
/// Algorithm: RANSAC + closed-form Horn (1987) quaternion solution.
///
/// Reference: B.K.P. Horn, "Closed-form solution of absolute orientation
///   using unit quaternions", JOSAA 1987.
class Sim3Solver {
   public:
    using Params = Sim3SolverParams;
    using Result = Sim3SolverResult;

    /// @param pts1  Points in frame 1.
    /// @param pts2  Corresponding points in frame 2.
    Sim3Solver(const std::vector<Eigen::Vector3d>& pts1,
               const std::vector<Eigen::Vector3d>& pts2,
               const Params& p = Params{});

    Result solve();

   private:
    /// Closed-form Horn solution on n>=3 correspondences (columns of p1, p2).
    /// Returns false if the system is degenerate.
    bool hornN(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2,
               double& s, Eigen::Matrix3d& R, Eigen::Vector3d& t) const;

    /// Count inliers for a given Sim3 hypothesis.
    int count_inliers(double s, const Eigen::Matrix3d& R,
                      const Eigen::Vector3d& t,
                      std::vector<bool>& mask) const;

    /// Refine the Sim3 using Horn on the full inlier set.
    void refine(double& s, Eigen::Matrix3d& R, Eigen::Vector3d& t,
                const std::vector<bool>& mask) const;

    const std::vector<Eigen::Vector3d>& pts1_;
    const std::vector<Eigen::Vector3d>& pts2_;
    Params p_;
};

}  // namespace sslam
