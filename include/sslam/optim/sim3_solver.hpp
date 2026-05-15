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
    bool   fix_scale{false};          ///< If true, force s=1 (stereo/RGB-D)
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

    /// @param pts1  Points in frame 1 (world frame).
    /// @param pts2  Corresponding points in frame 2 (world frame).
    Sim3Solver(const std::vector<Eigen::Vector3d>& pts1,
               const std::vector<Eigen::Vector3d>& pts2,
               const Params& p = Params{});

    /// Extended constructor that enables bidirectional reprojection-error
    /// inlier test (ORB-SLAM2 style) instead of 3-D Euclidean distance.
    ///
    /// @param obs1       2-D pixel observations in frame 1 (same order as pts1).
    /// @param obs2       2-D pixel observations in frame 2.
    /// @param max_err1   Per-correspondence chi² threshold for frame 1
    ///                   (e.g. 9.210 * sigma2[octave]).
    /// @param max_err2   Per-correspondence chi² threshold for frame 2.
    /// @param T_cw_1     World→camera pose of frame 1 (4×4 SE(3)).
    /// @param T_cw_2     World→camera pose of frame 2.
    /// @param fx,fy,cx,cy  Shared camera intrinsics.
    Sim3Solver(const std::vector<Eigen::Vector3d>& pts1,
               const std::vector<Eigen::Vector3d>& pts2,
               const std::vector<Eigen::Vector2d>& obs1,
               const std::vector<Eigen::Vector2d>& obs2,
               const std::vector<double>& max_err1,
               const std::vector<double>& max_err2,
               const Eigen::Matrix4d& T_cw_1,
               const Eigen::Matrix4d& T_cw_2,
               double fx, double fy, double cx, double cy,
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

    /// Bidirectional reprojection inlier check (used when use_reprojection_).
    int count_inliers_reproj(double s, const Eigen::Matrix3d& R,
                             const Eigen::Vector3d& t,
                             std::vector<bool>& mask) const;

    /// Refine the Sim3 using Horn on the full inlier set.
    void refine(double& s, Eigen::Matrix3d& R, Eigen::Vector3d& t,
                const std::vector<bool>& mask) const;

    const std::vector<Eigen::Vector3d>& pts1_;
    const std::vector<Eigen::Vector3d>& pts2_;
    Params p_;

    // --- Reprojection-mode data (empty when use_reprojection_ = false) ------
    bool use_reprojection_{false};
    double fx_{0}, fy_{0}, cx_{0}, cy_{0};
    std::vector<Eigen::Vector2d> obs1_, obs2_;
    std::vector<double>          max_err1_, max_err2_;
    Eigen::Matrix4d T_cw_1_{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d T_cw_2_{Eigen::Matrix4d::Identity()};
};

}  // namespace sslam
