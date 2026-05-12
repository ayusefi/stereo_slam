#include "sslam/optim/sim3_solver.hpp"

#include <Eigen/Dense>
#include <random>

namespace sslam {

namespace {

/// Build the symmetric 4×4 cross-covariance matrix M needed by Horn's method.
Eigen::Matrix4d build_horn_m(const Eigen::Matrix3d& sigma) {
    // Horn (1987) eq. 65 — Nx matrix from the cross-covariance.
    // sigma = sum_i (p2_i - mu2) * (p1_i - mu1)^T
    const double Sxx = sigma(0, 0), Sxy = sigma(0, 1), Sxz = sigma(0, 2);
    const double Syx = sigma(1, 0), Syy = sigma(1, 1), Syz = sigma(1, 2);
    const double Szx = sigma(2, 0), Szy = sigma(2, 1), Szz = sigma(2, 2);

    Eigen::Matrix4d N;
    N << Sxx + Syy + Szz,  Syz - Szy,          Szx - Sxz,          Sxy - Syx,
         Syz - Szy,        Sxx - Syy - Szz,    Sxy + Syx,          Szx + Sxz,
         Szx - Sxz,        Sxy + Syx,         -Sxx + Syy - Szz,    Syz + Szy,
         Sxy - Syx,        Szx + Sxz,          Syz + Szy,         -Sxx - Syy + Szz;
    return N;
}

}  // namespace

// ---------------------------------------------------------------------------

Sim3Solver::Sim3Solver(const std::vector<Eigen::Vector3d>& pts1,
                       const std::vector<Eigen::Vector3d>& pts2,
                       const Params& p)
    : pts1_(pts1), pts2_(pts2), p_(p)
{}

// ---------------------------------------------------------------------------

bool Sim3Solver::hornN(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2,
                       double& s, Eigen::Matrix3d& R,
                       Eigen::Vector3d& t) const
{
    const int n = static_cast<int>(p1.cols());
    if (n < 3) return false;

    // Centroids.
    const Eigen::Vector3d mu1 = p1.rowwise().mean();
    const Eigen::Vector3d mu2 = p2.rowwise().mean();

    // Centred point matrices (3×n).
    const Eigen::MatrixXd c1 = p1.colwise() - mu1;
    const Eigen::MatrixXd c2 = p2.colwise() - mu2;

    // Cross-covariance: sigma_ij = sum_k c1[i,k] * c2[j,k]  (Horn 1987 eq. 9)
    // This is C1 * C2^T  (note: NOT C2 * C1^T).
    const Eigen::Matrix3d sigma = (c1 * c2.transpose()).eval();

    const Eigen::Matrix4d N = build_horn_m(sigma);

    // Largest eigenvector → optimal unit quaternion (w,x,y,z).
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(N);
    const Eigen::Vector4d q_vec = eig.eigenvectors().col(3);
    const Eigen::Quaterniond q(q_vec(0), q_vec(1), q_vec(2), q_vec(3));
    R = q.normalized().toRotationMatrix();

    // Scale: s = trace(R * sigma) / ||C1||_F^2   (Horn 1987 eq. 42)
    // = sum_k c2_k . (R * c1_k) / sum_k ||c1_k||^2
    const double num   = (R * sigma).trace();
    const double denom = c1.squaredNorm();
    if (denom < 1e-9 || num < 1e-9) return false;
    s = num / denom;

    // Translation.
    t = mu2 - s * R * mu1;
    return true;
}

// ---------------------------------------------------------------------------

int Sim3Solver::count_inliers(double s, const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& t,
                               std::vector<bool>& mask) const
{
    const double th2 = p_.inlier_threshold_m * p_.inlier_threshold_m;
    mask.assign(pts1_.size(), false);
    int n = 0;
    for (std::size_t i = 0; i < pts1_.size(); ++i) {
        const Eigen::Vector3d p2_est = s * R * pts1_[i] + t;
        if ((p2_est - pts2_[i]).squaredNorm() < th2) {
            mask[i] = true;
            ++n;
        }
    }
    return n;
}

// ---------------------------------------------------------------------------

void Sim3Solver::refine(double& s, Eigen::Matrix3d& R,
                         Eigen::Vector3d& t,
                         const std::vector<bool>& mask) const
{
    std::vector<Eigen::Vector3d> in1, in2;
    for (std::size_t i = 0; i < pts1_.size(); ++i) {
        if (mask[i]) { in1.push_back(pts1_[i]); in2.push_back(pts2_[i]); }
    }
    if (in1.size() < 3) return;

    // Build dynamic column matrices (3×n).
    Eigen::MatrixXd p1_m(3, static_cast<int>(in1.size()));
    Eigen::MatrixXd p2_m(3, static_cast<int>(in2.size()));
    for (std::size_t i = 0; i < in1.size(); ++i) {
        p1_m.col(static_cast<int>(i)) = in1[i];
        p2_m.col(static_cast<int>(i)) = in2[i];
    }
    hornN(p1_m, p2_m, s, R, t);
}

// ---------------------------------------------------------------------------

Sim3Solver::Result Sim3Solver::solve() {
    const std::size_t n = pts1_.size();
    if (n < 3) return {};

    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    Result best;
    best.inlier_mask.assign(n, false);

    for (int iter = 0; iter < p_.max_iterations; ++iter) {
        // Draw 3 distinct indices.
        std::size_t idx[3];
        idx[0] = dist(rng);
        do { idx[1] = dist(rng); } while (idx[1] == idx[0]);
        do { idx[2] = dist(rng); } while (idx[2] == idx[0] || idx[2] == idx[1]);

        Eigen::MatrixXd p1_m(3, 3), p2_m(3, 3);
        for (int k = 0; k < 3; ++k) {
            p1_m.col(k) = pts1_[idx[k]];
            p2_m.col(k) = pts2_[idx[k]];
        }

        double s; Eigen::Matrix3d R; Eigen::Vector3d t;
        if (!hornN(p1_m, p2_m, s, R, t)) continue;
        if (s <= 0.0 || s > 100.0) continue;  // degenerate scale

        std::vector<bool> mask;
        const int n_in = count_inliers(s, R, t, mask);
        if (n_in > best.n_inliers) {
            best.n_inliers    = n_in;
            best.scale        = s;
            best.R            = R;
            best.t            = t;
            best.inlier_mask  = mask;
        }
    }

    if (best.n_inliers < p_.min_inliers) return {};

    // Refine over the full inlier set.
    refine(best.scale, best.R, best.t, best.inlier_mask);
    // Re-count after refinement.
    best.n_inliers = count_inliers(best.scale, best.R, best.t,
                                    best.inlier_mask);

    if (best.n_inliers < p_.min_inliers) return {};

    best.found = true;
    return best;
}

}  // namespace sslam
