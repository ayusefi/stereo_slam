// Frame-to-frame feature matching via map-point projection.
//
// Each stereo-triangulated point from the previous frame is projected into
// the current frame using a predicted SE(3) pose. The search uses a 2-D grid
// index so each projection query touches only the cells that overlap the
// search circle, keeping per-frame cost O(N · k) where k is the average
// number of keypoints per cell (small).
//
// Reference: ORB-SLAM2 ORBmatcher.cc::SearchByProjection (frame-to-frame).

#include "sslam/frontend/feature_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sslam {

namespace {

// Grid cell size (pixels). 64×64 gives ~300 cells for a 1241×376 KITTI image,
// with ~7 keypoints/cell on average at 2000 features.
constexpr int kCellSize = 64;

struct KeypointGrid {
    int cols, rows;
    std::vector<std::vector<int>> cells;

    KeypointGrid(int img_w, int img_h)
        : cols((img_w + kCellSize - 1) / kCellSize),
          rows((img_h + kCellSize - 1) / kCellSize),
          cells(static_cast<std::size_t>(cols) * rows) {}

    void insert(int idx, float u, float v) {
        const int c = std::clamp(static_cast<int>(u) / kCellSize, 0, cols - 1);
        const int r = std::clamp(static_cast<int>(v) / kCellSize, 0, rows - 1);
        cells[static_cast<std::size_t>(r) * cols + c].push_back(idx);
    }

    /// Append to `out` the indices of all keypoints in cells that overlap
    /// the axis-aligned bounding box [u-rad, u+rad] × [v-rad, v+rad].
    void query(float u, float v, float rad, std::vector<int>& out) const {
        const int c0 = std::clamp(static_cast<int>(u - rad) / kCellSize, 0, cols - 1);
        const int c1 = std::clamp(static_cast<int>(u + rad) / kCellSize, 0, cols - 1);
        const int r0 = std::clamp(static_cast<int>(v - rad) / kCellSize, 0, rows - 1);
        const int r1 = std::clamp(static_cast<int>(v + rad) / kCellSize, 0, rows - 1);
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                for (int idx : cells[static_cast<std::size_t>(r) * cols + c]) {
                    out.push_back(idx);
                }
            }
        }
    }
};

inline int hamming256(const uint8_t* a, const uint8_t* b) {
    int dist = 0;
    for (int i = 0; i < 32; ++i)
        dist += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    return dist;
}

}  // namespace

FeatureMatcher::FeatureMatcher(std::shared_ptr<const StereoCamera> cam)
    : FeatureMatcher(std::move(cam), Params{}) {}

FeatureMatcher::FeatureMatcher(std::shared_ptr<const StereoCamera> cam,
                               const Params& p)
    : cam_(std::move(cam)), params_(p) {}

std::vector<std::pair<int, int>> FeatureMatcher::match_by_projection(
    const Frame& prev,
    const Frame& curr,
    const Eigen::Matrix4d& T_curr_cw,
    float radius_scale) const {

    const int W = cam_->width;
    const int H = cam_->height;
    const double fx = cam_->fx, fy = cam_->fy;
    const double cx = cam_->cx, cy = cam_->cy;

    const int n_prev = static_cast<int>(prev.keypoints_left.size());
    const int n_curr = static_cast<int>(curr.keypoints_left.size());
    if (n_prev == 0 || n_curr == 0) return {};

    // --- Build grid index over curr keypoints ---------------------------
    KeypointGrid grid(W, H);
    for (int j = 0; j < n_curr; ++j) {
        grid.insert(j, curr.keypoints_left[j].pt.x, curr.keypoints_left[j].pt.y);
    }

    // T_prev_wc: prev camera frame → world.
    // For SE(3): if T_cw = [R | t], then T_wc = [R^T | -R^T*t].
    const Eigen::Matrix3d R_prev_T  = prev.T_cw.topLeftCorner<3, 3>().transpose();
    const Eigen::Vector3d t_prev_wc = -R_prev_T * prev.T_cw.topRightCorner<3, 1>();
    const Eigen::Matrix3d R_curr    = T_curr_cw.topLeftCorner<3, 3>();
    const Eigen::Vector3d t_curr    = T_curr_cw.topRightCorner<3, 1>();

    std::vector<std::pair<int, int>> matches;
    matches.reserve(static_cast<std::size_t>(n_prev));

    std::vector<int> candidates;
    candidates.reserve(64);

    for (int i = 0; i < n_prev; ++i) {
        if (prev.depth[i] <= 0.0f) continue;

        const float u_l = prev.keypoints_left[i].pt.x;
        const float v_l = prev.keypoints_left[i].pt.y;
        const float d   = prev.depth[i];
        const int   oct = prev.keypoints_left[i].octave;

        // --- Unproject to prev camera frame -----------------------------
        const Eigen::Vector3d p_c_prev(
            (static_cast<double>(u_l) - cx) * d / fx,
            (static_cast<double>(v_l) - cy) * d / fy,
            static_cast<double>(d));

        // --- Lift to world ----------------------------------------------
        // T_wc * p_c_prev
        const Eigen::Vector3d p_w = R_prev_T * p_c_prev + t_prev_wc;

        // --- Project into curr camera frame -----------------------------
        const Eigen::Vector3d p_c_curr = R_curr * p_w + t_curr;

        if (p_c_curr.z() <= 0.0) continue;

        const double u_pred = fx * p_c_curr.x() / p_c_curr.z() + cx;
        const double v_pred = fy * p_c_curr.y() / p_c_curr.z() + cy;

        if (u_pred < 0.0 || u_pred >= W || v_pred < 0.0 || v_pred >= H) continue;

        // --- Grid search ------------------------------------------------
        const float radius = params_.search_radius * radius_scale *
                             std::pow(params_.scale_factor, static_cast<float>(oct));

        candidates.clear();
        grid.query(static_cast<float>(u_pred), static_cast<float>(v_pred),
                   radius, candidates);
        if (candidates.empty()) continue;

        // --- Descriptor match with Lowe ratio ---------------------------
        int best_dist  = std::numeric_limits<int>::max();
        int best2_dist = std::numeric_limits<int>::max();
        int best_j     = -1;

        const uint8_t* dp = prev.descriptors_left.ptr<uint8_t>(i);
        for (int j : candidates) {
            const int dist = hamming256(dp, curr.descriptors_left.ptr<uint8_t>(j));
            if (dist < best_dist) {
                best2_dist = best_dist;
                best_dist  = dist;
                best_j     = j;
            } else if (dist < best2_dist) {
                best2_dist = dist;
            }
        }

        if (best_j < 0 || best_dist >= params_.hamming_threshold) continue;

        if (best2_dist < std::numeric_limits<int>::max()) {
            if (static_cast<float>(best_dist) >
                params_.lowe_ratio * static_cast<float>(best2_dist)) {
                continue;
            }
        }

        matches.emplace_back(i, best_j);
    }

    return matches;
}

}  // namespace sslam
