// Rectified-stereo L↔R feature matcher.
//
// For each left ORB keypoint:
//   1. Collect right candidates in a row band (epipolar constraint).
//   2. Descriptor match (Hamming + Lowe ratio).
//   3. Sub-pixel refine with 11×11 SAD + parabolic fit.
//   4. Validate disparity range → write right_u / depth into Frame.
//
// Reference: ORB-SLAM2 §III-B and ORBmatcher.cc::ComputeStereoMatches.

#include "sslam/frontend/stereo_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sslam {

namespace {

/// Compute the 256-bit Hamming distance between two ORB descriptors.
/// Each descriptor row is 32 bytes (256 bits).
inline int hamming256(const uint8_t* a, const uint8_t* b) {
    int dist = 0;
    for (int i = 0; i < 32; ++i) {
        dist += __builtin_popcount(static_cast<unsigned>(a[i] ^ b[i]));
    }
    return dist;
}

}  // namespace

StereoMatcher::StereoMatcher(std::shared_ptr<const StereoCamera> cam)
    : StereoMatcher(std::move(cam), Params{}) {}

StereoMatcher::StereoMatcher(std::shared_ptr<const StereoCamera> cam,
                             const Params& p)
    : cam_(std::move(cam)), params_(p) {
    // Auto max_disparity: anything closer than 0.5 m is suspicious indoors /
    // impossible for KITTI; cap at fx·b / 0.5 m.
    if (params_.max_disparity < 0.0f) {
        params_.max_disparity =
            static_cast<float>(cam_->fx * cam_->baseline / 0.5);
    }
}

void StereoMatcher::match(Frame& frame,
                          const std::vector<cv::KeyPoint>& right_kps,
                          const cv::Mat& right_descs) {
    const int n_left = static_cast<int>(frame.keypoints_left.size());
    frame.right_u.assign(n_left, -1.0f);
    frame.depth.assign(n_left, -1.0f);

    if (right_kps.empty() || frame.keypoints_left.empty()) return;

    // --- Build row-indexed lookup for right keypoints -------------------
    // We round each right keypoint's y to the nearest integer row so we can
    // quickly enumerate candidates inside [v_min, v_max].
    const int H = frame.right.rows;
    const int W = frame.right.cols;
    std::vector<std::vector<int>> row_to_right(H);
    for (int j = 0; j < static_cast<int>(right_kps.size()); ++j) {
        const int row = static_cast<int>(std::round(right_kps[j].pt.y));
        if (row >= 0 && row < H) {
            row_to_right[row].push_back(j);
        }
    }

    const int half_w = params_.sad_win_half;  // SAD window half-size

    // --- Match each left keypoint ---------------------------------------
    for (int i = 0; i < n_left; ++i) {
        const cv::KeyPoint& kp_l = frame.keypoints_left[i];
        const float u_l = kp_l.pt.x;
        const float v_l = kp_l.pt.y;
        const int   oct = kp_l.octave;

        // Row band scales with octave so that coarser-scale features tolerate
        // more rectification error (ORB-SLAM2 uses 2 px at level 0).
        const float row_band = params_.row_tolerance * std::pow(1.2f, oct);
        const int v_min = static_cast<int>(std::floor(v_l - row_band));
        const int v_max = static_cast<int>(std::ceil(v_l  + row_band));

        // --- Descriptor search ------------------------------------------
        // NOTE: initialise best_dist to INT_MAX (not the threshold) so the
        // threshold sentinel is never mistaken for a real second candidate,
        // which would corrupt the Lowe ratio check.
        int best_dist        = std::numeric_limits<int>::max();
        int second_best_dist = std::numeric_limits<int>::max();
        int best_j           = -1;

        const uint8_t* dl = frame.descriptors_left.ptr<uint8_t>(i);

        for (int v = std::max(0, v_min); v <= std::min(H - 1, v_max); ++v) {
            for (int j : row_to_right[v]) {
                // Epipolar constraint: right x must be ≤ left x
                // (allow 1 px slack for sub-pixel keypoint positions).
                if (right_kps[j].pt.x > u_l + 1.0f) continue;

                const uint8_t* dr = right_descs.ptr<uint8_t>(j);
                const int dist    = hamming256(dl, dr);

                if (dist < best_dist) {
                    second_best_dist = best_dist;
                    best_dist        = dist;
                    best_j           = j;
                } else if (dist < second_best_dist) {
                    second_best_dist = dist;
                }
            }
        }

        // Hard threshold
        if (best_j < 0 || best_dist >= params_.hamming_threshold) continue;

        // Lowe ratio — only meaningful when a genuine second candidate exists.
        if (second_best_dist < std::numeric_limits<int>::max()) {
            if (static_cast<float>(best_dist) >
                params_.lowe_ratio * static_cast<float>(second_best_dist)) {
                continue;
            }
        }

        // --- Sub-pixel SAD refinement -----------------------------------
        // Fit a parabola through SAD(u_r-1), SAD(u_r), SAD(u_r+1) in the
        // level-0 images to get a fractional right x-coordinate.

        const int u_r_int = static_cast<int>(std::round(right_kps[best_j].pt.x));
        const int u_l_int = static_cast<int>(std::round(u_l));
        const int v_int   = static_cast<int>(std::round(v_l));

        float u_r_subpix = static_cast<float>(u_r_int);  // fallback

        const bool left_ok  = (u_l_int - half_w >= 0) && (u_l_int + half_w < W) &&
                              (v_int   - half_w >= 0) && (v_int   + half_w < H);
        const bool right_ok = (u_r_int - half_w - 1 >= 0) &&
                              (u_r_int + half_w + 1 <  W);

        if (left_ok && right_ok) {
            // Compute SAD for offsets -1, 0, +1 applied to the right window.
            float sad[3] = {0.0f, 0.0f, 0.0f};
            for (int dy = -half_w; dy <= half_w; ++dy) {
                const uint8_t* row_l = frame.left.ptr<uint8_t>(v_int + dy);
                const uint8_t* row_r = frame.right.ptr<uint8_t>(v_int + dy);
                for (int dx = -half_w; dx <= half_w; ++dx) {
                    const float pl = static_cast<float>(row_l[u_l_int + dx]);
                    for (int off = -1; off <= 1; ++off) {
                        const float pr =
                            static_cast<float>(row_r[u_r_int + off + dx]);
                        sad[off + 1] += std::abs(pl - pr);
                    }
                }
            }

            // Parabolic fit: δ = 0.5·(c₀ - c₂) / (c₀ + c₂ - 2·c₁)
            const float denom = sad[0] + sad[2] - 2.0f * sad[1];
            if (std::abs(denom) > 1e-6f) {
                const float delta = 0.5f * (sad[0] - sad[2]) / denom;
                // Only apply if the parabola is convex and peak is in range
                if (denom > 0.0f && std::abs(delta) < 1.0f) {
                    u_r_subpix = static_cast<float>(u_r_int) + delta;
                }
            }
        }

        // --- Validate disparity and store -------------------------------
        const float disp = u_l - u_r_subpix;
        if (disp < params_.min_disparity || disp > params_.max_disparity) continue;

        frame.right_u[i] = u_r_subpix;
        frame.depth[i]   = static_cast<float>(cam_->fx * cam_->baseline / disp);
    }
}

}  // namespace sslam
