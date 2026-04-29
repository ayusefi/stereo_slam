// Faithful re-implementation of ORB-SLAM2's ORBextractor: image pyramid +
// per-cell FAST detection (with fallback threshold) + quadtree distribution
// for even spatial coverage. Descriptor computation is delegated to
// cv::ORB::compute (it is the standard BRIEF-256 with steered orientation).
//
// Reference: ORB-SLAM2 source `ORBextractor.cc`.

#include "sslam/frontend/orb_extractor.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <list>
#include <utility>

namespace sslam {

namespace {

// Border kept clear of features so the BRIEF-256 31×31 sample pattern stays
// inside the image even with sub-pixel orientation rotation.
constexpr int kEdgeThreshold = 19;

// Each FAST detection cell is W×W pixels (with a 3-px overlap so a corner
// near the edge of one cell is still detected by an adjacent cell).
constexpr int kCellSize = 30;

struct QuadNode {
    cv::Point2i UL, UR, BL, BR;
    std::vector<cv::KeyPoint> keys;
    bool no_more{false};

    void divide(QuadNode& n1, QuadNode& n2, QuadNode& n3, QuadNode& n4) const {
        const int halfX = static_cast<int>(std::ceil((UR.x - UL.x) / 2.0f));
        const int halfY = static_cast<int>(std::ceil((BR.y - UL.y) / 2.0f));
        n1.UL = UL;                              n1.UR = {UL.x + halfX, UL.y};
        n1.BL = {UL.x, UL.y + halfY};            n1.BR = {UL.x + halfX, UL.y + halfY};
        n2.UL = n1.UR;                           n2.UR = UR;
        n2.BL = n1.BR;                           n2.BR = {UR.x, UL.y + halfY};
        n3.UL = n1.BL;                           n3.UR = n1.BR;
        n3.BL = BL;                              n3.BR = {n1.BR.x, BL.y};
        n4.UL = n3.UR;                           n4.UR = n2.BR;
        n4.BL = n3.BR;                           n4.BR = BR;
        for (const auto& kp : keys) {
            if (kp.pt.x < n1.UR.x) {
                if (kp.pt.y < n1.BR.y) n1.keys.push_back(kp);
                else                   n3.keys.push_back(kp);
            } else {
                if (kp.pt.y < n1.BR.y) n2.keys.push_back(kp);
                else                   n4.keys.push_back(kp);
            }
        }
    }
};

}  // namespace

ORBExtractor::ORBExtractor(const Params& p) : params_(p) {
    pyramid_.resize(p.num_levels);
    scale_factors_.resize(p.num_levels);
    inv_scale_factors_.resize(p.num_levels);
    scale_factors_[0] = 1.0f;
    for (int i = 1; i < p.num_levels; ++i) {
        scale_factors_[i] = scale_factors_[i - 1] * p.scale_factor;
    }
    for (int i = 0; i < p.num_levels; ++i) {
        inv_scale_factors_[i] = 1.0f / scale_factors_[i];
    }

    // Distribute the feature budget across levels: per-level area scales by
    // (1/scale_factor)^(2i) but ORB-SLAM uses a linear factor (1/scale_factor)^i,
    // which empirically gives more features at coarser scales than pure-area.
    features_per_level_.resize(p.num_levels);
    const float factor = 1.0f / p.scale_factor;
    const float n_desired_l0 =
        p.num_features * (1.0f - factor) /
        (1.0f - std::pow(factor, static_cast<float>(p.num_levels)));
    int sum = 0;
    for (int i = 0; i < p.num_levels - 1; ++i) {
        features_per_level_[i] =
            static_cast<int>(std::round(n_desired_l0 * std::pow(factor, i)));
        sum += features_per_level_[i];
    }
    features_per_level_[p.num_levels - 1] = std::max(p.num_features - sum, 0);
}

void ORBExtractor::build_pyramid(const cv::Mat& image) {
    pyramid_[0] = image;
    for (int i = 1; i < params_.num_levels; ++i) {
        const cv::Size sz(
            static_cast<int>(std::round(pyramid_[i - 1].cols / params_.scale_factor)),
            static_cast<int>(std::round(pyramid_[i - 1].rows / params_.scale_factor)));
        cv::resize(pyramid_[i - 1], pyramid_[i], sz, 0, 0, cv::INTER_LINEAR);
    }
}

std::vector<cv::KeyPoint> ORBExtractor::distribute_quadtree(
    const std::vector<cv::KeyPoint>& candidates,
    int min_x, int max_x, int min_y, int max_y, int target_n) {

    // Initial root node grid: split horizontally so cells are roughly square.
    const int n_init = std::max(
        1, static_cast<int>(std::round(static_cast<float>(max_x - min_x) /
                                       std::max(1, max_y - min_y))));
    const float h_x = static_cast<float>(max_x - min_x) / n_init;

    std::list<QuadNode> nodes;
    std::vector<std::list<QuadNode>::iterator> roots;
    roots.reserve(n_init);
    for (int i = 0; i < n_init; ++i) {
        QuadNode n;
        n.UL = {static_cast<int>(h_x * i),       0};
        n.UR = {static_cast<int>(h_x * (i + 1)), 0};
        n.BL = {n.UL.x, max_y - min_y};
        n.BR = {n.UR.x, max_y - min_y};
        nodes.push_back(std::move(n));
        roots.push_back(std::prev(nodes.end()));
    }
    for (const auto& kp : candidates) {
        const int idx = std::min<int>(static_cast<int>(kp.pt.x / h_x), n_init - 1);
        roots[idx]->keys.push_back(kp);
    }
    for (auto it = nodes.begin(); it != nodes.end();) {
        if      (it->keys.empty())     it = nodes.erase(it);
        else if (it->keys.size() == 1) { it->no_more = true; ++it; }
        else                            ++it;
    }

    while (true) {
        const std::size_t prev_size = nodes.size();

        // Snapshot the iterators we want to expand this round.
        std::vector<std::list<QuadNode>::iterator> to_expand;
        to_expand.reserve(nodes.size());
        for (auto it = nodes.begin(); it != nodes.end(); ++it) {
            if (!it->no_more) to_expand.push_back(it);
        }
        if (to_expand.empty()) break;

        // Sort largest-first so we hit the target cleanly.
        std::sort(to_expand.begin(), to_expand.end(),
                  [](const auto& a, const auto& b) {
                      return a->keys.size() > b->keys.size();
                  });

        for (auto it : to_expand) {
            QuadNode n1, n2, n3, n4;
            it->divide(n1, n2, n3, n4);
            for (QuadNode* nn : {&n1, &n2, &n3, &n4}) {
                if (nn->keys.empty()) continue;
                if (nn->keys.size() == 1) nn->no_more = true;
                nodes.push_back(std::move(*nn));
            }
            nodes.erase(it);
            if (static_cast<int>(nodes.size()) >= target_n) break;
        }

        if (static_cast<int>(nodes.size()) >= target_n ||
            nodes.size() == prev_size) {
            break;
        }
    }

    // Pick the highest-response keypoint from each surviving cell.
    std::vector<cv::KeyPoint> out;
    out.reserve(nodes.size());
    for (auto& n : nodes) {
        if (n.keys.empty()) continue;
        const cv::KeyPoint* best = &n.keys.front();
        for (const auto& k : n.keys) {
            if (k.response > best->response) best = &k;
        }
        out.push_back(*best);
    }
    return out;
}

void ORBExtractor::compute_keypoints(
    std::vector<std::vector<cv::KeyPoint>>& all_kps) {

    all_kps.assign(params_.num_levels, {});

    for (int level = 0; level < params_.num_levels; ++level) {
        const cv::Mat& img = pyramid_[level];
        const int min_x = kEdgeThreshold - 3;
        const int max_x = img.cols - kEdgeThreshold + 3;
        const int min_y = kEdgeThreshold - 3;
        const int max_y = img.rows - kEdgeThreshold + 3;
        if (max_x - min_x < kCellSize || max_y - min_y < kCellSize) continue;

        std::vector<cv::KeyPoint> candidates;
        candidates.reserve(static_cast<std::size_t>(features_per_level_[level]) * 10);

        const int n_cols = std::max(1, (max_x - min_x) / kCellSize);
        const int n_rows = std::max(1, (max_y - min_y) / kCellSize);
        const int cell_w =
            static_cast<int>(std::ceil(static_cast<float>(max_x - min_x) / n_cols));
        const int cell_h =
            static_cast<int>(std::ceil(static_cast<float>(max_y - min_y) / n_rows));

        for (int i = 0; i < n_rows; ++i) {
            const int y0 = min_y + i * cell_h;
            const int y1 = std::min(y0 + cell_h + 6, max_y);
            if (y0 + 3 >= max_y) continue;
            for (int j = 0; j < n_cols; ++j) {
                const int x0 = min_x + j * cell_w;
                const int x1 = std::min(x0 + cell_w + 6, max_x);
                if (x0 + 3 >= max_x) continue;

                std::vector<cv::KeyPoint> cell_kps;
                cv::FAST(img.rowRange(y0, y1).colRange(x0, x1),
                         cell_kps, params_.ini_fast_threshold, true);
                if (cell_kps.empty()) {
                    cv::FAST(img.rowRange(y0, y1).colRange(x0, x1),
                             cell_kps, params_.min_fast_threshold, true);
                }
                for (auto& kp : cell_kps) {
                    kp.pt.x += j * cell_w;
                    kp.pt.y += i * cell_h;
                    candidates.push_back(kp);
                }
            }
        }

        auto distributed = distribute_quadtree(
            candidates, min_x, max_x, min_y, max_y, features_per_level_[level]);

        // Translate from "ROI-local" back to "level image" coordinates.
        for (auto& kp : distributed) {
            kp.pt.x += min_x;
            kp.pt.y += min_y;
            kp.octave = level;
            kp.size   = 31.0f * scale_factors_[level];
        }
        all_kps[level] = std::move(distributed);
    }
}

void ORBExtractor::detect(const cv::Mat& image,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors) {
    CV_Assert(image.type() == CV_8UC1);

    build_pyramid(image);

    std::vector<std::vector<cv::KeyPoint>> all_kps;
    compute_keypoints(all_kps);

    // Compute descriptors per level on the level image (so the BRIEF pattern
    // is sampled at the correct scale), then rescale keypoint coordinates
    // back to the original (level-0) image.
    auto orb = cv::ORB::create();
    std::vector<cv::Mat> level_descs(params_.num_levels);
    for (int level = 0; level < params_.num_levels; ++level) {
        if (all_kps[level].empty()) continue;
        orb->compute(pyramid_[level], all_kps[level], level_descs[level]);
        const float scale = scale_factors_[level];
        for (auto& kp : all_kps[level]) {
            kp.pt.x *= scale;
            kp.pt.y *= scale;
        }
    }

    int total = 0;
    for (const auto& v : all_kps) total += static_cast<int>(v.size());

    keypoints.clear();
    keypoints.reserve(total);
    descriptors.create(total, 32, CV_8U);
    int offset = 0;
    for (int level = 0; level < params_.num_levels; ++level) {
        if (all_kps[level].empty()) continue;
        const int n = static_cast<int>(all_kps[level].size());
        level_descs[level].copyTo(descriptors.rowRange(offset, offset + n));
        offset += n;
        keypoints.insert(keypoints.end(),
                         all_kps[level].begin(), all_kps[level].end());
    }
}

}  // namespace sslam
