#include "sslam/loop/place_recognizer.hpp"

#include <algorithm>

namespace sslam {

PlaceRecognizer::PlaceRecognizer(KeyFrameDatabase& db, double min_score)
    : db_(db), min_score_(min_score)
{}

// ---------------------------------------------------------------------------

std::vector<const KeyFrame*> PlaceRecognizer::query(const KeyFrame* q) {
    const auto raw = db_.query_loop_candidates(q, min_score_);

    // Update consecutive-positive counters.
    // For each candidate currently in `raw`:
    //   - If it was already tracked, increment its counter.
    //   - If it's new, add it with counter = 1.
    // Entries that do NOT appear in `raw` are dropped (streak broken).
    std::vector<std::pair<const KeyFrame*, int>> next;
    next.reserve(raw.size());

    for (const KeyFrame* kf : raw) {
        int count = 1;
        for (auto& [prev_kf, prev_count] : consecutive_) {
            if (prev_kf == kf) { count = prev_count + 1; break; }
        }
        next.emplace_back(kf, count);
    }
    consecutive_ = std::move(next);

    // Return only candidates that have reached 3 consecutive positives.
    std::vector<const KeyFrame*> confirmed;
    for (const auto& [kf, count] : consecutive_)
        if (count >= 3) confirmed.push_back(kf);

    return confirmed;
}

}  // namespace sslam
