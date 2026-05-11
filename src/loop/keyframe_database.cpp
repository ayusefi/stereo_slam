#include "sslam/loop/keyframe_database.hpp"

#include <DBoW2/BowVector.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace sslam {

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary& vocab)
    : vocab_(vocab),
      index_(static_cast<std::size_t>(vocab.size()))  // one list per word
{}

// ---------------------------------------------------------------------------

void KeyFrameDatabase::add(const KeyFrame* kf) {
    if (!kf || !kf->bow_computed()) return;
    std::scoped_lock lk(mutex_);
    for (const auto& [word_id, weight] : kf->bow())
        index_[static_cast<std::size_t>(word_id)].push_back(kf);
}

// ---------------------------------------------------------------------------

void KeyFrameDatabase::erase(const KeyFrame* kf) {
    if (!kf) return;
    std::scoped_lock lk(mutex_);
    for (const auto& [word_id, weight] : kf->bow()) {
        auto& lst = index_[static_cast<std::size_t>(word_id)];
        lst.remove(kf);
    }
}

// ---------------------------------------------------------------------------

std::vector<const KeyFrame*> KeyFrameDatabase::query_loop_candidates(
    const KeyFrame* q, double min_score) const
{
    if (!q || !q->bow_computed()) return {};

    const DBoW2::BowVector q_bow = q->bow();

    // Direct covisibility neighbours of q — exclude from candidates.
    std::unordered_set<const KeyFrame*> covis_set;
    for (const KeyFrame* nb : q->get_covisibility_keyframes(0))
        covis_set.insert(nb);
    covis_set.insert(q);  // also exclude self

    // Step 1: collect candidates sharing ≥ 1 word with q.
    // Track per-candidate word-overlap count for fast pre-filtering.
    std::unordered_map<const KeyFrame*, int> word_count;
    {
        std::scoped_lock lk(mutex_);
        for (const auto& [word_id, weight] : q_bow) {
            for (const KeyFrame* kf : index_[static_cast<std::size_t>(word_id)]) {
                if (covis_set.count(kf)) continue;
                ++word_count[kf];
            }
        }
    }

    if (word_count.empty()) return {};

    // Step 2: compute covisibility-neighbour BoW scores to set the
    // relative acceptance threshold.
    double best_covis_score = min_score;
    for (const KeyFrame* nb : covis_set) {
        if (nb == q || !nb->bow_computed()) continue;
        const double s = vocab_.score(q_bow, nb->bow());
        if (s > best_covis_score) best_covis_score = s;
    }
    const double threshold = 0.7 * best_covis_score;

    // Step 3: score each candidate and apply threshold.
    std::vector<const KeyFrame*> results;
    results.reserve(word_count.size());
    for (const auto& [kf, _] : word_count) {
        if (!kf->bow_computed()) continue;
        const double s = vocab_.score(q_bow, kf->bow());
        if (s >= threshold) results.push_back(kf);
    }

    return results;
}

}  // namespace sslam
