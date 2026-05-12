#include "sslam/loop/keyframe_database.hpp"

#include <DBoW2/BowVector.h>

#include <algorithm>
#include <iostream>
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

    // Step 2: compute minimum BoW similarity among the top-10 covisible KFs.
    // This sets a data-adaptive floor: any candidate that looks at least as
    // similar as the weakest of the 10 nearest neighbours passes the filter.
    constexpr int kTopCovis = 10;
    const auto top_covis = q->get_covisibility_keyframes(0);  // sorted by weight desc
    double min_covis_score = min_score;
    int covis_counted = 0;
    for (const KeyFrame* nb : top_covis) {
        if (nb == q || !nb->bow_computed()) continue;
        const double s = vocab_.score(q_bow, nb->bow());
        if (covis_counted == 0 || s < min_covis_score) min_covis_score = s;
        if (++covis_counted >= kTopCovis) break;
    }
    const double threshold = 0.7 * min_covis_score;

    // Step 3: score each candidate and apply threshold.
    std::vector<const KeyFrame*> results;
    results.reserve(word_count.size());
    double best_cand_score = 0.0;
    for (const auto& [kf, _] : word_count) {
        if (!kf->bow_computed()) continue;
        const double s = vocab_.score(q_bow, kf->bow());
        if (s >= threshold) results.push_back(kf);
        if (s > best_cand_score) best_cand_score = s;
    }

    // Diagnostic: print when candidates are found in the pre-filter.
    if (!word_count.empty()) {
        static int diag_count = 0;
        if (++diag_count % 20 == 0 || !results.empty())
            std::cerr << "[DB] candidates=" << word_count.size()
                      << " threshold=" << threshold
                      << " best_score=" << best_cand_score
                      << " passing=" << results.size() << "\n";
    }

    return results;
}

}  // namespace sslam
