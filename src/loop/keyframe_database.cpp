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

    // ------------------------------------------------------------------
    // Step 1: count shared words between q and every non-covisible KF.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // Step 2: maxCommonWords pre-filter (ORB-SLAM2: keep >= 0.8 * max).
    // ------------------------------------------------------------------
    int max_common = 0;
    for (const auto& [kf, cnt] : word_count)
        if (cnt > max_common) max_common = cnt;
    const int min_common = static_cast<int>(0.8 * max_common);

    // ------------------------------------------------------------------
    // Step 3: score surviving candidates; apply absolute lower bound.
    // ------------------------------------------------------------------
    std::unordered_map<const KeyFrame*, double> scores;
    scores.reserve(word_count.size());
    for (const auto& [kf, cnt] : word_count) {
        if (cnt < min_common) continue;
        if (!kf->bow_computed()) continue;
        const double s = vocab_.score(q_bow, kf->bow());
        if (s >= min_score) scores[kf] = s;
    }
    if (scores.empty()) return {};

    // ------------------------------------------------------------------
    // Step 4: covisibility-group score accumulation (ORB-SLAM2 style).
    //   group(kf) = {kf} ∪ top-10 covisible KFs of kf that are also candidates
    //   accum_score(kf) = sum of scores of group members
    // ------------------------------------------------------------------
    constexpr int kGroupSize = 10;
    double best_accum = 0.0;
    std::unordered_map<const KeyFrame*, double> accum_scores;
    accum_scores.reserve(scores.size());

    for (const auto& [kf, s] : scores) {
        double accum = s;
        int count = 0;
        for (const KeyFrame* nb :
             const_cast<KeyFrame*>(kf)->get_covisibility_keyframes(0)) {
            if (count++ >= kGroupSize) break;
            auto it = scores.find(nb);
            if (it != scores.end()) accum += it->second;
        }
        accum_scores[kf] = accum;
        if (accum > best_accum) best_accum = accum;
    }

    const double retain_thresh = 0.75 * best_accum;

    // ------------------------------------------------------------------
    // Step 5: for each qualifying group keep the member with highest
    //         individual score (avoid duplicates via a visited set).
    // ------------------------------------------------------------------
    std::unordered_set<const KeyFrame*> visited;
    std::vector<const KeyFrame*> results;

    // Sort by accumulated score descending so we pick the best group first.
    std::vector<std::pair<double, const KeyFrame*>> sorted_accum;
    sorted_accum.reserve(accum_scores.size());
    for (const auto& [kf, a] : accum_scores)
        if (a >= retain_thresh) sorted_accum.push_back({a, kf});
    // Break score ties by KeyFrame id so the candidate ordering is
    // independent of unordered_map iteration order (ASLR-dependent pointer
    // hashing).  Without this, two groups with equal accumulated scores can
    // swap positions across runs, changing which loop candidate fires first
    // and making loop closures non-deterministic.
    std::sort(sorted_accum.begin(), sorted_accum.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second->id() < b.second->id();
              });

    for (const auto& [accum, kf] : sorted_accum) {
        if (visited.count(kf)) continue;

        // Collect the group members and mark all as visited.
        const KeyFrame* best_kf = kf;
        double          best_s  = scores.at(kf);
        visited.insert(kf);

        int count = 0;
        for (const KeyFrame* nb :
             const_cast<KeyFrame*>(kf)->get_covisibility_keyframes(0)) {
            if (count++ >= kGroupSize) break;
            if (!scores.count(nb)) continue;
            visited.insert(nb);
            if (scores.at(nb) > best_s) {
                best_s  = scores.at(nb);
                best_kf = nb;
            }
        }
        results.push_back(best_kf);
    }

    return results;
}

// ---------------------------------------------------------------------------
// Relocalization candidate query.
// ---------------------------------------------------------------------------

std::vector<const KeyFrame*> KeyFrameDatabase::query_relocalization_candidates(
    const DBoW2::BowVector& q_bow, double min_score) const
{
    if (q_bow.empty()) return {};

    // Step 1: count shared words with all KFs (no covisibility exclusion).
    std::unordered_map<const KeyFrame*, int> word_count;
    {
        std::scoped_lock lk(mutex_);
        for (const auto& [word_id, weight] : q_bow) {
            for (const KeyFrame* kf : index_[static_cast<std::size_t>(word_id)]) {
                ++word_count[kf];
            }
        }
    }
    if (word_count.empty()) return {};

    // Step 2: maxCommonWords pre-filter.
    int max_common = 0;
    for (const auto& [kf, cnt] : word_count)
        if (cnt > max_common) max_common = cnt;
    const int min_common = static_cast<int>(0.8 * max_common);

    // Step 3: score surviving candidates.
    std::vector<std::pair<double, const KeyFrame*>> scored;
    for (const auto& [kf, cnt] : word_count) {
        if (cnt < min_common || !kf->bow_computed() || kf->is_bad()) continue;
        const double s = vocab_.score(q_bow, kf->bow());
        if (s >= min_score) scored.push_back({s, kf});
    }
    if (scored.empty()) return {};

    // Return sorted by score descending.
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<const KeyFrame*> results;
    results.reserve(scored.size());
    for (const auto& [s, kf] : scored) results.push_back(kf);
    return results;
}

}  // namespace sslam
