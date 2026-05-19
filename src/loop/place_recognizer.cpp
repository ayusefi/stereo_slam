#include "sslam/loop/place_recognizer.hpp"
#include "sslam/types/keyframe.hpp"

#include <algorithm>
#include <unordered_set>

namespace sslam {

PlaceRecognizer::PlaceRecognizer(KeyFrameDatabase& db, double min_score)
    : db_(db), min_score_(min_score)
{}

// ---------------------------------------------------------------------------

std::vector<const KeyFrame*> PlaceRecognizer::query(const KeyFrame* q)
{
    // 1. Get best-per-group candidates from the DB (B14 already applies the
    //    maxCommonWords pre-filter and group-accumulation scoring).
    const auto raw = db_.query_loop_candidates(q, min_score_);

    // 2. For each raw candidate build its covisibility group.
    std::vector<Group> cur_groups;
    cur_groups.reserve(raw.size());
    for (const KeyFrame* kf : raw) {
        Group g;
        g.push_back(kf);
        int cnt = 0;
        for (const KeyFrame* nb :
             const_cast<KeyFrame*>(kf)->get_covisibility_keyframes(0)) {
            if (cnt++ >= kGroupCovisK) break;
            g.push_back(nb);
        }
        cur_groups.push_back(std::move(g));
    }

    // 3. Match each current group against previous groups.
    //    consistency_count[i] = max prev_count of any matching prev group + 1,
    //    or 1 if no prev group shares a member.
    std::vector<int> consistency(cur_groups.size(), 1);
    std::vector<bool> prev_matched(prev_groups_.size(), false);

    for (std::size_t i = 0; i < cur_groups.size(); ++i) {
        const std::unordered_set<const KeyFrame*> cur_set(
            cur_groups[i].begin(), cur_groups[i].end());

        for (std::size_t j = 0; j < prev_groups_.size(); ++j) {
            const Group& pg = prev_groups_[j].first;
            const int    pc = prev_groups_[j].second;

            bool intersects = false;
            for (const KeyFrame* pkf : pg) {
                if (cur_set.count(pkf)) { intersects = true; break; }
            }
            if (!intersects) continue;

            prev_matched[j] = true;
            if (pc + 1 > consistency[i]) consistency[i] = pc + 1;
        }
    }

    // 4. Update prev_groups_ with the current generation.
    std::vector<std::pair<Group, int>> next_groups;
    next_groups.reserve(cur_groups.size());
    for (std::size_t i = 0; i < cur_groups.size(); ++i)
        next_groups.push_back({cur_groups[i], consistency[i]});

    // Carry forward unmatched previous groups (capped to avoid unbounded growth).
    for (std::size_t j = 0; j < prev_groups_.size(); ++j) {
        if (!prev_matched[j])
            next_groups.push_back({prev_groups_[j].first, prev_groups_[j].second});
    }
    prev_groups_ = std::move(next_groups);

    // 5. Return the first (best by DB order) member of each group whose
    //    consistency count has reached the threshold.
    std::vector<const KeyFrame*> confirmed;
    for (std::size_t i = 0; i < cur_groups.size(); ++i) {
        if (consistency[i] >= kConsistencyTh)
            confirmed.push_back(cur_groups[i][0]);  // [0] is the DB best-per-group KF
    }
    return confirmed;
}

void PlaceRecognizer::reset() {
    prev_groups_.clear();
}

}  // namespace sslam
