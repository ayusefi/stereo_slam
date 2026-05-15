#pragma once

#include "sslam/frontend/orb_vocabulary.hpp"
#include "sslam/types/keyframe.hpp"

#include <DBoW2/BowVector.h>

#include <list>
#include <mutex>
#include <vector>

namespace sslam {

/// Inverted-index database of KeyFrames, indexed by DBoW2 word IDs.
///
/// Thread-safety: add/erase/query are all guarded by a single mutex.
///
/// Loop-candidate query algorithm:
///   1. For each word in the query's BoW vector, walk the posting list.
///   2. Skip KFs that share a direct covisibility edge with the query KF
///      (they are neighbouring frames, not loops).
///   3. Score every remaining candidate with vocab.score(q->bow, c->bow).
///   4. Keep candidates whose score ≥ 0.7 × (best score among the query's
///      own covisibility neighbours).  If no neighbours have BoW, fall back
///      to min_score as the absolute threshold.
class KeyFrameDatabase {
   public:
    explicit KeyFrameDatabase(const ORBVocabulary& vocab);

    /// Add a KeyFrame to the inverted index (call after compute_bow).
    void add(const KeyFrame* kf);

    /// Remove a KeyFrame from all posting lists it appears in.
    void erase(const KeyFrame* kf);

    /// Query for loop-closure candidates.
    ///
    /// @param q          Query KeyFrame (must have BoW computed).
    /// @param min_score  Absolute lower bound; candidates below this are
    ///                   always rejected even if the relative threshold is
    ///                   lower.
    /// @return Non-owning pointers to candidate KFs (never the query itself).
    std::vector<const KeyFrame*> query_loop_candidates(
        const KeyFrame* q, double min_score = 0.0) const;

    /// Query for relocalization candidates using a raw BoW vector.
    /// Unlike query_loop_candidates, does NOT skip covisible KFs.
    ///
    /// @param bow        BoW vector of the query frame (not a KeyFrame).
    /// @param min_score  Absolute lower bound on similarity score.
    /// @return Non-owning pointers to candidate KFs.
    std::vector<const KeyFrame*> query_relocalization_candidates(
        const DBoW2::BowVector& bow, double min_score = 0.0) const;

   private:
    const ORBVocabulary&                vocab_;
    // One posting list per vocabulary word.
    std::vector<std::list<const KeyFrame*>> index_;
    mutable std::mutex                  mutex_;
};

}  // namespace sslam
