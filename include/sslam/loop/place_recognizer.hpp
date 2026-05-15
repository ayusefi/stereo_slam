#pragma once

#include "sslam/loop/keyframe_database.hpp"
#include "sslam/types/keyframe.hpp"

#include <vector>

namespace sslam {

/// Façade that wraps KeyFrameDatabase and enforces temporal consistency.
///
/// After the DB returns the best-per-covisibility-group candidates,
/// PlaceRecognizer applies ORB-SLAM2-style group temporal consistency:
///   1. For each candidate kf, build its covisibility group:
///      {kf} ∪ top-K covisible KFs of kf.
///   2. A group is "consistent" with a previous group if they share at
///      least one member.
///   3. Each group accumulates a consecutive consistency count; when the
///      count reaches kConsistencyTh (3), the best KF in the group is
///      returned as a confirmed loop candidate.
///   4. Groups that are not consistent with any previous group get count=1
///      and are carried forward for the next call.
class PlaceRecognizer {
   public:
    /// @param db         Inverted-index database (caller owns).
    /// @param min_score  Absolute BoW score lower bound (passed to db.query).
    explicit PlaceRecognizer(KeyFrameDatabase& db, double min_score = 0.04);

    /// Query for loop candidates using covisibility-group temporal consistency.
    ///
    /// @param q  Query KeyFrame (must have BoW computed).
    /// @return   Confirmed loop candidates (group consistent for >= 3 calls)
    ///           or empty if consistency has not been reached.
    std::vector<const KeyFrame*> query(const KeyFrame* q);

   private:
    KeyFrameDatabase& db_;
    double            min_score_;

    static constexpr int kConsistencyTh = 3;
    static constexpr int kGroupCovisK   = 10;

    // Each entry: (covisibility group members, consistency count).
    using Group = std::vector<const KeyFrame*>;
    std::vector<std::pair<Group, int>> prev_groups_;
};

}  // namespace sslam
