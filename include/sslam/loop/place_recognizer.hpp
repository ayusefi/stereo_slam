#pragma once

#include "sslam/loop/keyframe_database.hpp"
#include "sslam/types/keyframe.hpp"

#include <vector>

namespace sslam {

/// Façade that wraps KeyFrameDatabase and enforces temporal consistency.
///
/// A candidate must appear in ≥ 3 consecutive calls to query() to be
/// returned as a confirmed loop candidate.  This eliminates spurious
/// single-frame matches that arise from perceptual aliasing.
///
/// The consecutive-positive window is reset whenever the query KF
/// changes (i.e. on every call the "window" is updated with the new
/// candidates).
class PlaceRecognizer {
   public:
    /// @param db    Inverted-index database (caller owns).
    /// @param min_score  Absolute BoW score lower bound (passed to db.query).
    explicit PlaceRecognizer(KeyFrameDatabase& db, double min_score = 0.04);

    /// Query for loop candidates using temporal consistency.
    ///
    /// @param q  Query KeyFrame (must have BoW computed).
    /// @return   Confirmed loop candidates (seen in ≥ 3 consecutive calls)
    ///           or empty if temporal consistency has not been reached.
    std::vector<const KeyFrame*> query(const KeyFrame* q);

   private:
    KeyFrameDatabase& db_;
    double            min_score_;

    // Temporal consistency: count how many consecutive times each candidate
    // has appeared.  Reset when the candidate disappears.
    std::vector<std::pair<const KeyFrame*, int>> consecutive_;
};

}  // namespace sslam
