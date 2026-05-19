#pragma once

#include "sslam/frontend/orb_vocabulary.hpp"
#include "sslam/loop/keyframe_database.hpp"
#include "sslam/loop/loop_diagnostics.hpp"
#include "sslam/loop/place_recognizer.hpp"
#include "sslam/mapping/local_mapping.hpp"
#include "sslam/optim/full_ba.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace sslam {

/// Background thread that detects loop closures and corrects the map.
///
/// Pipeline per incoming KeyFrame (enqueued by LocalMapping):
///   1. Wait for ≥ 3 consecutive candidates from PlaceRecognizer.
///   2. Compute Sim3 between query KF and best candidate via RANSAC.
///   3. Fuse duplicate MapPoints across the loop.
///   4. Run Essential-Graph pose-graph optimisation with the loop edge.
///   5. Propagate corrections to all KF poses and MP positions.
///   6. Trigger a background full BA.
///
/// Thread safety:
///   enqueue_keyframe() is the only method called from another thread
///   (the LocalMapping thread).  All map mutations happen on this thread,
///   with LocalMapping paused (request_stop/resume) during the critical
///   section.
class LoopClosing {
   public:
    using Ptr = std::shared_ptr<LoopClosing>;

    /// Tunable loop-closing parameters.  All values have defaults that match
    /// the previous hardcoded behaviour, so constructing with Params{} is
    /// backward-compatible.
    struct Params {
        /// Minimum BoW similarity score to consider a candidate.
        double   min_bow_score{0.04};
        /// Minimum BoW descriptor matches needed to attempt Sim3.
        int      min_bow_matches{20};
        /// Minimum 3-D correspondences for Sim3 RANSAC.
        int      min_correspondences{20};
        /// Minimum Sim3 RANSAC inliers (coarse gate).
        int      min_ransac_inliers{20};
        /// Minimum inliers after SearchByProjection + final Sim3 refinement.
        int      min_fused_matches{30};
        /// Maximum loop candidates to evaluate per query KeyFrame.
        int      max_candidates_per_kf{3};
        /// Minimum Sim3 inlier ratio (inliers / total correspondences).
        double   min_sim3_inlier_ratio{0.15};
        /// Maximum Sim3 reprojection RMSE [metres] for sanity check.
        double   max_sim3_rmse_m{50.0};
        /// Reject PGO previews that create an implausibly large adjacent-KF
        /// jump in the optimized trajectory.
        double   max_pgo_adjacent_step_m{12.0};
        /// Minimum KeyFrames between consecutive loop corrections.
        /// Prevents rapid re-triggering while the map is still being updated.
        uint64_t cooldown_kfs{20};
    };

    /// @param map      Shared map.
    /// @param lm       LocalMapping — paused during map correction.
    /// @param vocab    ORB vocabulary (non-owning).
    /// @param db       KeyFrameDatabase (non-owning).
    /// @param params   Tunable parameters (optional; uses defaults if omitted).
    LoopClosing(Map::Ptr map,
                LocalMapping::Ptr lm,
                const ORBVocabulary* vocab,
                KeyFrameDatabase* db);
    LoopClosing(Map::Ptr map,
                LocalMapping::Ptr lm,
                const ORBVocabulary* vocab,
                KeyFrameDatabase* db,
                const Params& params);
    ~LoopClosing();

    void start();     ///< Spawn the loop-closing thread.
    void shutdown();  ///< Signal stop and join.

    /// Block until the loop queue is empty and no candidate is being checked.
    void wait_until_idle();

    /// Called from the LocalMapping thread after a KF has been fully
    /// processed (BoW computed, BA done).  Non-blocking.
    void enqueue_keyframe(KeyFrame::Ptr kf);

    /// True when no KF is being processed.
    bool is_idle() const { return !processing_.load(); }

    /// Number of loop closures successfully executed so far.
    int loop_count() const { return loop_count_.load(); }

    /// Set an optional JSONL logger for loop attempt diagnostics.
    /// Non-owning; the logger must outlive this LoopClosing instance.
    void set_loop_logger(LoopLogger* logger) { logger_ = logger; }

   private:
    void run();

    /// Try to detect and close a loop for the given query KF.
    /// Returns true if a loop was detected and corrected.
    bool try_close_loop(KeyFrame* q);

    // --- Data members --------------------------------------------------------
    Map::Ptr            map_;
    LocalMapping::Ptr   local_mapping_;
    const ORBVocabulary* vocab_;   // non-owning
    KeyFrameDatabase*   db_;       // non-owning
    Params              params_;

    std::unique_ptr<PlaceRecognizer> recognizer_;
    FullBA::Ptr                      full_ba_;  ///< background full BA after each correction
    LoopLogger*                      logger_{nullptr};  ///< optional diagnostics (non-owning)

    // Work queue — LocalMapping thread pushes, LoopClosing thread pops.
    std::deque<KeyFrame::Ptr> queue_;
    mutable std::mutex        queue_mutex_;
    std::condition_variable   queue_cv_;
    std::condition_variable   idle_cv_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> processing_{false};
    std::atomic<int>  loop_count_{0};
    uint64_t          last_loop_kf_id_{0};
    std::thread       thread_;
};

}  // namespace sslam
