#pragma once

#include "sslam/frontend/orb_vocabulary.hpp"
#include "sslam/loop/keyframe_database.hpp"
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

    /// @param map      Shared map.
    /// @param lm       LocalMapping — paused during map correction.
    /// @param vocab    ORB vocabulary (non-owning).
    /// @param db       KeyFrameDatabase (non-owning).
    LoopClosing(Map::Ptr map,
                LocalMapping::Ptr lm,
                const ORBVocabulary* vocab,
                KeyFrameDatabase* db);
    ~LoopClosing();

    void start();     ///< Spawn the loop-closing thread.
    void shutdown();  ///< Signal stop and join.

    /// Called from the LocalMapping thread after a KF has been fully
    /// processed (BoW computed, BA done).  Non-blocking.
    void enqueue_keyframe(KeyFrame::Ptr kf);

    /// True when no KF is being processed.
    bool is_idle() const { return !processing_.load(); }

    /// Number of loop closures successfully executed so far.
    int loop_count() const { return loop_count_.load(); }

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

    std::unique_ptr<PlaceRecognizer> recognizer_;
    FullBA::Ptr                      full_ba_;  ///< background full BA after each correction

    // Work queue — LocalMapping thread pushes, LoopClosing thread pops.
    std::deque<KeyFrame::Ptr> queue_;
    mutable std::mutex        queue_mutex_;
    std::condition_variable   queue_cv_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> processing_{false};
    std::atomic<int>  loop_count_{0};
    std::thread       thread_;
};

}  // namespace sslam
