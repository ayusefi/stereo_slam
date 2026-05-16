#pragma once

#include "sslam/camera/stereo_camera.hpp"
#include "sslam/frontend/orb_vocabulary.hpp"
#include "sslam/optim/ba.hpp"
#include "sslam/types/keyframe.hpp"
#include "sslam/types/map.hpp"
#include "sslam/types/mappoint.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace sslam {

class LoopClosing;      // forward declaration — avoids circular include
class KeyFrameDatabase; // forward declaration

/// Background thread that refines the map after each KeyFrame insertion.
///
/// Pipeline per KeyFrame:
///   1. Triangulate new MapPoints with covisible KeyFrames.
///   2. Cull bad MapPoints (low observation ratio or too few observers).
///   3. Local BA — jointly optimise local KF poses and MP positions.
///   4. Cull redundant KeyFrames (≥ 90% of MPs seen by ≥ 3 other KFs).
///
/// Thread safety:
///   - enqueue_keyframe() is the only method called from the Tracking thread.
///   - All Map mutations happen on the LocalMapping thread.
///   - request_stop() / is_idle() are safe to call from any thread.
class LocalMapping {
   public:
    using Ptr = std::shared_ptr<LocalMapping>;

    struct Params {
        /// Number of strongest covisible KeyFrames used for triangulation.
        int max_triangulation_neighbours{5};
        /// Number of strongest covisible KeyFrames checked for redundancy.
        int max_cull_neighbours{20};
        /// Number of newer KeyFrames allowed before under-observed MPs are culled.
        int mappoint_grace_kfs{3};
        /// Minimum observations required after the grace period.
        int min_mappoint_observations{3};
        /// Bundle adjustment parameters (e.g. local window size).
        ba::Params ba{};
    };

    struct BaStats {
        uint64_t runs{0};
        double total_ms{0.0};
        double max_ms{0.0};

        double avg_ms() const {
            return runs > 0 ? total_ms / static_cast<double>(runs) : 0.0;
        }
    };

    /// @param map  Shared map; LocalMapping holds a shared_ptr.
    /// @param cam  Shared camera calibration.
    LocalMapping(Map::Ptr map, std::shared_ptr<const StereoCamera> cam);
    ~LocalMapping();

    /// Spawn the processing thread.
    void start();

    /// Signal the thread to finish and join it.
    void shutdown();

    /// Push a newly-inserted KeyFrame onto the work queue.
    /// Thread-safe; called from the Tracking thread.
    void enqueue_keyframe(KeyFrame::Ptr kf);

    /// True when the queue is empty and no KF is being processed.
    bool is_idle() const;

    /// Block until the input queue is empty and the current KF is done.
    void wait_until_idle();

    /// Set mapping and BA parameters (call before start()).
    void set_params(const Params& p) { params_ = p; }

    /// Set vocabulary for BoW computation (call before start()).
    /// If not set, BoW computation is skipped.
    void set_vocabulary(const ORBVocabulary* vocab) { vocab_ = vocab; }

    /// Set the LoopClosing consumer (call before start()).
    /// If set, each fully-processed KF is forwarded after BA completes.
    void set_loop_closing(LoopClosing* lc) { loop_closing_ = lc; }

    /// Set the KeyFrameDatabase so culled KFs are erased from it.
    void set_keyframe_database(KeyFrameDatabase* db) { kf_db_ = db; }

    /// Snapshot of local BA timing statistics.
    BaStats ba_stats() const;

    /// Request that Local Mapping pauses after completing the current KF.
    /// Used by Loop Closing before running pose-graph correction.
    void request_stop();
    void resume();
    bool is_stopped() const;
    /// Block until LocalMapping has entered the paused state.
    /// Must be called after request_stop().
    void wait_until_stopped();

   private:
    void run();

    // --- Sub-steps ---------------------------------------------------------
    void triangulate_new_mappoints(KeyFrame* kf);
    void cull_mappoints(KeyFrame* current_kf);
    void cull_keyframes(KeyFrame* kf);
    void mark_processing_done();
    void record_ba_time(double ms);

    // --- Data members ------------------------------------------------------
    Map::Ptr                              map_;
    std::shared_ptr<const StereoCamera>   cam_;
    const ORBVocabulary*                  vocab_{nullptr};        // non-owning
    LoopClosing*                          loop_closing_{nullptr}; // non-owning
    KeyFrameDatabase*                     kf_db_{nullptr};        // non-owning
    Params                                params_;

    // Work queue (Tracking thread pushes, LocalMapping thread pops).
    std::deque<KeyFrame::Ptr>  queue_;
    mutable std::mutex         queue_mutex_;
    std::condition_variable    queue_cv_;
    std::condition_variable    idle_cv_;

    mutable std::mutex stats_mutex_;
    BaStats           ba_stats_;

    std::atomic<bool>  stop_{false};       // shutdown flag
    std::atomic<bool>  stop_requested_{false};  // pause flag
    std::atomic<bool>  stopped_{false};    // currently paused
    std::atomic<bool>  processing_{false}; // KF being processed now

    std::thread thread_;
};

}  // namespace sslam
