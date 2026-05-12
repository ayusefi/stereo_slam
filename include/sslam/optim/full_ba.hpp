#pragma once

#include "sslam/types/map.hpp"

#include <atomic>
#include <future>
#include <memory>

namespace sslam {

/// One-shot background worker that runs a full BA over the entire map.
///
/// Spawned by LoopClosing after each pose-graph correction.  If a new
/// loop fires before the previous run finishes, the in-flight run is
/// cancelled and a new one replaces it.
///
/// Cancellation is cooperative: the optimiser checks cancel_ between
/// outer iterations and returns early if set.
class FullBA {
   public:
    using Ptr = std::shared_ptr<FullBA>;

    explicit FullBA(Map::Ptr map);

    /// Launch a full BA in the background.  Cancels any running job first.
    void trigger();

    /// Wait for the current job (if any) to finish.
    void wait();

   private:
    void run();

    Map::Ptr           map_;
    std::atomic<bool>  cancel_{false};
    std::future<void>  future_;
    mutable std::mutex mutex_;  ///< guards future_
};

}  // namespace sslam
