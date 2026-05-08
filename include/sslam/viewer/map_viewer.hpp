#pragma once

#include "sslam/types/map.hpp"

#include <Eigen/Core>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace sslam {

/// Real-time 3-D map viewer powered by Pangolin.
///
/// Runs its own thread at ~30 Hz.  Reads Map and the live tracking pose
/// without writing any SLAM state.
///
/// Lifecycle:
///   auto viewer = std::make_shared<MapViewer>(map);
///   viewer->start();
///   // per frame: viewer->set_current_pose(T_cw);
///   viewer->shutdown();
class MapViewer {
   public:
    using Ptr = std::shared_ptr<MapViewer>;

    /// @param map  Live map.  MapViewer holds a read-only shared_ptr.
    explicit MapViewer(std::shared_ptr<const Map> map);
    ~MapViewer();

    /// Spawn the Pangolin window and viewer thread.
    void start();

    /// Signal the viewer to stop and join its thread.
    void shutdown();

    /// Block until the user closes the Pangolin window (or shutdown() is
    /// called from another thread).  Call this after finishing all frames
    /// to keep the map visible for examination.
    void wait_until_closed();

    /// Update the live camera frustum shown in the viewer.
    /// Thread-safe; call after every process_frame().
    /// @param T_cw  World-to-camera 4×4 transform.
    void set_current_pose(const Eigen::Matrix4d& T_cw);

   private:
    void run();  ///< Pangolin event loop; runs on thread_.

    std::shared_ptr<const Map> map_;
    Eigen::Matrix4d            current_T_cw_{Eigen::Matrix4d::Identity()};
    mutable std::mutex         pose_mutex_;
    std::atomic<bool>          stop_{false};
    std::thread                thread_;
};

}  // namespace sslam
