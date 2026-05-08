#pragma once

#include "sslam/types/keyframe.hpp"
#include "sslam/types/mappoint.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sslam {

/// The persistent map: owns KeyFrames and MapPoints.
///
/// All containers are protected by mutex_.  Callers that need a consistent
/// snapshot across multiple calls must hold map.mutex_ for the duration.
///
/// Lock ordering: Map::mutex_ → KeyFrame::obs_mutex_ → MapPoint::pos_mutex_
class Map {
   public:
    using Ptr = std::shared_ptr<Map>;

    Map() = default;

    void add_keyframe(KeyFrame::Ptr kf);
    void add_mappoint(MapPoint::Ptr mp);

    std::vector<KeyFrame::Ptr> get_all_keyframes() const;
    std::vector<MapPoint::Ptr> get_all_mappoints() const;

    std::size_t keyframe_count() const;
    std::size_t mappoint_count() const;

    /// Return KeyFrames sharing >= min_shared MapPoint observations with kf,
    /// sorted descending by shared count.  min_shared = 15 per Phase 2 spec.
    std::vector<KeyFrame::Ptr> local_map_around(const KeyFrame* kf,
                                                int min_shared = 15) const;

    /// mutex_ protects keyframes_ and mappoints_.
    /// Exposed for callers that need multi-step atomic access.
    mutable std::mutex mutex_;

   private:
    std::unordered_map<uint64_t, KeyFrame::Ptr> keyframes_;
    std::unordered_map<uint64_t, MapPoint::Ptr> mappoints_;
};

}  // namespace sslam
