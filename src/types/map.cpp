#include "sslam/types/map.hpp"

namespace sslam {

void Map::add_keyframe(KeyFrame::Ptr kf) {
    if (!kf) return;
    kf->set_map(this);
    std::scoped_lock lk(mutex_);
    keyframes_[kf->id()] = std::move(kf);
}

void Map::remove_keyframe(uint64_t id) {
    std::scoped_lock lk(mutex_);
    keyframes_.erase(id);
}

void Map::add_mappoint(MapPoint::Ptr mp) {
    if (!mp) return;
    mp->set_map(this);
    std::scoped_lock lk(mutex_);
    mappoints_[mp->id()] = std::move(mp);
}

void Map::remove_mappoint(uint64_t id) {
    std::scoped_lock lk(mutex_);
    mappoints_.erase(id);
}

uint64_t Map::allocate_mappoint_id() {
    return next_mappoint_id_.fetch_add(1, std::memory_order_relaxed);
}

std::vector<KeyFrame::Ptr> Map::get_all_keyframes() const {
    std::scoped_lock lk(mutex_);
    std::vector<KeyFrame::Ptr> v;
    v.reserve(keyframes_.size());
    for (const auto& [id, kf] : keyframes_)
        v.push_back(kf);
    return v;
}

std::vector<MapPoint::Ptr> Map::get_all_mappoints() const {
    std::scoped_lock lk(mutex_);
    std::vector<MapPoint::Ptr> v;
    v.reserve(mappoints_.size());
    for (const auto& [id, mp] : mappoints_)
        v.push_back(mp);
    return v;
}

std::size_t Map::keyframe_count() const {
    std::scoped_lock lk(mutex_);
    return keyframes_.size();
}

std::size_t Map::mappoint_count() const {
    std::scoped_lock lk(mutex_);
    return mappoints_.size();
}

std::vector<KeyFrame::Ptr> Map::local_map_around(const KeyFrame* kf,
                                                  int min_shared) const {
    // get_covisibility_keyframes acquires kf->obs_mutex_ — held after Map::mutex_,
    // which is correct per the documented lock ordering.
    std::scoped_lock lk(mutex_);

    const std::vector<KeyFrame*> covis =
        kf->get_covisibility_keyframes(min_shared);

    std::vector<KeyFrame::Ptr> result;
    result.reserve(covis.size());
    for (KeyFrame* other : covis) {
        const auto it = keyframes_.find(other->id());
        if (it != keyframes_.end())
            result.push_back(it->second);
    }
    return result;
}

}  // namespace sslam
