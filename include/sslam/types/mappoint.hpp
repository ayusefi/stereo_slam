#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace sslam {

class KeyFrame;  // forward-declaration — see keyframe.hpp
class Map;       // forward-declaration — see map.hpp

/// A persistent 3-D landmark in the map.
///
/// Owned by Map via shared_ptr. KeyFrames that observe this point hold a
/// shared_ptr to it as well (co-ownership). Back-references to KeyFrames
/// are raw non-owning pointers; Map is the canonical owner of both.
///
/// Thread-safety:
///   pos_mutex_  — guards pos_w_, normal_, min_distance_, max_distance_
///   obs_mutex_  — guards observations_, descriptor_
///   bad_        — std::atomic (lock-free lifecycle flag)
class MapPoint {
   public:
    using Ptr    = std::shared_ptr<MapPoint>;
    /// Raw non-owning KF pointer → feature index in that KF.
    using ObsMap = std::unordered_map<KeyFrame*, int>;

    /// @param id      Unique identifier (assigned by Tracking / Map).
    /// @param pos_w   Initial world-frame position (metres).
    /// @param ref_kf  Non-owning pointer to the KeyFrame that created this MP.
    MapPoint(uint64_t id, const Eigen::Vector3d& pos_w, KeyFrame* ref_kf);

    uint64_t id() const { return id_; }
    uint64_t created_kf_id() const { return created_kf_id_; }

    // --- World position (guarded by pos_mutex_) --------------------------

    Eigen::Vector3d get_world_pos() const;
    void            set_world_pos(const Eigen::Vector3d& pos_w);

    // --- Observations (guarded by obs_mutex_) ----------------------------

    /// Add or update: this MP is visible at feat_idx in kf.
    /// kf is a raw non-owning pointer — see CODING_STYLE.md §Memory.
    void add_observation(KeyFrame* kf, int feat_idx);

    /// Remove the observation for kf (e.g. after outlier rejection).
    void remove_observation(KeyFrame* kf);

    /// Return the feat_idx in kf, or -1 if kf does not observe this MP.
    int get_feat_idx(KeyFrame* kf) const;

    /// Snapshot of the full observations map.
    ObsMap get_observations() const;

    int n_observations() const;

    // --- Representative descriptor (guarded by obs_mutex_) ---------------

    /// ORB descriptor with minimum median Hamming distance to all others.
    /// Returns an empty cv::Mat until compute_descriptor() is called.
    cv::Mat get_descriptor() const;

    /// Recompute the representative descriptor from current observations.
    /// Call after adding or removing observations.
    void compute_descriptor();

    // --- Scale-invariance range (guarded by pos_mutex_) ------------------

    float min_distance() const;
    float max_distance() const;

    /// Recompute mean viewing normal and depth range from current observations.
    /// Uses ORBExtractor defaults (scale 1.2, 8 levels).
    /// TODO: pass ORBExtractor::Params instead of hardcoding.
    void update_normal_and_depth();

    /// Mean viewing direction (unit vector).  Zero until first update.
    Eigen::Vector3d mean_normal() const;

    // --- Lifecycle -------------------------------------------------------

    bool is_bad() const { return bad_.load(std::memory_order_relaxed); }

    /// Mark this MapPoint as bad. Erases all observations from the
    /// observing KeyFrames and removes the MP from the owning Map (if set).
    /// Idempotent — safe to call from multiple threads.
    void set_bad();

    /// Redirect all observations of this MapPoint to `other`, then mark
    /// this MP as bad.  ORB-SLAM2 MapPoint::Replace equivalent.
    /// `other` must be non-null, non-bad, and different from this.
    void replace(const MapPoint::Ptr& other);

    /// Return the MP this one was replaced with, or nullptr.
    MapPoint* get_replaced() const;

    /// Set the owning Map back-pointer. Called by Map::add_mappoint().
    /// Non-owning; the Map outlives every MP it owns.
    void set_map(Map* m) { map_ = m; }

    // --- Tracking visibility counters (for MP culling) -------------------
    /// Increment when the MP is projected into a frame (visible candidate).
    void inc_visible(int n = 1) { n_visible_.fetch_add(n, std::memory_order_relaxed); }
    /// Increment when the MP is actually matched in a frame.
    void inc_found(int n = 1)   { n_found_.fetch_add(n,   std::memory_order_relaxed); }
    int n_visible() const { return n_visible_.load(std::memory_order_relaxed); }
    int n_found()   const { return n_found_.load(std::memory_order_relaxed); }

   private:
    const uint64_t  id_;
    uint64_t        created_kf_id_{0};
    KeyFrame*       ref_kf_;   ///< Non-owning reference KF — see CODING_STYLE.md §Memory.
    Map*            map_{nullptr};   ///< Non-owning owner Map (set via set_map).
    MapPoint*       replaced_with_{nullptr};  ///< Set by replace(); guarded by obs_mutex_.

    // --- Guarded by pos_mutex_ -------------------------------------------
    mutable std::mutex pos_mutex_;
    Eigen::Vector3d    pos_w_;
    Eigen::Vector3d    normal_{Eigen::Vector3d::Zero()};
    float              min_distance_{0.0f};
    float              max_distance_{0.0f};

    // --- Guarded by obs_mutex_ -------------------------------------------
    mutable std::mutex obs_mutex_;
    ObsMap             observations_;
    cv::Mat            descriptor_;

    std::atomic<bool> bad_{false};
    std::atomic<int>  n_visible_{1};  ///< Times projected into a tracking frame
    std::atomic<int>  n_found_{1};    ///< Times actually matched (≥1 at creation)
};

}  // namespace sslam
