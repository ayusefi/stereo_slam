#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace sslam {

/// One record per loop candidate that reached Sim3 RANSAC.
/// Filled progressively in LoopClosing::try_close_loop and flushed at each
/// accept/reject decision point.
struct LoopAttemptStats {
    uint64_t query_kf_id{0};
    uint64_t candidate_kf_id{0};
    double   bow_score{0.0};
    int      bow_matches{0};
    int      correspondences_3d{0};
    int      sim3_inliers{0};
    double   sim3_inlier_ratio{0.0};
    double   sim3_scale{1.0};
    double   sim3_rmse_m{0.0};
    int      graph_vertices{0};
    int      graph_edges{0};
    int      graph_components{0};
    double   max_pose_correction_m{0.0};
    double   max_pose_correction_deg{0.0};
    double   max_adjacent_step_m{0.0};
    bool     accepted{false};
    std::string reject_reason;
};

/// Thread-safe JSONL writer for loop attempt diagnostics.
///
/// Each call to record() appends one JSON object followed by a newline.
/// A default-constructed (null) logger silently discards records.
class LoopLogger {
   public:
    LoopLogger() = default;

    /// Open @p path for appending.  The file is created if it does not exist.
    explicit LoopLogger(const std::string& path);
    ~LoopLogger();

    /// Append one record as a JSON object.  Thread-safe.
    void record(const LoopAttemptStats& s);

    bool is_open() const { return file_.is_open(); }

   private:
    std::ofstream file_;
    std::mutex    mutex_;
};

}  // namespace sslam
