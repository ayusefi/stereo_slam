#include "sslam/loop/loop_diagnostics.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace sslam {

LoopLogger::LoopLogger(const std::string& path)
    : file_(path, std::ios::out | std::ios::trunc) {}

LoopLogger::~LoopLogger() {
    if (file_.is_open()) file_.close();
}

namespace {
/// Escape a string for embedding in JSON.
std::string json_str(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    return out;
}
}  // namespace

void LoopLogger::record(const LoopAttemptStats& s) {
    if (!file_.is_open()) return;

    const auto finite = [](double v) {
        return std::isfinite(v) ? v : 0.0;
    };

    std::ostringstream o;
    o << std::fixed << std::setprecision(6);
    o << "{"
      << "\"query_kf_id\":"        << s.query_kf_id        << ","
      << "\"candidate_kf_id\":"    << s.candidate_kf_id    << ","
    << "\"bow_score\":"          << finite(s.bow_score)          << ","
      << "\"bow_matches\":"        << s.bow_matches        << ","
      << "\"correspondences_3d\":" << s.correspondences_3d << ","
      << "\"sim3_inliers\":"       << s.sim3_inliers       << ","
    << "\"sim3_inlier_ratio\":"  << finite(s.sim3_inlier_ratio)  << ","
    << "\"sim3_scale\":"         << finite(s.sim3_scale)         << ","
    << "\"sim3_rmse_m\":"        << finite(s.sim3_rmse_m)        << ","
      << "\"graph_vertices\":"     << s.graph_vertices     << ","
      << "\"graph_edges\":"        << s.graph_edges        << ","
      << "\"graph_components\":"   << s.graph_components   << ","
    << "\"max_pose_correction_m\":"   << finite(s.max_pose_correction_m)   << ","
    << "\"max_pose_correction_deg\":" << finite(s.max_pose_correction_deg) << ","
    << "\"max_adjacent_step_m\":"     << finite(s.max_adjacent_step_m)     << ","
      << "\"accepted\":"           << (s.accepted ? "true" : "false") << ","
      << "\"reject_reason\":\""    << json_str(s.reject_reason) << "\""
      << "}\n";

    std::scoped_lock lk(mutex_);
    file_ << o.str();
    file_.flush();
}

}  // namespace sslam
