#pragma once

#include <DBoW2/DBoW2.h>

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace sslam {

/// Thin wrapper around DBoW2's OrbVocabulary that adds:
///   - ORB-SLAM2/3 custom text-file format loading (the canonical ORBvoc.txt)
///   - On-the-fly gzip decompression (for .txt.gz files)
///
/// Subclasses OrbVocabulary so it can access protected members to populate
/// the tree from the custom text format without requiring a modified DBoW2.
///
/// Usage:
///   ORBVocabulary vocab;
///   vocab.load("/path/to/ORBvoc.txt.gz");   // or ORBvoc.txt
///   DBoW2::BowVector   bow;
///   DBoW2::FeatureVector fv;
///   vocab.transform(descriptors, bow, fv, 4);
class ORBVocabulary : public OrbVocabulary {
   public:
    ORBVocabulary() = default;

    /// Load from file.
    ///
    /// Dispatch:
    ///   *.txt.gz  → decompress to temp string, then load_text_stream
    ///   *.txt     → load_text_stream (ORB-SLAM2 custom format)
    ///   other     → DBoW2 native cv::FileStorage format
    ///
    /// @throws std::runtime_error on I/O or parse errors.
    void load(const std::string& path);

    bool empty() const { return m_words.empty(); }

   private:
    /// Parse the ORB-SLAM2/3 custom text vocabulary format from a string.
    /// Format:
    ///   Line 1: k L scoring_type weighting_type
    ///   Remaining lines (one per non-root node):
    ///     parent_id is_leaf d[0] d[1] … d[31] weight
    void load_text_stream(const std::string& content);
};

}  // namespace sslam
