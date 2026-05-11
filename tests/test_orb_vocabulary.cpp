/// test_orb_vocabulary.cpp
///
/// Acceptance criteria:
///   - Vocab loads in < 2 s (wall time)
///   - transform() on 100 random 256-bit descriptors yields a non-empty
///     BowVector

#include "sslam/frontend/orb_vocabulary.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstring>
#include <random>

namespace {
constexpr const char* kVocabPath =
    "thirdparty/vocab/ORBvoc.txt";

// Timing limits (ms) that define the acceptance bar.
constexpr double kMaxLoadMs    = 2000.0;  // vocab cold load
constexpr double kMaxTransformMs =  5.0;  // single KF descriptor transform

static bool vocab_present() {
    // Try to open the file; skip test gracefully if not downloaded.
    if (FILE* f = std::fopen(kVocabPath, "rb")) { std::fclose(f); return true; }
    return false;
}

TEST(ORBVocabulary, LoadAndTransform) {
    if (!vocab_present()) {
        GTEST_SKIP() << "Vocabulary file not found at " << kVocabPath
                     << ". Run scripts/download_vocab.sh first.";
    }

    sslam::ORBVocabulary vocab;
    const auto t0 = std::chrono::steady_clock::now();
    vocab.load(kVocabPath);
    const auto t1 = std::chrono::steady_clock::now();
    ASSERT_FALSE(vocab.empty()) << "vocab.load() yielded empty vocabulary";

    const double load_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(load_ms, kMaxLoadMs) << "Vocab load took " << load_ms << " ms";
    EXPECT_FALSE(vocab.empty());

    // Build 100 random 32-byte ORB descriptors (fixed seed for repeatability)
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::vector<cv::Mat> descs;
    descs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        cv::Mat d(1, 32, CV_8U);
        for (int j = 0; j < 32; ++j) d.at<uint8_t>(0, j) = static_cast<uint8_t>(byte_dist(rng));
        descs.push_back(d);
    }

    DBoW2::BowVector    bow;
    DBoW2::FeatureVector fv;
    vocab.transform(descs, bow, fv, 4);

    EXPECT_FALSE(bow.empty()) << "BowVector should not be empty after transform";

    // Acceptance 4.2: compute_bow on one KF's descriptors must be < 5 ms.
    const auto cb_t0 = std::chrono::steady_clock::now();
    DBoW2::BowVector    bow2;
    DBoW2::FeatureVector fv2;
    vocab.transform(descs, bow2, fv2, 4);
    const auto cb_t1 = std::chrono::steady_clock::now();
    const double bow_ms =
        std::chrono::duration<double, std::milli>(cb_t1 - cb_t0).count();
    EXPECT_LT(bow_ms, kMaxTransformMs)
        << "compute_bow took " << bow_ms << " ms (limit " << kMaxTransformMs << " ms)";
}

}  // namespace
