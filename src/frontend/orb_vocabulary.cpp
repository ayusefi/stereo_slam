#include "sslam/frontend/orb_vocabulary.hpp"

#include <DBoW2/FORB.h>

#include <zlib.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace sslam {

// ---------------------------------------------------------------------------

void ORBVocabulary::load(const std::string& path) {
    const bool is_gz  = path.size() >= 3 && path.substr(path.size() - 3) == ".gz";
    const bool is_txt = (path.size() >= 4 && path.substr(path.size() - 4) == ".txt")
                     || is_gz;

    if (is_txt) {
        // Decompress if needed, then parse the ORB-SLAM2 custom text format.
        std::string content;
        if (is_gz) {
            gzFile gz = gzopen(path.c_str(), "rb");
            if (!gz)
                throw std::runtime_error("ORBVocabulary: cannot open: " + path);
            char buf[65536];
            int  n;
            while ((n = gzread(gz, buf, sizeof(buf))) > 0)
                content.append(buf, static_cast<std::size_t>(n));
            if (n < 0) {
                gzclose(gz);
                throw std::runtime_error("ORBVocabulary: gzip read error: " + path);
            }
            gzclose(gz);
        } else {
            // Plain text — let the stream parser read directly from a file stream.
            // Read the whole file into a string so load_text_stream can work.
            FILE* f = std::fopen(path.c_str(), "rb");
            if (!f)
                throw std::runtime_error("ORBVocabulary: cannot open: " + path);
            std::fseek(f, 0, SEEK_END);
            const long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            content.resize(static_cast<std::size_t>(sz));
            std::fread(content.data(), 1, static_cast<std::size_t>(sz), f);
            std::fclose(f);
        }
        load_text_stream(content);
    } else {
        // Standard DBoW2 cv::FileStorage (YAML) format.
        OrbVocabulary::load(path);
        if (empty())
            throw std::runtime_error("ORBVocabulary: load failed (empty vocab): " + path);
    }
}

// ---------------------------------------------------------------------------
// Parse the ORB-SLAM2/3 custom text vocabulary format.
//
// Format:
//   Line 1:  k L scoring_type weighting_type
//   One line per non-root node:
//     parent_id is_leaf d[0]…d[31] weight
//
// Nodes are appended in BFS order; the root (id=0) is implicit.
// is_leaf==1 → word (leaf), is_leaf==0 → internal node.
// ---------------------------------------------------------------------------
void ORBVocabulary::load_text_stream(const std::string& content) {
    const char* p   = content.c_str();
    const char* end = p + content.size();

    // Fast helpers that operate on the raw buffer directly — avoids
    // constructing one std::istringstream per line (~1M allocations).
    auto skip_ws   = [&]() { while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) ++p; };
    auto skip_line = [&]() { while (p < end && *p != '\n') ++p; if (p < end) ++p; };
    auto read_int  = [&](int& val) -> bool {
        skip_ws();
        char* q = nullptr;
        val = static_cast<int>(std::strtol(p, &q, 10));
        if (!q || q == p) return false;
        p = q; return true;
    };
    auto read_uint8 = [&](uint8_t& val) -> bool {
        skip_ws();
        char* q = nullptr;
        val = static_cast<uint8_t>(std::strtol(p, &q, 10));
        if (!q || q == p) return false;
        p = q; return true;
    };
    auto read_double = [&](double& val) -> bool {
        skip_ws();
        char* q = nullptr;
        val = std::strtod(p, &q);
        if (!q || q == p) return false;
        p = q; return true;
    };

    // --- Header ---
    int k{0}, L{0}, n_scoring{0}, n_weighting{0};
    if (!read_int(k) || !read_int(L) || !read_int(n_scoring) || !read_int(n_weighting)
        || k <= 0 || L <= 0)
        throw std::runtime_error("ORBVocabulary: invalid header in vocabulary file");
    skip_line();

    m_k         = k;
    m_L         = L;
    m_scoring   = static_cast<DBoW2::ScoringType>(n_scoring);
    m_weighting = static_cast<DBoW2::WeightingType>(n_weighting);
    createScoringObject();

    m_nodes.clear();
    m_words.clear();

    const int expected_nodes =
        static_cast<int>((std::pow(k, L + 1) - 1.0) / (k - 1));
    m_nodes.reserve(static_cast<std::size_t>(expected_nodes));
    m_words.reserve(static_cast<std::size_t>(std::pow(k, L)));

    // Root node (id = 0).
    m_nodes.emplace_back(0);

    while (p < end) {
        skip_ws();
        if (p >= end) break;
        if (*p == '\n') { ++p; continue; }

        const int nid = static_cast<int>(m_nodes.size());
        m_nodes.emplace_back(nid);
        auto& node = m_nodes.back();

        int pid{0}, is_leaf{0};
        if (!read_int(pid) || !read_int(is_leaf)) { skip_line(); continue; }

        node.parent = pid;
        m_nodes[static_cast<std::size_t>(pid)].children.push_back(nid);

        // Descriptor: 32 bytes, each encoded as a decimal integer.
        cv::Mat desc(1, DBoW2::FORB::L, CV_8U);
        for (int i = 0; i < DBoW2::FORB::L; ++i) {
            uint8_t byte{0};
            read_uint8(byte);
            desc.at<uint8_t>(0, i) = byte;
        }
        node.descriptor = desc;

        double weight{0.0};
        read_double(weight);
        node.weight = weight;

        if (is_leaf) {
            const int wid = static_cast<int>(m_words.size());
            m_words.push_back(&m_nodes.back());
            m_nodes.back().word_id = wid;
        }

        skip_line();
    }

    if (m_words.empty())
        throw std::runtime_error("ORBVocabulary: no words parsed — wrong format?");
}

}  // namespace sslam
