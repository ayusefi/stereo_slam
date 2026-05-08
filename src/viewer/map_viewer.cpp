#include "sslam/viewer/map_viewer.hpp"

#include "sslam/types/keyframe.hpp"
#include "sslam/types/mappoint.hpp"

#include <pangolin/pangolin.h>

#include <algorithm>
#include <chrono>
#include <thread>

namespace sslam {

namespace {

/// Draw a wire-frame camera frustum at world-frame pose T_wc.
/// sz is scale in metres; r/g/b are the colour (0–1 range).
void draw_frustum(const Eigen::Matrix4d& T_wc, float sz,
                  float r, float g, float b, float lw = 1.0f) {
    // Frustum corners in camera frame (camera looks along +z, y down).
    const float w  = sz * 0.5f;
    const float h  = sz * 0.375f;  // 4:3 aspect
    const float d  = sz;

    const Eigen::Matrix3f R = T_wc.block<3, 3>(0, 0).cast<float>();
    const Eigen::Vector3f t = T_wc.block<3, 1>(0, 3).cast<float>();

    // Transform corner from camera to world frame.
    auto wf = [&](float cx, float cy, float cz) {
        const Eigen::Vector3f v = R * Eigen::Vector3f(cx, cy, cz) + t;
        return v;
    };

    const Eigen::Vector3f tl = wf(-w,  h, d), tr = wf( w,  h, d),
                          br = wf( w, -h, d), bl = wf(-w, -h, d);

    glLineWidth(lw);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    auto emit = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b_) {
        glVertex3f(a.x(), a.y(), a.z());
        glVertex3f(b_.x(), b_.y(), b_.z());
    };
    // Edges from origin to corners.
    emit(t, tl); emit(t, tr); emit(t, br); emit(t, bl);
    // Front face rectangle.
    emit(tl, tr); emit(tr, br); emit(br, bl); emit(bl, tl);
    glEnd();
}

/// Invert a T_cw (world→camera) to get T_wc (camera frame in world).
inline Eigen::Matrix4d invert_se3(const Eigen::Matrix4d& T_cw) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d R = T_cw.block<3, 3>(0, 0);
    T.block<3, 3>(0, 0) = R.transpose();
    T.block<3, 1>(0, 3) = -R.transpose() * T_cw.block<3, 1>(0, 3);
    return T;
}

}  // namespace

// ---------------------------------------------------------------------------

MapViewer::MapViewer(std::shared_ptr<const Map> map) : map_(std::move(map)) {}

MapViewer::~MapViewer() { shutdown(); }

void MapViewer::start() {
    thread_ = std::thread(&MapViewer::run, this);
}

void MapViewer::shutdown() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

void MapViewer::wait_until_closed() {
    // Block until the Pangolin window is closed by the user (ShouldQuit()
    // returns true in run()) or until shutdown() is called concurrently.
    if (thread_.joinable()) thread_.join();
}

void MapViewer::set_current_pose(const Eigen::Matrix4d& T_cw) {
    std::scoped_lock lk(pose_mutex_);
    current_T_cw_ = T_cw;
}

void MapViewer::run() {
    constexpr int kW = 1280, kH = 720;
    pangolin::CreateWindowAndBind("sslam :: Map Viewer", kW, kH);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Perspective projection; initial bird's-eye look-from-behind position.
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(kW, kH, 500, 500, kW / 2, kH / 2, 0.1, 10000),
        pangolin::ModelViewLookAt(0.0, -10.0, -20.0,
                                  0.0,   0.0,   0.0,
                                  0.0,  -1.0,   0.0));

    constexpr int kPanelW = 200;
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0,
                   pangolin::Attach::Pix(kPanelW), 1.0,
                   -static_cast<double>(kW) / kH)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kPanelW));

    pangolin::Var<bool>  ui_follow   ("ui.Follow Camera",   true, true);
    pangolin::Var<float> ui_pt_size  ("ui.Point Size",      1.0f, 1.0f, 5.0f);
    pangolin::Var<bool>  ui_show_kfs ("ui.Show KeyFrames",  true, true);
    pangolin::Var<bool>  ui_show_mps ("ui.Show Map Points", true, true);
    pangolin::Var<bool>  ui_show_cov ("ui.Show Graph",      true, true);
    pangolin::Var<bool>  ui_reset    ("ui.Reset View",      false, false);

    while (!pangolin::ShouldQuit() && !stop_) {
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --------------------------------------------------------------------
        // Snapshot: call public accessors (each acquires and releases its
        // own mutex) — no map mutex held during GL calls.
        // --------------------------------------------------------------------

        struct KFSnap {
            uint64_t             id;
            Eigen::Matrix4d      T_wc;
            Eigen::Vector3d      center;
            std::vector<Eigen::Vector3d> cov_centers;
        };
        std::vector<KFSnap>          kf_snaps;
        std::vector<Eigen::Vector3d> mp_positions;

        // KeyFrames
        {
            const auto kfs = map_->get_all_keyframes();
            kf_snaps.reserve(kfs.size());
            for (const auto& kf : kfs) {
                if (!kf || kf->is_bad()) continue;
                KFSnap snap;
                snap.id     = kf->id();
                snap.T_wc   = invert_se3(kf->get_pose());
                snap.center = snap.T_wc.block<3, 1>(0, 3);
                if (ui_show_cov) {
                    for (KeyFrame* nb : kf->get_covisibility_keyframes(15)) {
                        if (nb && !nb->is_bad())
                            snap.cov_centers.push_back(
                                invert_se3(nb->get_pose()).block<3, 1>(0, 3));
                    }
                }
                kf_snaps.push_back(std::move(snap));
            }
        }

        // Sort by id so the trajectory poly-line follows insertion order,
        // not unordered_map hash order (which causes spurious jumps).
        std::sort(kf_snaps.begin(), kf_snaps.end(),
                  [](const KFSnap& a, const KFSnap& b) { return a.id < b.id; });

        // MapPoints
        if (ui_show_mps) {
            const auto mps = map_->get_all_mappoints();
            mp_positions.reserve(mps.size());
            for (const auto& mp : mps)
                if (mp && !mp->is_bad())
                    mp_positions.push_back(mp->get_world_pos());
        }

        // Live camera pose
        Eigen::Matrix4d T_cw_live;
        { std::scoped_lock lk(pose_mutex_); T_cw_live = current_T_cw_; }
        const Eigen::Matrix4d T_wc_live = invert_se3(T_cw_live);

        // --------------------------------------------------------------------
        // Follow camera / reset view
        // --------------------------------------------------------------------
        if (pangolin::Pushed(ui_reset)) {
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(0.0, -10.0, -20.0,
                                          0.0,   0.0,   0.0,
                                          0.0,  -1.0,   0.0));
        } else if (ui_follow) {
            const Eigen::Vector3d c   = T_wc_live.block<3, 1>(0, 3);
            // Camera's forward (+z) and up (-y) axes in world frame.
            const Eigen::Vector3d fwd = T_wc_live.block<3, 1>(0, 2);
            const Eigen::Vector3d up  = -T_wc_live.block<3, 1>(0, 1);
            const Eigen::Vector3d eye = c - fwd * 20.0 + up * 5.0;
            s_cam.SetModelViewMatrix(
                pangolin::ModelViewLookAt(
                    eye.x(), eye.y(), eye.z(),
                    c.x(),   c.y(),   c.z(),
                    up.x(),  up.y(),  up.z()));
        }

        d_cam.Activate(s_cam);

        // --------------------------------------------------------------------
        // Draw MapPoints (white point cloud)
        // --------------------------------------------------------------------
        if (ui_show_mps && !mp_positions.empty()) {
            glPointSize(static_cast<float>(ui_pt_size));
            glColor3f(0.85f, 0.85f, 0.85f);
            glBegin(GL_POINTS);
            for (const auto& p : mp_positions)
                glVertex3d(p.x(), p.y(), p.z());
            glEnd();
        }

        // --------------------------------------------------------------------
        // Draw trajectory (red poly-line through KF centres)
        // --------------------------------------------------------------------
        if (kf_snaps.size() > 1u) {
            glLineWidth(2.0f);
            glColor3f(0.9f, 0.1f, 0.1f);
            glBegin(GL_LINE_STRIP);
            for (const auto& s : kf_snaps)
                glVertex3d(s.center.x(), s.center.y(), s.center.z());
            glEnd();
        }

        // --------------------------------------------------------------------
        // Draw covisibility edges (grey lines between KF centres)
        // --------------------------------------------------------------------
        if (ui_show_cov && !kf_snaps.empty()) {
            glLineWidth(1.0f);
            glColor3f(0.45f, 0.45f, 0.45f);
            glBegin(GL_LINES);
            for (const auto& s : kf_snaps)
                for (const auto& nb : s.cov_centers) {
                    glVertex3d(s.center.x(), s.center.y(), s.center.z());
                    glVertex3d(nb.x(), nb.y(), nb.z());
                }
            glEnd();
        }

        // --------------------------------------------------------------------
        // Draw KeyFrame frustums (blue, 0.2 m scale)
        // --------------------------------------------------------------------
        if (ui_show_kfs)
            for (const auto& s : kf_snaps)
                draw_frustum(s.T_wc, 0.2f, 0.25f, 0.25f, 0.9f);

        // Draw live camera frustum (green, 0.4 m scale, thicker line)
        draw_frustum(T_wc_live, 0.4f, 0.1f, 0.9f, 0.1f, 2.0f);

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

}  // namespace sslam
