#include "sslam/system.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace sslam {

void save_trajectory_kitti(const std::string&                  path,
                           const std::vector<Eigen::Matrix4d>& T_cw_vec) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("save_trajectory_kitti: cannot open '" +
                                 path + "' for writing");
    }

    f << std::fixed << std::setprecision(9);

    for (const auto& T_cw : T_cw_vec) {
        // T_wc = SE(3) inverse of T_cw
        const Eigen::Matrix3d R_wc = T_cw.topLeftCorner<3, 3>().transpose();
        const Eigen::Vector3d t_wc = -R_wc * T_cw.topRightCorner<3, 1>();

        // Row-major 3×4 output: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz
        f << R_wc(0,0) << ' ' << R_wc(0,1) << ' ' << R_wc(0,2) << ' ' << t_wc(0) << ' '
          << R_wc(1,0) << ' ' << R_wc(1,1) << ' ' << R_wc(1,2) << ' ' << t_wc(1) << ' '
          << R_wc(2,0) << ' ' << R_wc(2,1) << ' ' << R_wc(2,2) << ' ' << t_wc(2) << '\n';
    }
}

}  // namespace sslam
