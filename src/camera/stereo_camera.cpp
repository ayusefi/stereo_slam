#include "sslam/camera/stereo_camera.hpp"

#include <stdexcept>

namespace sslam {

Eigen::Vector3d StereoCamera::backproject(double u, double v, double disparity) const {
    if (disparity <= 0.0) {
        throw std::invalid_argument("StereoCamera::backproject: disparity must be > 0");
    }
    const double Z = fx * baseline / disparity;
    const double X = (u - cx) * Z / fx;
    const double Y = (v - cy) * Z / fy;
    return {X, Y, Z};
}

}  // namespace sslam
