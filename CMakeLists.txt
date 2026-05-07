cmake_minimum_required(VERSION 3.20)
project(sslam VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# --- Dependencies (system) -----------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui calib3d features2d)
find_package(spdlog QUIET)
find_package(fmt QUIET)

# --- Vendored header-only deps -------------------------------------------
add_library(sophus INTERFACE)
target_include_directories(sophus INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Sophus
)
target_link_libraries(sophus INTERFACE Eigen3::Eigen)

# --- Vendored g2o (motion-only BA) ----------------------------------------
# Disable everything we don't need so the build stays fast.
set(G2O_BUILD_APPS         OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
set(G2O_USE_OPENGL         OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM2D_TYPES         OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM2D_ADDON_TYPES   OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_DATA_TYPES           OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SCLAM2D_TYPES        OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM3D_TYPES         ON  CACHE BOOL "" FORCE)
set(G2O_BUILD_SLAM3D_ADDON_TYPES   OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_ICP_TYPES            OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SIM3_TYPES           OFF CACHE BOOL "" FORCE)
set(G2O_BUILD_SBA_TYPES            ON  CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/g2o EXCLUDE_FROM_ALL)

# --- Library -------------------------------------------------------------
add_library(sslam
    src/camera/stereo_camera.cpp
    src/dataset/kitti_loader.cpp
    src/frontend/orb_extractor.cpp
    src/frontend/stereo_matcher.cpp
    src/frontend/feature_matcher.cpp
    src/tracking/tracking.cpp
    src/optim/ba.cpp
)

target_include_directories(sslam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sslam PUBLIC
    Eigen3::Eigen
    sophus
    ${OpenCV_LIBS}
    g2o::core
    g2o::stuff
    g2o::types_sba
    g2o::solver_dense
)

if(spdlog_FOUND)
    target_link_libraries(sslam PUBLIC spdlog::spdlog)
    target_compile_definitions(sslam PUBLIC SSLAM_HAVE_SPDLOG=1)
endif()

# --- Apps ----------------------------------------------------------------
add_subdirectory(apps)

# --- Tests ---------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
