# sslam — Stereo Visual SLAM

A from-scratch stereo visual SLAM system in C++17, built for learning and benchmarking. Architecture follows ORB-SLAM3 with three concurrent threads: **Tracking**, **Local Mapping**, and **Loop Closing**.

## Build

```bash
sudo apt install libeigen3-dev libopencv-dev libsuitesparse-dev \
                 libgoogle-glog-dev libgflags-dev libpangolin-dev \
                 libfmt-dev libspdlog-dev libgtest-dev
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Run

```bash
./build/apps/kitti_stereo /path/to/kitti/sequences/00 configs/kitti.yaml
```

## Tech Stack

- **C++17**, CMake
- Eigen 3.4, Sophus — geometry & Lie groups
- OpenCV 4 — feature extraction (ORB), stereo matching
- g2o — bundle adjustment & pose graph optimization
- DBoW2 — bag-of-words place recognition
- spdlog, GTest, Pangolin

## Results — KITTI Odometry

| Seq | Frames | ATE (aligned) | Loop closures |
|-----|--------|---------------|---------------|
| 00  | 4541   | 2.44 m        | 3             |
| 02  | 4661   | 7.58 m        | 1             |
| 07  | 1101   | 2.47 m        | 0             |
| 08  | 4071   | 5.17 m        | 0             |

> Latest benchmark run from `bench/49dca96`. ATE = Absolute Trajectory Error after Umeyama alignment.

### Combined Top-Down Comparison

![Combined trajectory comparison](results/traj_00_02_07_08.png)

## Architecture

```
stereo frames
      │
      ▼
Tracking thread (30 Hz)  ──▶ pose
      │ keyframes
      ▼
Local Mapping (~5 Hz)    ──▶ Map (shared, mutex-protected)
      │ keyframes              ▲
      ▼                        │
Loop Closing (~1 Hz)  ─────────┘
```

## License

TBD.
