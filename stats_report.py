import json
import numpy as np
import os

def load_poses(filename):
    if not os.path.exists(filename):
        return None
    poses = []
    with open(filename, 'r') as f:
        for line in f:
            data = list(map(float, line.split()))
            if len(data) == 12:
                # KITTI ground truth: 3x4 matrix
                mat = np.array(data).reshape(3, 4)
                poses.append(mat[:, 3])
            elif len(data) == 7:
                # Estimate: format typically tx ty tz qx qy qz qw
                poses.append(data[:3])
    return np.array(poses)

def compute_stats(gt, est):
    n = min(len(gt), len(est))
    gt = gt[:n]
    est = est[:n]
    errors = np.linalg.norm(gt - est, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    max_idx = np.argmax(errors)
    max_err = errors[max_idx]
    final_err = errors[-1]
    return rmse, mean_err, median_err, max_idx, max_err, final_err

# Loop log analysis
loop_file = '/tmp/sslam_full_final_loops.jsonl'
total_records = 0
accepted = 0
rejections = {}
if os.path.exists(loop_file):
    with open(loop_file, 'r') as f:
        for line in f:
            total_records += 1
            try:
                data = json.loads(line)
                if data.get('accepted') is True:
                     accepted += 1
                else:
                     reason = data.get('reject_reason', 'unknown')
                     rejections[reason] = rejections.get(reason, 0) + 1
            except:
                pass

# Trajectory analysis
gt_file = 'data/kitti/dataset/poses/00.txt'
est_file = '/tmp/sslam_full_final.txt'

gt_poses = load_poses(gt_file)
est_poses = load_poses(est_file)

print(f"Loop Log Analysis ({loop_file}):")
print(f"  Total records: {total_records}")
print(f"  Accepted: {accepted}")
print(f"  Rejections: {rejections}")

if gt_poses is not None and est_poses is not None:
    rmse, mean, median, max_idx, max_err, final = compute_stats(gt_poses, est_poses)
    print(f"\nTrajectory Stats vs {gt_file}:")
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  Mean Error: {mean:.4f} m")
    print(f"  Median Error: {median:.4f} m")
    print(f"  Max Error: {max_err:.4f} m at frame {max_idx}")
    print(f"  Final Error: {final:.4f} m")
else:
    print("\nCould not compute trajectory stats.")
