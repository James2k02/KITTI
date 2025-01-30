import cv2
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm

class VisualOdometry():
    def __init__(self, data_dir):
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt')
        self.gt_poses = self._load_poses(data_dir + '/poses.txt')
        self.images_l = self._load_images(data_dir + '/image_0')
        self.images_r = self._load_images(data_dir + '/image_1')
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.prev_pts = None  # Initialize prev_pts as None

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ').reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images

    def track_features(self, prev_img, curr_img, prev_pts):
        if prev_pts is None or len(prev_pts) == 0:
            return np.array([]), np.array([])
        prev_pts = np.float32(prev_pts).reshape(-1, 1, 2)  # Ensure correct format
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **self.lk_params)
        valid_prev_pts = prev_pts[status.flatten() == 1]
        valid_curr_pts = curr_pts[status.flatten() == 1]
        return valid_prev_pts, valid_curr_pts

    def match_features(self, img1, img2):
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return pts1, pts2

    def estimate_pose(self, Q1, q2):
        _, rvec, tvec, inliers = cv2.solvePnPRansac(Q1, q2, self.K_l, None, flags=cv2.SOLVEPNP_EPNP)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        return T

    def get_pose(self, i):
        img1, img2 = self.images_l[i - 1], self.images_l[i]
        if i == 1:
            tp1, tp2 = self.match_features(img1, img2)
            self.prev_pts = tp2  # Initialize prev_pts with matched points
        else:
            tp1, tp2 = self.track_features(self.images_l[i - 2], img1, self.prev_pts)

        if tp1.size == 0 or tp2.size == 0:
            return np.eye(4)

        self.prev_pts = tp2  # Update previous points for tracking
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, tp1.T, tp2.T)
        Q1 /= Q1[3]
        Q1 = Q1[:3].T

        T = self.estimate_pose(Q1, tp2)
        return T

def main():
    data_dir = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/01'
    vo = VisualOdometry(data_dir)

    gt_path = []
    estimated_path = []

    cur_pose = np.eye(4)  # Initialize the current pose

    for i in tqdm(range(1, len(vo.gt_poses))):
        if i == 1:
            cur_pose = vo.gt_poses[0]
        else:
            T = vo.get_pose(i)
            print(f"Frame {i}: Transformation Matrix T:\n{T}")
            cur_pose = np.matmul(cur_pose, T)
            print(f"Frame {i}: Current Pose:\n{cur_pose}")

        gt_path.append((vo.gt_poses[i][0, 3], vo.gt_poses[i][2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        print(f"Frame {i}: Estimated Position: {estimated_path[-1]}")

    if estimated_path:
        plt.plot(*zip(*gt_path), label="Ground Truth")
        plt.plot(*zip(*estimated_path), label="Estimated Path")
        plt.legend()
        plt.show()
    else:
        print("No estimated path was generated.")

if __name__ == "__main__":
    main()