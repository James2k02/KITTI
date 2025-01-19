import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
# cv2 is the OpenCV library that's used for computer vision tasks. In this case, feature detecting and matching, and pose estimation
# numpy is a library for numerical computations. In this case, used for matrix operations and linear algebra
# matplot is a library that will be used for plotting graphs and visualizing data. In this case, used to display the trajectories and errors

'''Function to load the stereo images (image_0 and image_1 in KITTI dataset)'''
def load_stereo_images(folder, frame_id):
    # Loading the left and right stereo images for a specified frame
    left_img_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/01/image_0/{:06d}.png".format(frame_id) # the frame_id:06d zero pads it to 6 digits so frame 1 is 000001
    right_img_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/01/image_1/{:06d}.png".format(frame_id)
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE) # reads the image from that path and specifies to load it as a single-channel grayscale image
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    return left_img, right_img

'''Function to compute disparity map using the stereo images'''
def compute_disparity(left_img, right_img):
    # Computes the disparity map, which encodes pixel shifts between the left and right images
    # Disparity is the horizontal difference in the position of a point in the left vs. right image
    # If a 3D point is close ot cam, the disparity is large (shifts more between the 2 images) and if far then disparity is smaller
    # Directly related to depth. By using the disparity map, the depth can be calculated of each pixel
    stereo = cv2.StereoSGBM_create(minDisparity = 0, numDisparities = 96, blockSize = 15, P1 = 8*3*15**2, P2 = 32*3*15**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 50, speckleRange = 2) # creates a stereo block-matching object for disparity computation
    # numDisparities is the disparity search range and must be divisible by 16, larger values allow the alg to match features farther apart
    # blockSize = 15 is the size of the block used for matching
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0 # computes disparity map by comparing blocks (windows of pixels) between the left and right images
    # it converts values from fixed-point representation to floating-point and stores disparities scaled by 16 for precision
    disparity = cv2.medianBlur(disparity, 5)  # Denoise disparity map
    return disparity

'''Function to convert disparity to depth'''
def disp_to_depth (disparity, baseline, focal_length):
    # Converts the disparity mpa into a dpeth map using the stereo cam geometry
    # Formula is Z = f * b / disparity
    depth = np.zeros_like(disparity, dtype = np.float32) # initializes a depth map filled with zeros
    depth[disparity > 0] = (focal_length * baseline) / disparity[disparity > 0] # computes the depth only for valid disparity values (> 0)
    depth[depth > 50] = 0  # Cap depth values to remove outliers
    return depth

'''Function for feature detection and matching'''
def feature_detect_match(left_img, depth, prev_kp, prev_des, K):
    # Detecting keypoints in the left image
    # Matched features between the previous frame and current frame
    # Converts keypoints to 3D points using the depth map
    # The right image is not used here because the disparity and depth maps are already done so at this step, we just need to match features between frames not left and right images
    akaze = cv2.AKAZE_create(threshold = 0.001) # creating an AKAZE detector object (AKAZE is best in terms of a balance between robustness and speed, especially for real-time scenarios)
    kp, des = akaze.detectAndCompute(left_img, None) # detecting keypoints and computes descriptors for img (None means no mask is used so detect features everywhere)
    # a mask is an optional binary image that specifies the region of interest when detecting keypoints
    # can play around with different masks to rule out unwanted areas for detection
    # kp is a list of keypoints and each keypoint has a location coordinate, scale, and angle
    
    # # Ensure descriptors are not None and in the correct type
    # if des is not None:
    #     des = des.astype(np.uint8)
    # if prev_des is not None:
    #     prev_des = prev_des.astype(np.uint8)
        
    matches = []
    
    # Matches descriptors from previous frame with descriptors in the current frame to track features over time
    if prev_des is not None:
        prev_des = prev_des.astype(np.uint8)
        flann = cv2.FlannBasedMatcher(dict(algorithm = 6, table_number = 6, key_size = 12, multi_probe_level = 1), dict(checks = 50))
        # alg 1 is using KDTree (can research different ones), trees is no. of trees in KDTree (higher = more accurate but slower)
        # checks is the no. of times alg checks for best match (higher = slower but better matches)
        raw_matches = flann.knnMatch(prev_des, des, k = 2) # k = 2 finds the 2 best matches for each descriptor in the previous frame
        for match_group in raw_matches: # Lowe's ratio test to filter matches to remove false positives
            if len(match_group) == 2:
                m, n = match_group
            # for a match to be valid, the closest match (m) must be significantly better than the second-best match (m)
            if m.distance < 0.6 * n.distance: # the distance is the similarity between two descriptors (smallers = more similar)
                matches.append(m) # append just add it to the end of a list (increases the size of the list by that size)
                
    # Converting keypoints to 3D points using the depth map and camera intrinsics
    points_3d = []
    valid_matches = []
    for match in matches:
        u, v = np.int32(prev_kp[match.queryIdx].pt) # extracting the pixel coordinates
        z = depth[v, u] # retrieves the depth value for the pixel (u, v) from the depth map
        if z > 0: # only considers points with valid depth values (> 0)
            # Use the intrinsic matrix K to back-project the pixel coordinates into 3D space
            x = (u - K[0, 2]) * z / K[0, 0] # x = (u - cx) * z / fx --> cx and cy are the principal points (image center)
            y = (v - K[1, 2]) * z / K[1, 1] # y = (v - cy) * z / fy --> fx and fy are the focal lengths in the x and y direction
            points_3d.append([x, y, z])
            valid_matches.append(match)
    return kp, des, np.array(points_3d, dtype=np.float32), valid_matches

'''Function to estimate pose'''
def estimate_pose(points_3d, kp, matches, K):
    # Estimates relative pose (rotation R and translation t) of the cam between two frames by solving the Perspective-n-Point (PnP) problem
    points_2d = np.float32([kp[m.trainIdx].pt for m in matches]) # extracts 2D image coorindates of keypoints in current frame that correspond to the matched 3D points from previous frame
    _, R_vec, t, inliers = cv2.solvePnPRansac(points_3d, points_2d, K, None) # estimates the cam's rotation and translation by solving PnP
    # the PnP problem finds the camera pose that best aligns a set of 3D points (P) with their corresponding 2D projections (p) in the image, using the cam's intrinsic matrix K
    # p = K * (R * P + t)
    # none means no distortion coefficients (assumes undistorted images)
    R, _ = cv2.Rodrigues(R_vec) # converts the calculated rotation vector into a rotation matrix
    return R, t, inliers

'''Function to load calibration file'''
def load_calibration(file_path):
    # Reads the given camera calibration file
    # Extracts the intrinsic matrix (K) of left cam (bc it is the reference cam)
    # Computes the baseline (B) which is the distance between the left and right cameras
    with open(file_path, 'r') as f: # open the calib file in read mode
        lines = f.readlines()# reads all lines from the file and stores them as a list of strings (lines)
    P0 = np.array([float(x) for x in lines[0].split()[1:]]).reshape(3, 4) # splits the first line (P0) into individual elements and skips the first element (P0:)
    # also converts each string into a floating-point number and converts the list into a numpy array and reshapes it into a 3x4 matrix
    P1 = np.array([float(x) for x in lines[1].split()[1:]]).reshape(3, 4)
    K = P0[:3, :3] # extracts intrinsic matrix from projection matrix
    baseline = abs(P0[0, 3] - P1[0, 3]) / P0[0, 0] # B = |tx (left) - tx (right)| / fx
    return K, baseline

'''Function to visualize matches in real-time'''
def match_visualization(left_img, kp, matches, prev_img, prev_kp):
    # Visualizing feature matches live in a video window
    # Visually reporesents the matces between keypoints in the previous frame (prev_img) and the current frame (left_img) by connecting matched keypoints with lines
    
    # Check for empty keypoints or matches
    if not matches or not prev_kp or not kp:
        print("No matches or keypoints to visualize.")
        return

    # Filter invalid matches
    valid_matches = []
    for match in matches:
        if 0 <= match.queryIdx < len(prev_kp) and 0 <= match.trainIdx < len(kp):
            valid_matches.append(match)

    if not valid_matches:
        print("No valid matches to visualize.")
        return
    
    match_img = cv2.drawMatches(prev_img, prev_kp, left_img, kp, matches, None, flags = 2) # draws keypoints and lines connecting matched keypoints between two images
    cv2.imshow("Feature Detection and Matching", match_img)
    cv2.waitKey(1)
    
'''Function to align trajectories'''
def align_trajectories(ground_truth, estimated):
    """Align the estimated trajectory with the ground truth."""
    def make_homogeneous(matrix):
        """Ensure a matrix is 4x4 by adding [0, 0, 0, 1] if it's 3x4."""
        if matrix.shape == (3, 4):
            return np.vstack((matrix, [0, 0, 0, 1]))
        return matrix

    # Ensure all matrices are 4x4
    ground_truth = [make_homogeneous(pose) for pose in ground_truth]
    estimated = [make_homogeneous(pose) for pose in estimated]

    # Compute alignment matrix
    gt_initial = ground_truth[0]
    est_initial = estimated[0]
    alignment_matrix = np.linalg.inv(gt_initial) @ est_initial

    # Apply alignment matrix to all estimated poses
    aligned_trajectory = []
    for pose in estimated:
        aligned_pose = alignment_matrix @ pose
        aligned_trajectory.append(aligned_pose)

    return aligned_trajectory

'''Function to plot'''
def plot_trajectories_and_errors(ground_truth, poses):
    """Plot trajectories and errors."""
    gt_positions = [pose[:3, 3] for pose in ground_truth]
    est_positions = [pose[:3, 3] for pose in poses]
    gt_rotations = [pose[:3, :3] for pose in ground_truth]
    est_rotations = [pose[:3, :3] for pose in poses]

    translational_errors = []
    rotational_errors = []

    for gt_pos, est_pos, gt_rot, est_rot in zip(gt_positions, est_positions, gt_rotations, est_rotations):
        translational_error = np.linalg.norm(gt_pos - est_pos)
        translational_errors.append(translational_error)

        rot_diff = gt_rot.T @ est_rot
        angle = np.arccos(np.clip((np.trace(rot_diff) - 1) / 2, -1.0, 1.0))
        rotational_errors.append(np.degrees(angle))

    plt.figure(figsize=(15, 7))
    
    # Ground Truth and Estimated Trajectories
    plt.subplot(1, 2, 1)
    plt.plot([p[0] for p in gt_positions], [p[1] for p in gt_positions], label="Ground Truth")
    plt.plot([p[0] for p in est_positions], [p[1] for p in est_positions], label="Estimated", color="orange")
    plt.legend()
    plt.title("Trajectory (X vs Y)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()

    # Translational Error
    plt.subplot(1, 2, 2)
    plt.plot(range(len(translational_errors)), translational_errors, label="Translation Error")
    plt.legend()
    plt.title("Translation Error")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (meters)")
    plt.grid()

    plt.tight_layout()
    plt.show()
   
'''Main Pipeline'''
def main():
    # Paths
    dataset_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset"
    calib_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/01/calib.txt"
    gt_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/01/poses.txt"

    # Load calibration data
    K, baseline = load_calibration(calib_path) # loads cam's intrinsic matrix (K) and baseline (B) from the calibration file
    ground_truth = np.loadtxt(gt_path).reshape(-1, 3, 4) # loads the ground truth trjectory from the file, each line corresponds to a pose matrix (3x4)
    # also reshapes the ground truth data into a series of 3 x 4 matrices, each representing the cam's pose at a specific time step

    # Convert ground truth to 4x4 homogeneous matrices
    ground_truth_homogeneous = []
    for pose in ground_truth:
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :4] = pose
        ground_truth_homogeneous.append(pose_4x4)

    # Align ground truth to the camera coordinate system
    gt_aligned = [np.linalg.inv(ground_truth_homogeneous[0]) @ pose for pose in ground_truth_homogeneous]
    
    # Count the total number of frames
    image_folder = os.path.join(dataset_path, "sequences/01/image_0")
    total_frames = len([f for f in os.listdir(image_folder) if f.endswith(".png")])
    
    # Initialize variables
    trajectory = np.eye(4) # initializes the cam's pose as an identity matrix, also represents the starting pose of the camera where T = [R t, 0 1] (transformation matrix)
    poses = [trajectory[:3, :].copy()] # a list to store the estimated poses for each frame and starts with the initial pose (3x4)
    prev_kp, prev_des = None, None # stores keypoints and descriptors from the previous frame (initialized as none bc no previous frame at start)
    # prev_depth = None # stores the depth map from the previous frame for 3D point generation (also initialized as none)

    for frame_id in range(500):  
        # Load stereo images
        left_img, right_img = load_stereo_images(dataset_path, frame_id) # loading left and right grayscale images for the current frame

        # Compute disparity and depth
        disparity = compute_disparity(left_img, right_img) # computes the disparity map, which represents pixel shifts between the left and right images
        depth = disp_to_depth(disparity, baseline, K[0, 0]) # converts the disparity map into a depth map using the cam's baseline and focal length

        # Detect and match features
        kp, des, points_3d, matches = feature_detect_match(left_img, depth, prev_kp, prev_des, K)                                                   
        # detects keypoints and computes descriptors in the current frame
        # matches features between the current and previous frames
        # converts matched keypoints into 3D points using the depth map and intrinsic matrix
        
        if len(points_3d) > 0 and len(matches) > 10:
            # Estimate pose
            R, t, inliers = estimate_pose(points_3d, kp, matches, K) # estimates the cam's relative rotation (R) and translation (t) between the current and previous frames using 3D-2D correspondences
            # Update trajectory
            trajectory = trajectory @ np.vstack((np.hstack((R, t)), [0, 0, 0, 1])) # new pose updated by multiplying current trajectory with transformation matrix
            poses.append(trajectory[:3, :].copy()) # 3x4 pose matric is appended to poses list

            # # Visualize matches
            # if prev_kp is not None:
            #     match_visualization(left_img, kp, matches, prev_img = left_img, prev_kp=kp)

        # Update previous frame data
        prev_kp, prev_des = kp, des

    aligned_poses = align_trajectories(ground_truth, poses)
    plot_trajectories_and_errors(ground_truth, aligned_poses)

if __name__ == "__main__":
    main()
    




