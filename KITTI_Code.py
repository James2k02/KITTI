import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths to calibration file and image directory
calibration_file = "dataset/sequences/00/calib.txt"
image_dir = "dataset/sequences/00/image_0"

def visual_odometry(images, K, output_video=False):
    # Load the first frame and initialize ORB detector
    prev_frame = cv2.imread(os.path.join(image_dir, images[0]), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    prev_kp, prev_des = orb.detectAndCompute(prev_frame, None)

    # Initialize trajectory and pose
    trajectory = [np.array([0, 0, 0])]
    pose = np.eye(4)

    # Initialize video writer if saving output
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (prev_frame.shape[1], prev_frame.shape[0]))
        
    # Define resizing dimensions
    max_width = 960  # Half of 1920
    max_height = 540  # Half of 1080

    for i in range(1, len(images)):
        print(f"Processing frame {i}/{len(images)}")
        
        # Load the current frame
        curr_frame = cv2.imread(os.path.join(image_dir, images[i]), cv2.IMREAD_GRAYSCALE)

        # Detect and match features
        curr_kp, curr_des = orb.detectAndCompute(curr_frame, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(prev_des, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        pts_prev = np.array([prev_kp[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_curr = np.array([curr_kp[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Estimate Essential Matrix
        E, _ = cv2.findEssentialMat(pts_curr, pts_prev, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts_curr, pts_prev, K)

        # Update pose
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.ravel()
        pose = pose @ new_pose
        trajectory.append(pose[:3, 3])

        # Visualize Feature Matching
        matched_frame = cv2.drawMatches(prev_frame, prev_kp, curr_frame, curr_kp, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Feature Matching", matched_frame)
        
        # Resize the matched frame
        height, width = matched_frame.shape[:2]
        scale = min(max_width / width, max_height / height)  # Scale to fit within max dimensions
        resized_matched_frame = cv2.resize(matched_frame, (int(width * scale), int(height * scale)))

        cv2.imshow("Feature Matching", resized_matched_frame)

        # Visualize Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        hsv = np.zeros_like(cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2BGR))
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        optical_flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow("Dense Optical Flow", optical_flow_vis)
        
        # Resize optical flow visualization
        height, width = optical_flow_vis.shape[:2]
        resized_optical_flow = cv2.resize(optical_flow_vis, (int(width * scale), int(height * scale)))

        cv2.imshow("Dense Optical Flow", resized_optical_flow)

        # Save frame to video (if enabled)
        if output_video:
            video_writer.write(matched_frame)

        # Wait briefly to display the frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame and features
        prev_frame = curr_frame
        prev_kp, prev_des = curr_kp, curr_des

    # Release video writer
    if output_video:
        video_writer.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    return trajectory

def main():
   
    # Read calibration file
    with open(calibration_file, 'r') as f:
        calib_lines = f.readlines()
    
    # Extract the intrinsic matrix
    P0_line = calib_lines[0].strip().split()[1:]  # Skip the "P0:" label
    P0 = np.array([float(value) for value in P0_line]).reshape(3, 4)
    K = P0[:, :3]  # Intrinsic matrix

    print("Intrinsic Matrix (K):")
    print(K)

    # Load images
    images = sorted(os.listdir(image_dir))
    print(f"Loaded {len(images)} images for processing.")

    # Run visual odometry with feature matching and optical flow visualization
    trajectory = visual_odometry(images, K, output_video=True)

    # Plot trajectory
    trajectory = np.array(trajectory)
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 2], label="Trajectory")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Visual Odometry Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()




