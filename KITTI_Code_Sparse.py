import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths to calibration file and image directory
calibration_file = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/calib.txt"
image_dir = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_0"

def visual_odometry(images, K, output_video=False):
    # Load the first frame
    prev_frame = cv2.imread(os.path.join(image_dir, images[0]), cv2.IMREAD_GRAYSCALE)

    # Detect initial keypoints using Shi-Tomasi corner detector
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)

    # Initialize video writers if saving output
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        sparse_writer = cv2.VideoWriter('output_sparse.avi', fourcc, 20.0, (960, 540))
        feature_writer = cv2.VideoWriter('output_feature.avi', fourcc, 20.0, (960, 540))

    # Define resizing dimensions for display
    max_width = 960
    max_height = 540

    for i in range(1, len(images)):
        print(f"Processing frame {i}/{len(images)}")

        # Load the current frame
        curr_frame = cv2.imread(os.path.join(image_dir, images[i]), cv2.IMREAD_GRAYSCALE)

        # Skip if there are no keypoints in the previous frame
        if prev_pts is None or len(prev_pts) == 0:
            print("No keypoints to track. Skipping frame.")
            prev_frame = curr_frame
            prev_pts = cv2.goodFeaturesToTrack(curr_frame, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
            continue

        # Calculate sparse optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None)

        # Skip if no points were successfully tracked
        if curr_pts is None or status is None or len(curr_pts) == 0:
            print("Tracking failed. Skipping frame.")
            prev_frame = curr_frame
            prev_pts = cv2.goodFeaturesToTrack(curr_frame, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
            continue

        # Select good points
        good_prev_pts = prev_pts[status.flatten() == 1]
        good_curr_pts = curr_pts[status.flatten() == 1]

        # Sparse Optical Flow Visualization
        flow_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2BGR)
        for p1, p2 in zip(good_prev_pts, good_curr_pts):
            x1, y1 = p1.ravel()
            x2, y2 = p2.ravel()
            cv2.arrowedLine(flow_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, tipLength=0.5)

        # Resize the sparse optical flow frame
        height, width = flow_frame.shape[:2]
        scale = min(max_width / width, max_height / height)
        resized_flow_frame = cv2.resize(flow_frame, (int(width * scale), int(height * scale)))

        # Feature Matching Visualization
        # Draw matched features between consecutive frames
        orb = cv2.ORB_create()
        prev_kp = [cv2.KeyPoint(p[0][0], p[0][1], 1) for p in good_prev_pts]  # Convert points to KeyPoints
        curr_kp = [cv2.KeyPoint(p[0][0], p[0][1], 1) for p in good_curr_pts]
        matches = [cv2.DMatch(i, i, 0) for i in range(len(prev_kp))]  # Create dummy matches
        matched_frame = cv2.drawMatches(prev_frame, prev_kp, curr_frame, curr_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Resize the feature matching frame
        height, width = matched_frame.shape[:2]
        resized_matched_frame = cv2.resize(matched_frame, (int(width * scale), int(height * scale)))

        # Display Sparse Optical Flow and Feature Matching
        cv2.imshow("Sparse Optical Flow", resized_flow_frame)
        cv2.imshow("Feature Matching", resized_matched_frame)

        # Save frames to video (if enabled)
        if output_video:
            sparse_writer.write(resized_flow_frame)
            feature_writer.write(resized_matched_frame)

        # Update previous frame and points
        prev_frame = curr_frame
        prev_pts = good_curr_pts.reshape(-1, 1, 2)

        # Wait briefly to display the frames
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video writers
    if output_video:
        sparse_writer.release()
        feature_writer.release()
        print("Videos saved: 'output_sparse.avi' and 'output_feature.avi'")

    # Close all OpenCV windows
    cv2.destroyAllWindows()


    return

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






