import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
import os
from scipy.optimize import least_squares
import cupy as cp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class VIO():
    def __init__(self, data_dir):
        # load the intrinsic and extrinic calibration parameters of the stereo cam
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calibration/calib.txt') # K and P are the intrinsic and projection matrices for left and right cam
        
        # load the ground truth poses and put them into a list or array called gt_poses
        path = data_dir + '/data/oxts/data'
        self.gt_poses = self.raw_ground_truth(path)
        
        # load images from left and right cam (stores all of them which is why running it take a bit at the beginning)
        self.images_l = self._load_images(data_dir + '/data/image_00/data')
        self.images_r = self._load_images(data_dir + '/data/image_01/data')
        
        # create a stereo block matching object for computing disparity maps between left and right images
        # (min possible disp value; 0 = no shift, range of disparity values, size of block, smoothness parameters)
        # disparity value is just a number that represents the horizontal shift (in pixels) of a point between the left and right images
        self.disparity = cv2.cuda_StereoBM.create(numDisparities = 64, blockSize = 15)
        
        # compute and stores disparity map for first pair of images then divides the map by 16 to normalize them bc OpenCV disp values are scaled by 16
        # dividing by 16 restores the original disparity value in real-world units
        self.disparities = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        
        # initializes ORB        
        self.detector = cv2.cuda_ORB.create(2750) # ORB is a feature detector and descriptor extractor that is faster than SIFT and SURF
       
        # create a brute force matcher for comparing feature descriptors
        self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)

    @staticmethod # inidicates that the following method does not depend on instance variables or methods

    def _load_calib(filepath):
        '''Loads the calibration file of the camera which includes intrinsics and projection matrix for both cams'''
        # Intrinsic parameters describe the internal characteristics of the cam (focal length, pixel size, optical center)
        # - they are used to map 3D points in the cam's field of view to 2D pixel coordinates
        
        # Projection matrices describe how 3D points are projected into 2D image space for each cam
        # - includes both intrinsic parameters and extrinsic parameters (position and oreintation relative to a world frame)
        
        with open(filepath, 'r') as f:
            # read first line from file then parses the line into a numpy array then splits line into numbers based on spaces
            params = np.fromstring(f.readline(), dtype = np.float64, sep = ' ') # fromstring converts text data into numerical data
            P_l = np.reshape(params, (3, 4)) # 3x4 matrix used to project 3D points into the left camera image (reshaping ensures data is in correct format)
            K_l = P_l[0:3, 0:3] # 3x3 matrix defining the internal properties of the left camera (extracts top-left 3x3 sub matrix of P_l)
            
            # same steps repeated for right cam
            params = np.fromstring(f.readline(), dtype = np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r # these values are extremely important for stereo matching and computing the disparity maps
    
    @staticmethod
    
    def raw_ground_truth(oxts_path):
        '''Goes through oxts file, computes the ground truth position and orientation of the vehicle'''
        imu_files = sorted(glob.glob(oxts_path + '/*.txt')) # list of all the IMU files
        
        # Loading first file to use as reference origin
        initial_data = np.loadtxt(imu_files[0])
        lat0, long0, alt0 = initial_data[:3]
        
        ground_truth_poses = []
        
        # Processing files
        for file in imu_files:
            data = np.loadtxt(file)
            lat, long, alt = data[:3]
            roll, pitch, yaw = data[3:6]
            
            def latlong_to_cart(lat, long, lat0, long0):
                '''Converts the given latitude and longitude to local Catesian (meteres) using a reference from GPS'''
                R_earth = 6378137 # radius of the Earth in meters
                scale = np.cos(np.radians(lat0)) # scale factor for UTM
                x = scale * (long - long0) * (np.pi / 180) * R_earth 
                y = (lat - lat0) * (np.pi / 180) * R_earth
                return x, y
            
            # Convert GPS to local Cartesian coordinates
            x, y = latlong_to_cart(lat, long, lat0, long0) 
            z = alt - alt0
            
            # Convert roll, pitch, and yaw to rotation matrix
            R_mat = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            
            # Create the transformation matrix
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = [x, y, z]
            
            # Append the transformation matrix to the list
            ground_truth_poses.append(T)
            
        return np.array(ground_truth_poses)
    
    
    def _load_images(self, filepath):
        '''Loads the images from the specified directory'''
        # create a list of file paths for all images in that directory
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))] # sorts the list of files in ascending order
        
        # reads each image file from image_paths and loads it as a grayscale image
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

        return images
    
    @staticmethod
    
    def _form_transf(R, t):
        '''Forms a 4x4 transformation matrix from a rotation matrix and translation vector'''
        # create a 4x4 identity matrix (diagonal elements are 1 and rest are 0)
        T = np.eye(4, dtype = np.float64)
        
        # insert the rotation matrix into top-left 3x3 block of T
        # rotation matrix defines how to rotate a point or object in 3D space
        T[:3, :3] = R
        
        # insert translation vector into top-right column of T
        # translation vector specifies how far an object or point should be shifted in 3D space along the x, y, and z axes
        T[:3, 3] = t
        return T
    
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        '''Calculates the reprojection residuals for the given transformation'''
        # Get the rotation vector
        r = dof[:3] # the first 3 elements of dof, representing the rotation vector in axis-angle form
        
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r) # converts rotation vector r into 3x3 matrix R 
        
        # Get the translation vector
        t = dof[3:] # the last 3 elements of dof, representing the translation vector in 3D
        
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t) # uses previous function to combine rotation matrix R and translation vector t into a single 4x4 transformation matrix transf
        
        # Projecting 3D points from first image to the second image
        f_projection = np.matmul(self.P_l, transf) 
        
        # Projecting 3D points from second image to the first image
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))
        
        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1)) # makes a column of ones with the same number of rows as q1
        Q1 = np.hstack([Q1, ones]) # just horizontally stacks (concatenates) arrays --> basically converts it into homogeneous coordinates so we can do matrix multiplication
        Q2 = np.hstack([Q2, ones])
        
       # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        
        # Un-homogenize because its in the form [x, y, w] where w is the scale factor so we divide my w to get it back to 2D coordinates [x/w, y/w]
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        
        return residuals
    
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp = 0.0, max_disp = 100.0):
        '''Calculates the right keypoints (feature points)'''
        
        def get_idxs(q, disp):
            # This subfunction retrieves disparity values at the feature point coordinates (q) from the disparity map (disp)
            # also applies a mask to filter points whose disparity values are within the range [min_disp, max_disp]
            q_idx = q.astype(int) # converts feature points to integer indices (disp maps are 2D arrays so we need integer coordinates to index into them)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]] # retrieves the disparity value for each feature point (q) from the disparity map (disp)
            return disp, cp.where(cp.logical_and(min_disp < disp, disp < max_disp), True, False) # keeps points where disp is between min and max disp
        
        # Convert inputs to CuPy arrays
        q1 = cp.asarray(q1)
        q2 = cp.asarray(q2)
        disp1 = cp.asarray(disp1)
        disp2 = cp.asarray(disp2)
        
         # Get the disparities for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        # Ensures that only points that are valid in both frames are kept
        in_bounds = cp.logical_and(mask1, mask2)
        
        # Filter the points using the mask which ensures that only feature points with valid disparity values in both frames are kept
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        # Creating copies to do calculations
        q1_r, q2_r = cp.copy(q1_l), cp.copy(q2_l)
        
        # For stereo images, the right image point is horizontally shifted by the disparity value (so only the x value not y)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return cp.asnumpy(q1_l), cp.asnumpy(q1_r), cp.asnumpy(q2_l), cp.asnumpy(q2_r) # valid feature points in the left images and the corresponding feature points in the right images
    
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        '''Triangulates 2D points from both images to get 3D points'''
        # Check if points are empty
        if q1_l.size == 0 or q1_r.size == 0 or q2_l.size == 0 or q2_r.size == 0:
            raise ValueError("No points provided for triangulation.")
        
        # Validate number of points match bc for triangulation, each point in the left image must have a corresponding point in the right image
        if q1_l.shape[0] != q1_r.shape[0]:
            raise ValueError("Number of points in q1_l and q1_r do not match.")

        # Ensure the inputs are of correct shape and type to ensure compatibility with the triangulatePoints function
        q1_l = q1_l.T.astype(np.float32)  # Shape (2, N)
        q1_r = q1_r.T.astype(np.float32)  # Shape (2, N)
        q2_l = q2_l.T.astype(np.float32)  # Shape (2, N)
        q2_r = q2_r.T.astype(np.float32)  # Shape (2, N)
        
        # Triangulate points from i-1'th image
        # Uses projection matrices and 2D point to compute 3D coordinates
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l, q1_r)
        
        # Un-homogenize
        # Homogeneous coordinates represent points as [x,y,z,w] so we divide by w to get Cartesian coordinates [x,y,z] = [x/w, y/w, z/w]
        Q1 = np.transpose(Q1[:3] / Q1[3]) 

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l, q2_r)
        
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        
        return Q1, Q2 # returns 3D coordinates of both frames (converting from cupy to numpy arrays)
    
    def estimate_pose(self, q1, q2, Q1, Q2, max_iter = 100):
        '''Estimates the transformation matrix'''
        early_termination_threshold = 5 # will terminate optimization early if no improvement is found for 5 iterations

        # Initialize the min_error and early_termination counter
        min_error = float('inf') # tracks the smallest reprojection error found so far
        early_termination = 0 # counts consecutive iterations where the error does not improve

        for _ in range(max_iter): # each iteration randomly samples feature points, optimizes the transformation matrix, and tracks the best solution
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6) # random indices of 6 points
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx] # corresponding 2D and 3D points

            # Make the start guess
            in_guess = np.zeros(6) # 6 parameters: rx, ry, rz, tx, ty, tz for rotation and translation
            
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method = 'trf', loss= 'huber', f_scale = 2.0, max_nfev = 200,
                                    args = (sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2) # .x represents the optimized parameters --> there are different callouts (can look into if curious)
            error = error.reshape((Q1.shape[0] * 2, 2)) # the output of error is a flattened 1D array but we need them as a pair of [x,y] differences for each points
            error = np.sum(np.linalg.norm(error, axis = 1)) # calculates the Euclidean norm (distance) of each residual pair [x,y] --> gets magnitude of the error for each point
            # also sums the errors to give the total reprojection error used to determine if the current transformation is better than the previous one

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = self._form_transf(R, t)
        
        return transformation_matrix
    
    def match_features(self, img1, img2):
        '''Detecting and matching keypoints in the stereo images'''
        # Convert images to GpuMat
        gpu_img1 = cv2.cuda_GpuMat()
        gpu_img2 = cv2.cuda_GpuMat()
        gpu_img1.upload(img1)
        gpu_img2.upload(img2)
        
        # Detect keypoints and descriptors for both images
        kp1_gpu, des1_gpu = self.detector.detectAndComputeAsync(gpu_img1, None) # None is no mask
        kp2_gpu, des2_gpu = self.detector.detectAndComputeAsync(gpu_img2, None)
        
        # Convert from GPU to CPU to use later on
        kp1 = self.detector.convert(kp1_gpu) 
        kp2 = self.detector.convert(kp2_gpu)
        
        # Match the descriptors between the two images using k-nearest neighbors (k-NN)
        matches = self.matcher.knnMatch(des1_gpu, des2_gpu, k = 2) # a list of matches, where each match contains two neighbors  (k = 2)

        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches: # m is the best (closest match) and n is the second-best match
            # we use the second-best to help determine if the best match is distinct or ambiguous
            if m.distance < 0.6 * n.distance: # Lowe's Ratio test to reduce false positives in matching by ensuring the best match is distinct from the second-best match
                good_matches.append(m) # if it satisfies the test, then add to good_matches

        # For each "good match", retrieves the 2D coordinates of the corresponding keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) # retrieving 2D position fo the keypoint in img1 corresponding to match m
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        return pts1, pts2
        
    def get_pose(self, i):
        '''Calculates the transformation matrix for the i-th frame'''
        # Get the i-1'th image and i'th image
        img1, img2 = self.images_l[i - 1], self.images_l[i]
        
        # Use ThreadPoolExecutor to parallelize feature matching and disparity calculation
        with ThreadPoolExecutor(max_workers = 2) as executor:
            # Submit feature matching task
            future_match = executor.submit(self.match_features, img1, img2)
        
            # Submit disparity computation task
            future_disp = executor.submit(self.disparity.compute, img2, self.images_r[i])
        
            # Get the results
            tp1, tp2 = future_match.result()
            disparity = future_disp.result()
            
        # Calculate the disparities
        self.disparities.append(np.divide(disparity.astype(np.float32), 16))
        
        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1, tp2, self.disparities[i - 1], self.disparities[i])
                   
        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        
        return transformation_matrix
    
    def compute_errors(self, ground_truth_poses, estimated_poses):
        '''Calculates the translational and rotational errors between the ground truth and estimated poses'''
        trans_errors = []
        rot_errors = []
        
        for gt_pose, est_pose in zip(ground_truth_poses, estimated_poses):
            # Extract the translation vectors from the ground truth and estimated poses
            gt_trans = gt_pose[:3, 3]
            est_trans = est_pose[:3, 3]
            
            # Calculate the translational error
            trans_error = np.linalg.norm(gt_trans - est_trans)
            trans_errors.append(trans_error)
            
            # Extract the rotation matrices from the ground truth and estimated poses
            gt_rot = gt_pose[:3, :3]
            est_rot = est_pose[:3, :3]
            
            # Compute relative rotation matrix
            relative_rot = np.dot(gt_rot.T, est_rot)
            
            # Compute rotation error in degrees
            angle = np.arccos((np.trace(relative_rot) - 1) / 2)
            rot_errors.append(np.degrees(angle))
            
        return trans_errors, rot_errors

    
    def visualize_trajectory_with_errors(self, gt_poses, est_poses, trans_errors, rot_errors):
        '''Visualizes the ground truth and estimated camera paths along with the translational and rotational errors'''
        # Extract the x and y coordinates of the ground truth and estimated paths
        ground_truth_xy = np.array([pose[:2, 3] for pose in gt_poses])
        estimated_xy = np.array([pose[:2, 3] for pose in est_poses])
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize = (8, 6))
               
        # Plot ground truth and estimated trajectories
        axs[0].plot(ground_truth_xy[:, 0], ground_truth_xy[:, 1], marker = 'o', markersize = 1, linestyle = '-', color = 'blue', label = "Ground Truth Trajectory")
        axs[0].plot(estimated_xy[:, 0], estimated_xy[:, 1], marker = 'o', markersize = 1, linestyle = '-', color = 'red', label = "Estimated Trajectory")
        
        axs[0].set_xlabel("X Position (meters)")
        axs[0].set_ylabel("Y Position (meters)")
        axs[0].set_title("Ground Truth vs. Estimated Trajectory (X-Y Plane)")
        axs[0].legend()
        axs[0].grid()
        
        # Plot translational errors
        axs[1].plot(range(len(trans_errors)), trans_errors, marker = 'o', markersize = 1, linestyle = '-', color = 'red', label = "Translational Error (meters)")
        axs[1].set_xlabel("Frame Index")
        axs[1].set_ylabel("Error (meters)")
        axs[1].set_title("Translational Error Over Time")
        axs[1].legend()
        axs[1].grid()
        
        # Plot rotational errors
        axs[2].plot(range(len(rot_errors)), rot_errors, marker = 'o', markersize = 1, linestyle = '-', color = 'orange', label="Rotational Error (degrees)")
        axs[2].set_label("Frame Index")
        axs[2].set_ylabel("Error (degrees)")
        axs[2].set_title("Rotational Error Over Time")
        axs[2].legend()
        axs[2].grid()
        
        # Show the plots
        plt.tight_layout()
        plt.show()
           
def main():
    '''This function integrates the visual odometry pipeline by calling get_pose and computes the estimated cam trajectory while comparing it to the ground truth'''
    
    # Path to dataset and initialization
    data_dir = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/Sequence3'
    vo = VIO(data_dir) # creates an instance of this class so will call __init__ when this happens

    # Use preloaded ground truth poses
    ground_truth_poses = vo.gt_poses
    
    # Initialize estimated path storage
    estimated_poses = [] 
    
    # Iterate over frames
    for i, gt_pose in enumerate(tqdm(ground_truth_poses, unit = "poses")): 
        if i < 1:
            cur_pose = gt_pose # for the first frame, the current pose is initialized to the ground truth pose (no motion estimation because no previous frames to compare with)
        else:
            transf = vo.get_pose(i) # computes the transformation matrix for the ith frame
            cur_pose = np.matmul(cur_pose, transf) # updates the current pose

        estimated_poses.append(cur_pose)
        
    # Compute the translational and rotational errors
    trans_errors, rot_errors = vo.compute_errors(ground_truth_poses, estimated_poses)

    # Visualize the estimated path along with the errors
    vo.visualize_trajectory_with_errors(ground_truth_poses, estimated_poses, trans_errors, rot_errors)

if __name__ == "__main__":
    main()
