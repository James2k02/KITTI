import cv2 # OpenCV library for image processing and feature extraction
import os # File and directory management
import numpy as np # Numerical operations
from scipy.optimize import least_squares # Optimization for non-linear least squares problems
import matplotlib.pyplot as plt # Plotting for visualization
from tqdm import tqdm # Progress bar for loops
from sklearn.linear_model import RANSACRegressor

class VisualOdometry():
    def __init__(self, data_dir): # the __init__ method initializes the VisualOdometry class so it gets called when an object of this class is created
        
        # load the intrinsic and extrinic calibration parameters of the stereo cam
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(data_dir + '/calib.txt') # K and P are the intrinsic and projection matrices for left and right cam
        
        # load the ground truth poses and put them into a list or array called gt_poses
        self.gt_poses = self._load_poses(data_dir + '/poses.txt')
        
        # load images from left and right cam (stores all of them which is why running it take a bit at the beginning)
        self.images_l = self._load_images(data_dir + '/image_0')
        self.images_r = self._load_images(data_dir + '/image_1')
        
        # block is the size of the window used for comparing pixel intensities between the left and right images
        block = 15 # larger block sizes average more pixel intensities which smooths out noise but reduces detail in the map
        P1 = block * block * 8 # penalty for small changes in disparity between neighboring pixel (higher values reduce noise in disparity maps)
        P2 = block * block * 32 # penalty for larger changes in disparity (high value encourages smoother disparity maps)
        
        # create a stereo block matching object for computing disparity maps between left and right images
        # (min possible disp value; 0 = no shift, range of disparity values, size of block, smoothness parameters)
        # disparity value is just a number that represents the horizontal shift (in pixels) of a point between the left and right images
        self.disparity = cv2.StereoSGBM_create(minDisparity = 0, numDisparities = 64, blockSize = block, P1 = P1, P2 = P2, disp12MaxDiff = 1, uniquenessRatio = 12, speckleWindowSize = 75, speckleRange = 2, preFilterCap = 63) 
        
        # compute and stores disparity map for first pair of images then divides the map by 16 to normalize them bc OpenCV disp values are scaled by 16
        # dividing by 16 restores the original disparity value in real-world units
        self.disparities = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
        
        # initializes AKAZW which is a robust method for detecting and describing key points in images (optimized for speed so real-time)
        #self.akaze = cv2.AKAZE_create()
        self.detector = cv2.ORB_create(5000)#SIFT_create(nfeatures = 9000, contrastThreshold = 0.02, edgeThreshold = 17, sigma = 1.2)

        # index_params = dict(algorithm =1, trees=5)
        # search_params = dict(checks=50)
        
        # create a brute force matcher for comparing feature descriptors
        # self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False) # Hamming distance as metric for comparing, disable cross-checking to allow for more flexible matches
        # self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks = 100)  # Increase 'checks' for better accuracy
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    @staticmethod # inidicates that the following method does not depend on instance variables or methods
        
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K_l (ndarray): Intrinsic parameters for left camera. Shape (3,3)
        P_l (ndarray): Projection matrix for left camera. Shape (3,4)
        K_r (ndarray): Intrinsic parameters for right camera. Shape (3,3)
        P_r (ndarray): Projection matrix for right camera. Shape (3,4)
        """
        # Intrinsic parameters describe the internal characteristics of the cam (focal length, pixel size, optical center)
        # - they are used to map 3D points in the cam's field of view to 2D pixel coordinates
        
        # Projection matrices describe how 3D points are projected into 2D image space for each cam
        # - includes boht intrinsic parameters and extrinsic parameters (position and oreintation relative to a world frame)
        
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
      
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses. Shape (n, 4, 4)
        """
        
        # Homogeneous coordinates enable the representation of both rotation and translation in a single matrix
        # - makes matrix multiplication applicable for combining multiple transformations
        # - allows you to handle transformations in one-go instead of separating rotation and translation
        
        # initialize an empty list to store the ground truth pose matrices
        poses = [] # a list of 4x4 numpy arrays where each array represents a pose matrix 
        # each 4x4 array has: 3x3 rotation matrix in upper left corner, 3x1 translation vector in last column, and homogenrous coordinates in last row [0, 0, 0, 1]
        
        with open(filepath, 'r') as f:
            for line in f.readlines(): # iterates over each line in the file bc each line corresponds to a single ground truth pose stored as 12 space-separated values
                
                # read current line and convert to numpy array of type float64 and store it in T
                T = np.fromstring(line, dtype = np.float64, sep = ' ') 
                
                # reshape the 1D array from above line into a 2D array with shape 3x4
                T = T.reshape(3, 4)
                
                # appends (attaches) an additional row [0,0,0,1] to T to make it a 4x4 homogeneous transformation matrix
                T = np.vstack((T, [0, 0, 0, 1]))
                
                # add transformation matrix T to poses list
                poses.append(T)
        return poses

    @staticmethod
    
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images. Shape (n, height, width)
        """
        
        # Loads all the images into a list
        # - the list is a list of numpy arrays where each array represents a grayscale image with dimensions (height, width)
        
        # create a list of file paths for all images in that directory
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))] # sorts the list of files in ascending order
        
        # reads each image file from image_paths and loads it as a grayscale image
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        return images
    
    @staticmethod
    
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix. Shape (3,3)
        t (list): The translation vector. Shape (3)

        Returns
        -------
        T (ndarray): The transformation matrix. Shape (4,4)
        """
        
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
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # The purpose of this function is to calculate the reprojection residuals which measure how well 3D points project back into 2D image points under a given cam transformation
        # The reprojection residuals quantify the different between:
        #   - the observed 2D feature points in an image (q1, q2)
        #   - the predicted 2D feature points obtained by projecting 3D points back onto the images using a camera transformation (T)
        # In visual odometry, reprojection residulas help verify:
        #   - if the estimated camera pose is accurate
        #   - if the 3D points and camera transformations are consistent with the observed 2D image points
        # This function checks the quality of the transformation (dof - degrees of freedom) between two frames
        #   - if the residuals are small, the transformation is accurate, if large then transformation needs to be refined
        
        # Example: 
        
        # Get the rotation vector
        r = dof[:3] # the first 3 elements of dof, representing the rotation vector in axis-angle form
        
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r) # converts rotation vector r into 3x3 matrix R 
        
        # Get the translation vector
        t = dof[3:] # the last 3 elements of dof, representing the translation vector in 3D
        
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = self._form_transf(R, t) # uses previous function to combine rotation matrix R and translation vector t into a single 4x4 transformation matrix transf

        # Create the projection matrix for the i-1'th image and i'th image
        # Projecting 3D points from first image to the second image
        # mathematically: f_projection = P_l dot T --> T transforms points from the first frame to second frame
        # f_projection is the projection matrix that points from the first frame (Q1) to the second frame (q2_pred) using the transformation T
        # predicts where 3D points from the first image (Q1) should appear in the second image (q2_pred)
        f_projection = np.matmul(self.P_l, transf) 
        
        # Projecting 3D points from second image to the first image
        # mathematically: b_projection = P_l dot T_inv --> uses inverse transformation matrix to "undo" the camera movement and map the points back
        # b_projection is the projection matrix that maps 3D points from the second image (Q2) to the first image (q1_pred)
        # uses the inverse transformation matrix to reverse the camera movement and map the points forward
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
        
        # Quick notes: 
        #   - T is directional, describing how the camera moved from frame 1 to frame 2
        #   - However, when applied to different data (Q1 vs. Q2), it inherently handles the "reverse" direction without needing inversion
        #   - That’s why f_projection is used for q1_pred—it already contains the right transformation for projecting points from frame 2 back to frame 1
        #   - The transformation matrix T works in the "reverse" direction when applied to points in frame 2 (Q2)
        #   - Only use T^-1 when projecting 3D points from frame 1 (Q1) into frame 2 (q2_pred) --> bc T^-1 applies the "forward" transformation to points that originally came from frame 1
        
        return residuals
      
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """
        # This function is used to compute feature points and their corresponding disparity values for stereo images (left and right) in consecutive frames
        # It ensures that feature points in both images (left and right) are valid based on disparity values
        # Creates a mapping between feature points in the left image (q1, q2) and the corresponding feature points in the right image (q1_r, q2_r) using disparity
        
        def get_idxs(q, disp):
            # This subfunction retrieves disparity values at the feature point coordinates (q) from the disparity map (disp)
            # also applies a mask to filter points whose disparity values are within the range [min_disp, max_disp]
            q_idx = q.astype(int) # converts feature points to integer indices (disp maps are 2D arrays so we need integer coordinates to index into them)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]] # retrieves the disparity value for each feature point (q) from the disparity map (disp)
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False) # keeps points where disp is between min and max disp
        
        # Get the disparities for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        # Combine the masks 
        # Ensures that only points that are valid in both frames are kept
        in_bounds = np.logical_and(mask1, mask2)
        
        # Filter the points using the mask which ensures that only feature points with valid disparity values in both frames are kept
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        # Calculate the right feature points 
        # Creating copies to do calculations
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        
        # For stereo images, the right image point is horizontally shifted by the disparity value (so only the x value not y)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r # valid feature points in the left images and the corresponding feature points in the right images
    
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images 
        
        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # This function takes the 2D feature points in the left and right images of two consecutive frames and computes their 3D coordinates in the world using triangulation 
        # Triangulation computes the 3D position of a point by using its 2D projections in two images (eg. left and right images in a stereo pair) with the cam's intrinsic and extrinsic parameters
        # 3D point = intersection of projection rays from the two cams
                
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
        
        # Check shapes
        # print(f"q1_l.T shape: {q1_l.shape}, q1_r.T shape: {q1_r.shape}")
    
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
        return Q1, Q2 # returns 3D coordinates of both frames
    
    def estimate_pose(self, q1, q2, Q1, Q2, max_iter = 100, ransac_thresh = 3.0): # max iterations for optimization process
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # This functions estimates the transformation matrix between two frames based on feature points and their 3D positions
        
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
            # Optimizes the transformation parameters to minimize
            # Technically don't need the optimization and can just use identity matrix as initial guess but this helps with noise
            # self.reprojection_residuals: computes the reprojection error for given transformation parameters
            # in_guess: initial guess for optimization
            # method = 'lm': Levenberg-Marquardt optimization method (good for non-linear problems)
            # max_nfev = 200: max number of function evaluations
            # args: passes the sampled 2D and 3D points as arguments
            opt_res = least_squares(self.reprojection_residuals, in_guess, method = 'trf', loss= 'huber', f_scale = 2.0, max_nfev = 200,
                                    args = (sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            # Will measure how well the optimized transformation aligns the points
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
    
    # def estimate_pose(self, q1, q2, Q1, Q2):
    #     model = RANSACRegressor(residual_threshold=2.0)
    #     model.fit(Q1, Q2)
    #     inliers = model.inlier_mask_

    #     q1_inliers = q1[inliers]
    #     q2_inliers = q2[inliers]
    #     Q1_inliers = Q1[inliers]
    #     Q2_inliers = Q2[inliers]

    #     in_guess = np.zeros(6)
    #     opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
    #                         args=(q1_inliers, q2_inliers, Q1_inliers, Q2_inliers))

    #     r = opt_res.x[:3]
    #     R, _ = cv2.Rodrigues(r)
    #     t = opt_res.x[3:]
    #     transformation_matrix = self._form_transf(R, t)
    #     return transformation_matrix
    
    def match_features(self, img1, img2):
        # This function is part of a feature-matching pipeline that identifies corresponding points in two images based on feature descriptors
        # The function matches keypoints (feature points0 between two images by:
        #   - detecting and describing keypoints in both images
        #   - matching the descriptors between the two images using a k-nearest neighbor (k-NN) approach
        #   - filtering the matches to keep only the "good matches"
        #   - returning the 2D coordinates of the matched points in both images
        
        # Detect keypoints and descriptors for both images
        # Keypoints are points of interest like corners, edges, blobs
        # Descriptors are numeric vectors describing the local image patch around each keypoint used to compare keypoints between images        
        kp1, des1 = self.detector.detectAndCompute(img1, None) # None is no mask
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # Match the descriptors between the two images using k-nearest neighbors (k-NN)
        # For each descriptor in des1, it finds the k = 2 closest descriptors in des2 based on Euclidean distance
        matches = self.matcher.knnMatch(des1, des2, k = 2) # a list of matches, where each match contains two neighbors  (k = 2)
        #matches = self.matcher.match(des1, des2)
        #matches = sorted(matches, key=lambda x: x.distance)

        good_matches = []
        for m, n in matches: # m is the best (closest match) and n is the second-best match
            # we use the second-best to help determine if the best match is distinct or ambiguous
            if m.distance < 0.6 * n.distance: # Lowe's Ratio test to reduce false positives in matching by ensuring the best match is distinct from the second-best match
                good_matches.append(m) # if it satisfies the test, then add to good_matches

        # For each "good match", retrieves the 2D coordinates of the corresponding keypoints
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) # retrieving 2D position fo the keypoint in img1 corresponding to match m
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return pts1, pts2

    
    def get_pose(self, i): # i is the index of the current frame
        """
        Calculates the transformation matrix for the i'th frame

        Parameters
        ----------
        i (int): Frame index

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        # This function calculates the transformation matrix for the ith frame which describes the cam's motion (rotation and translation) between consecutive frames
        
        # Get the i-1'th image and i'th image
        # We use the left images bc they are typically used for initial processing (eg. feature matching) before incorporating right images for depth calculations
        img1, img2 = self.images_l[i - 1], self.images_l[i]

        # Match features between the two frames
        tp1, tp2 = self.match_features(img1, img2) # tp1 and tp2 are the matched points in img1 and corresponding matched points in img2 respectively
        
        # Calculate the disparities
        # The disparity is calculated between the left image (ith frame) and the right image (ith frame) then divided by 16 to get actual values
        self.disparities.append(np.divide(self.disparity.compute(img2, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        # Commutes the corresponding feature points in the right images based on the disparity values
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1, tp2, self.disparities[i - 1], self.disparities[i])
                   
        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        
        # High level process:
        #   1) Get consecutive frames
        #   2) Match feature points to get tp1 and tp1 which are the matched points in left camera
        #   3) Compute the disparity map for depth estimation
        #   4) Compute the corresponding points in the right images using disparity values
        #   5) Calculate the 3D points
        #   6) Esimate transformation matrix
        return transformation_matrix
    
def visualize_paths_with_error_and_rotation(gt_path, estimated_path, rotation_errors, title="Visual Odometry Path with Error"):
    errors = [np.linalg.norm(np.array(gt) - np.array(est)) for gt, est in zip(gt_path, estimated_path)]

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot([p[0] for p in gt_path], [p[1] for p in gt_path], label="Ground Truth Path", color="green")
    plt.plot([p[0] for p in estimated_path], [p[1] for p in estimated_path], label="Estimated Path", color="blue")
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(range(len(errors)), errors, label="Translational Error", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Error (meters)")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(range(len(rotation_errors)), rotation_errors, label="Rotational Error (degrees)", color="orange")
    plt.xlabel("Frame Index")
    plt.ylabel("Rotational Error (degrees)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def main():
    # This function integrates the visual odometry pipeline by calling get_pose and computes the estimated cam trajectory while comparing it to the ground truth
    
    data_dir = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00'
    vo = VisualOdometry(data_dir) # creates an instance of this class so will call __init__ when this happens

    gt_path = [] # stores the ground truth cam positions (from vo.gt_poses)
    estimated_path = [] # stores the estimated cam positions computed using get_pose
    rotation_errors = [] # tracks the angular error in rotation for each frame

    max_frames = 495
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit = "poses")): # iterates overs the ground truth poses (vo.gt_poses), one for each frame
        if i < 1:
            cur_pose = gt_pose # for the first frame, the current pose is initialized to the ground truth pose (no motion estimation because no previous frames to compare with)
        else:
            transf = vo.get_pose(i) # computes the transformation matrix for the ith frame
            cur_pose = np.matmul(cur_pose, transf) # updates the current pose

            gt_rotation = gt_pose[:3, :3] # ground truth rotation matrix
            est_rotation = cur_pose[:3, :3] # estimated rotation matrix
            relative_rotation = np.dot(gt_rotation.T, est_rotation) # relative rotation matrix
            angle = np.arccos((np.trace(relative_rotation) - 1) / 2) # calculating the angle of rotation
            rotation_errors.append(np.degrees(angle)) # convert to degrees then append to rotation_errors

        # Extracting the x and z coordinates of the cam position from the grouth truth pose and estimate pose and append them to specified variables
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3])) # it's [0, 3] and [2, 3] because the translation matrix is in the last row, 0 and 2 are x and z respectively
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

    visualize_paths_with_error_and_rotation(gt_path, estimated_path, rotation_errors)

if __name__ == "__main__":
    main()    
    
'''FULL PIPELINE EXPLANATION'''

# The main goal of the code is to estimate the cam's motion (trajectory) from stereo images using visual odometry

# INITIALIZATION
#   1) Loading Calibration, Images, and Ground Truth
#     - _load_calib(filepath): loads intrinsic and projection matrices for left and right cam which are used to transform 3D points into image space and calculate disparities
#     - _load_poses(filepath): reads the ground truth poses (4x4 transformation matrices) for evaluation and visualization
#     - _load_images(filepath): loads grayscale images from the dataset for both left and right cams
#     - purpose of this section is initialize cam parameters, ground truth, and image data
#
#   2) Disparity Map Initialization
#     - cv2.StereoSGBM_create(): initializes a stereo block matching object for computing disparity maps which are used for depth esimation by matching pixels between left and right images
#
#   3) Feature Detector and Matcher Initialization
#     - AKAZE Detector (self.akaze): detects and describes feature points in the images (can change methods if needed)
#     - BFMatcher (self.matcher): matches features descriptors using the Hamming distance (can change methods if needed)
#
# PIPELINE FOR EACH FRAME (1-5 done in get_pose function)
#   1) Feature Detecting and Matching
#     - match_features(img1, img2): detects feature points and descriptors in two consecutive frames and matches descriptors between frames using k-NN and Lowe's test to find good points
#   
#   2) Disparity Map Calculation
#     - self_disparity.compute(img2, self.images_r[i]): computes the disparity map for the current frame (left-right images) then normalized by dividing by 16 to get the actual disparity
#                                                       values which is used for depth esimation (enables 3D triangulation)
#   3) Calculate Stereo Feature Points
#    - calculate_right_qs(q1, q2, disp1, disp2): uses disparity maps to calculate corresponding points in the right image for matched points in the left image and filters points based
#                                                based on valid disparity ranges (the purpose is to ensure accurate stereo correspondences for depth calculation)
#
#   4) Triangulate 3D Points
#     - calc_3d(q1_l, q1_r, q2_l, q2_r): triangulates 3D points from matched feature points in stereo pairs (left and right images); converts 2D keypoints into 3D coordinates using cam
#                                        calibration (the purpose is to create 3D points in the world frame for pose estimation)
#
#   5) Estimate Pose
#     - estimate_pose(q1, q2, Q1, Q2): estimates the cam's transformation between two frames; it also optimizes
#                                             the pose by performing least-squares optimization to minimize reprojection errors then converts the optimized rotation and translation into a
#                                             4x4 transformation matrix (the purpose is to determine the cam's motion between consecutive frames)
#
#   6) Update Pose
#     - cur_pose = np.matmul(cur_pose, transf): updates the current pose by multiplying the previous pose with the estimated transformation matrix
#
# POST-PROCESSING AND VISUALIZATION
#   1) Error Calculation
#     - Rotational and translational error
#   2) Visualization
#     - plotting ground truth vs. estimated trajectory, translational error over frames, and rotational error over frames