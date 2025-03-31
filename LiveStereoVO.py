import cv2 # OpenCV library for image processing and feature extraction
import numpy as np # Numerical operations
from scipy.optimize import least_squares # Optimization for non-linear least squares problems
import matplotlib.pyplot as plt # Plotting for visualization
from tqdm import tqdm # Progress bar for loops
import pyrealsense2 as rs # Intel RealSense library for camera capture

class VisualOdometry():
    def __init__(self, K_l, P_l, K_r, P_r):  
        self.K_l, self.P_l = K_l, P_l
        self.K_r, self.P_r = K_r, P_r
                  
        self.disparity = cv2.cuda_StereoBM.create(numDisparities = 64, blockSize = 15)
        self.disparities = []
                    
        self.detector = cv2.cuda_ORB.create(2750) # ORB is a feature detector and descriptor extractor that is faster than SIFT and SURF
       
        self.matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)

        
    @staticmethod # inidicates that the following method does not depend on instance variables or methods
    
    def _load_calib(filepath):
        
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype = np.float64, sep = ' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            
            params = np.fromstring(f.readline(), dtype = np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r 
    
    @staticmethod
           
    def _form_transf(R, t):
        
        T = np.eye(4, dtype = np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):        

        r = dof[:3] 
        
        R, _ = cv2.Rodrigues(r) 
        
        t = dof[3:] 
        
        transf = self._form_transf(R, t) 

        f_projection = np.matmul(self.P_l, transf) 

        b_projection = np.matmul(self.P_l, np.linalg.inv(transf)) 

        ones = np.ones((q1.shape[0], 1)) 
        Q1 = np.hstack([Q1, ones]) 
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)
        
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        q2_pred = Q1.dot(b_projection.T)
        
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]
        
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()       
        return residuals
    
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):        
        def get_idxs(q, disp):
            q_idx = q.astype(int) 
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
    
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        
        in_bounds = np.logical_and(mask1, mask2)
        
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r 
    
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        
        if q1_l.size == 0 or q1_r.size == 0 or q2_l.size == 0 or q2_r.size == 0:
            raise ValueError("No points provided for triangulation.")
        if q1_l.shape[0] != q1_r.shape[0]:
            raise ValueError("Number of points in q1_l and q1_r do not match.")

        q1_l = q1_l.T.astype(np.float32)  
        q1_r = q1_r.T.astype(np.float32)  
        q2_l = q2_l.T.astype(np.float32) 
        q2_r = q2_r.T.astype(np.float32)  
        
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l, q1_r)
        Q1 = np.transpose(Q1[:3] / Q1[3]) 

        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l, q2_r)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        
        return Q1, Q2 
    
    def estimate_pose(self, q1, q2, Q1, Q2, max_iter = 100): 
        
        early_termination_threshold = 5
        
        min_error = float('inf') 
        early_termination = 0 

        for _ in range(max_iter): 
            sample_idx = np.random.choice(range(q1.shape[0]), 6) 
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx] 

            in_guess = np.zeros(6) 

            opt_res = least_squares(self.reprojection_residuals, in_guess, method = 'trf', loss= 'huber', f_scale = 2.0, max_nfev = 200,
                                    args = (sample_q1, sample_q2, sample_Q1, sample_Q2))

            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2) 
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis = 1))

            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                break

        r = out_pose[:3]

        R, _ = cv2.Rodrigues(r)

        t = out_pose[3:]

        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix
    
    def match_features(self, img1, img2):

        gpu_img1 = cv2.cuda_GpuMat()
        gpu_img2 = cv2.cuda_GpuMat()
        gpu_img1.upload(img1)
        gpu_img2.upload(img2)
        
        kp1_gpu, des1_gpu = self.detector.detectAndComputeAsync(gpu_img1, None)
        kp2_gpu, des2_gpu = self.detector.detectAndComputeAsync(gpu_img2, None)
      
        kp1 = self.detector.convert(kp1_gpu) 
        kp2 = self.detector.convert(kp2_gpu)
        
        matches = self.matcher.knnMatch(des1_gpu, des2_gpu, k = 2) 

        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        return pts1, pts2
    
    def get_pose_from_images(self, img1_l, img2_l, img2_r):
        
        disparity = self.disparity.compute(img2_l, img2_r).astype(np.float32) / 16
        self.disparities.append(disparity)
        
        tp1, tp2 = self.match_features(img1_l, img2_l)
        
        disparity_prev = self.disparities[-2]
        
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1, tp2, disparity_prev, disparity)
        
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix
    
def get_intrinsics_and_projections(pipeline):
    profile = pipeline.get_active_profile()
    left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    intr_l = left.get_intrinsics()
    intr_r = right.get_intrinsics()
    K_l = np.array([[intr_l.fx, 0, intr_l.ppx], [0, intr_l.fy, intr_l.ppy], [0, 0, 1]])
    K_r = np.array([[intr_r.fx, 0, intr_r.ppx], [0, intr_r.fy, intr_r.ppy], [0, 0, 1]])
    extr = right.get_extrinsics_to(left)
    T = np.eye(4)
    T[:3, 3] = -np.array(extr.translation)
    P_l = np.hstack((K_l, np.zeros((3, 1))))
    P_r = np.hstack((K_r, K_r @ T[:3, 3:4]))
    return K_l, P_l, K_r, P_r

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)
    
    device = pipeline.get_active_profile().get_device()
    depth_sensor = device.first_depth_sensor()[0]
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        
    align = rs.align(rs.stream.infrared)
    return pipeline, align

def get_stereo_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    img_l = np.asanyarray(aligned.get_infrared_frame(1).get_data())
    img_r = np.asanyarray(aligned.get_infrared_frame(2).get_data())
    return img_l, img_r
          
def main():
    
    pipeline, align = init_realsense()
    K_l, P_l, K_r, P_r = get_intrinsics_and_projections(pipeline)
    vo = VisualOdometry(K_l, P_l, K_r, P_r) 

    cur_pose = np.eye(4) 
    trajectory = []

    # Initialize with first stereo frame
    img1_l, img1_r = get_stereo_frames(pipeline, align)
    disparity = vo.disparity.compute(img1_l, img1_r).astype(np.float32) / 16
    vo.disparities = [disparity]
    
    plt.ion()
    while True:
        img2_l, img2_r = get_stereo_frames(pipeline, align)
        
        try:
            transf = vo.get_pose_from_images(img1_l, img2_l, img2_r)
            cur_pose = cur_pose @ transf
            trajectory.append((cur_pose[0, 3], cur_pose[1, 3]))
            
            # Visualize trajectory live
            if len(trajectory) > 1:
                traj = np.array(trajectory)
                plt.clf()
                plt.plot(traj[:, 0], traj[:, 1], Label = "Estimated Path")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Live VO Trajectory")
                plt.pause(0.001)
                
        except Exception as e:
            print("Pose estimation failed: ", e)
            continue
       
        img1_l, img1_r = img2_l, img2_r
        
        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    pipeline.stop()
    plt.ioff()
    plt.show()  



if __name__ == "__main__":
    main()
     
    
    