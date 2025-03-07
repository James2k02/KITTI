import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def latlong_to_cart(lat, long, lat0, long0):
    '''Converts the given latitude and longitude to local Catesian (meteres) using a reference from GPS'''
    R_earth = 6378137 # radius of the Earth in meters
    scale = np.cos(np.radians(lat0)) # scale factor for UTM
    x = scale * (long - long0) * (np.pi / 180) * R_earth 
    y = (lat - lat0) * (np.pi / 180) * R_earth
    return x, y

def ground_truth(oxts_path):
    '''Goes through oxts file, computes the ground truth position and orientation of the vehicle'''
    imu_files = sorted(glob.glob(oxts_path + '/*.txt')) # list of all the IMU files
    
    # Loading first file to use as reference origin
    initial_data = np.loadtxt(imu_files[0])
    lat0, long0 = initial_data[:2]
    xy_positions = []
    
    # Processing files
    for file in imu_files:
        data = np.loadtxt(file)
        lat, long = data[:2]
        x, y = latlong_to_cart(lat, long, lat0, long0) # convert GPS to local Cartesian
        xy_positions.append([x, y]) # store the x and y positions
    return np.array(xy_positions)

# IMU file path
oxts_path = "C:/Users/james/OneDrive/Documents/University/Year 4/dataset/Sequence3/data/oxts/data"

# Load the ground truth positions
ground_truth_positions = ground_truth(oxts_path)

# Plot full ground truth trajectory
plt.figure(figsize=(8, 6))
plt.plot(ground_truth_positions[:, 0], ground_truth_positions[:, 1], marker = 'o', markersize = 1, linestyle = '-', color = 'blue', label = "Ground Truth Trajectory")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.title("KITTI Full Ground Truth Trajectory (X-Y Plane)")
plt.legend()
plt.grid()
plt.show()