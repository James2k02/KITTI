'''FEATURE DETECTION'''
# import numpy as np
# import cv2 
# import time

# image_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_0/000122.png'
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# gpu_img = cv2.cuda_GpuMat()
# gpu_img.upload(image)

# orb = cv2.cuda_ORB.create(
#     nfeatures = 3000,         # More keypoints for better tracking
#     scaleFactor = 1.2,        # Adjust scale to capture more details
#     nlevels = 8,              # Pyramid levels for multi-scale detection
#     edgeThreshold = 10,       # Reduce to detect features near edges
#     patchSize = 15
# )

# start_time = time.time()
# keypoints_gpu, descriptors_gpu = orb.detectAndComputeAsync(gpu_img, None)
# end_time = time.time()

# keypoints = orb.convert(keypoints_gpu)
# descriptors = descriptors_gpu.download()

# print(f"Detected {len(keypoints)} keypoints in {end_time - start_time:.4f} seconds using CUDA.")

# img_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# for kp in keypoints:
#     x, y = int(kp.pt[0]), int(kp.pt[1])
#     cv2.circle(img_keypoints, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # Sexy small green dot

# cv2.imshow("Keypoints", img_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''FEATURE MATCHING'''
# import numpy as np
# import cv2
# import time

# # Paths to KITTI images
# img0_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_0/000122.png'
# img1_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_1/000122.png'

# # Read grayscale images
# img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
# img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

# # Upload to GPU
# gpu_img0 = cv2.cuda_GpuMat()
# gpu_img1 = cv2.cuda_GpuMat()
# gpu_img0.upload(img0)
# gpu_img1.upload(img1)

# # ORB CUDA detector
# orb = cv2.cuda_ORB.create(nfeatures=3000)

# # Detect and compute
# kp_gpu0, des_gpu0 = orb.detectAndComputeAsync(gpu_img0, None)
# kp_gpu1, des_gpu1 = orb.detectAndComputeAsync(gpu_img1, None)

# # Convert back to CPU
# kp0 = orb.convert(kp_gpu0)
# kp1 = orb.convert(kp_gpu1)
# des0 = des_gpu0.download()
# des1 = des_gpu1.download()

# # Match using BFMatcher (Hamming for ORB)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des0, des1)

# # Sort by distance (best matches first)
# matches = sorted(matches, key=lambda x: x.distance)

# # Draw matches vertically instead of side-by-side
# img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:50], None, flags=2)

# # Resize to better fit 1920x1080 by stacking vertically
# max_width = 1920
# max_height = 1080
# h, w = img_matches.shape[:2]
# scaling_factor = min(max_width / w, max_height / h, 0.8)  # scale down a bit more
# new_size = (int(w * scaling_factor), int(h * scaling_factor))
# img_resized = cv2.resize(img_matches, new_size)

# # Show and save
# cv2.imshow("Feature Matches (Vertical)", img_resized)
# cv2.imwrite("matches_output_vertical.png", img_resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''DISPARITY MAP'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths to the stereo images (left: image_0, right: image_1)
left_img_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_0/000122.png'
right_img_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_1/000122.png'

# Load the images in grayscale
left = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)

# Create StereoSGBM matcher
min_disp = 0
num_disp = 16 * 12  # must be divisible by 16
block_size = 5

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 3 * block_size ** 2,
    P2=32 * 3 * block_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Compute the disparity map
disparity = stereo.compute(left, right).astype(np.float32) / 16.0

# Mask invalid disparity values
disparity[disparity < min_disp] = min_disp

# Plot with colorbar
plt.figure(figsize=(12, 6))
disp_img = plt.imshow(disparity, cmap='plasma')  # or 'inferno', 'magma', 'jet'
plt.colorbar(label='Disparity (closer = brighter)')
plt.title("Disparity Map with Depth Scale")
plt.axis('off')
plt.savefig("disparity_map_colored.png", bbox_inches='tight')
plt.show()


