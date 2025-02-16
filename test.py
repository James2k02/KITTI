import numpy as np
import cv2 
import time

image_path = 'C:/Users/james/OneDrive/Documents/University/Year 4/dataset/sequences/00/image_0/000122.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(image)

orb = cv2.cuda_ORB.create(
    nfeatures = 3000,         # More keypoints for better tracking
    scaleFactor = 1.2,        # Adjust scale to capture more details
    nlevels = 8,              # Pyramid levels for multi-scale detection
    edgeThreshold = 10,       # Reduce to detect features near edges
    patchSize = 15
)

start_time = time.time()
keypoints_gpu, descriptors_gpu = orb.detectAndComputeAsync(gpu_img, None)
end_time = time.time()

keypoints = orb.convert(keypoints_gpu)
descriptors = descriptors_gpu.download()

print(f"Detected {len(keypoints)} keypoints in {end_time - start_time:.4f} seconds using CUDA.")

img_keypoints = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(img_keypoints, (x, y), radius=3, color=(0, 0, 255), thickness=-1)  # Sexy small green dot

cv2.imshow("Keypoints", img_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

