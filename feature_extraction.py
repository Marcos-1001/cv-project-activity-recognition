
# %% 
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import time



def high_pass_filter(image, cutoff_frequency):
    """
    Apply a high-pass filter to the image using FFT.
    """
    # Convert image to float32
    image_float = np.float32(image)

    # Perform FFT
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Create a mask for the high-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    # Apply the mask to the shifted DFT
    dft_shifted_filtered = dft_shifted * mask

    # Inverse FFT to get the filtered image
    dft_filtered_shifted = np.fft.ifftshift(dft_shifted_filtered)
    filtered_image = cv2.idft(dft_filtered_shifted)
    
    # Get the magnitude of the complex result
    filtered_image_magnitude = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    return filtered_image_magnitude

def low_pass_filter(image, cutoff_frequency):
    """
    Apply a low-pass filter to the image using FFT.
    """
    # Convert image to float32
    image_float = np.float32(image)

    # Perform FFT
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Create a mask for the low-pass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = cutoff_frequency
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1

    # Apply the mask to the shifted DFT
    dft_shifted_filtered = dft_shifted * mask

    # Inverse FFT to get the filtered image
    dft_filtered_shifted = np.fft.ifftshift(dft_shifted_filtered)
    filtered_image = cv2.idft(dft_filtered_shifted)
    
    # Get the magnitude of the complex result
    filtered_image_magnitude = cv2.magnitude(filtered_image[:, :, 0], filtered_image[:, :, 1])
    
    return filtered_image_magnitude


def flow_to_feature_vector(flow, num_bins=8):
    # Extract dx and dy
    dx = flow[..., 0]
    dy = flow[..., 1]

    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    # Normalize angles to [0, 360)
    ang = ang % 360

    # Histogram of angles (direction)
    hist_ang, _ = np.histogram(ang, bins=num_bins, range=(0, 360), weights=mag)
    hist_ang = hist_ang / (np.sum(hist_ang) + 1e-6)  # normalize

    # Statistical features
    mean_dx = np.mean(dx)
    mean_dy = np.mean(dy)
    std_dx = np.std(dx)
    std_dy = np.std(dy)

    # Magnitude stats
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)

    # Feature vector
    feature_vector = np.hstack([mean_dx, mean_dy, std_dx, std_dy, mean_mag, std_mag, hist_ang])

    return feature_vector

# Import necessary libraries
 


# Params for corner detection
feature_params = dict(maxCorners=20,  # We want only one feature
                      qualityLevel=0.1,  # Quality threshold 
                      minDistance=7,  # Max distance between corners, not important in this case because we only use 1 corner
                      blockSize=7)






############################ Parameters ####################################

""" 
winSize --> size of the search window at each pyramid level
Smaller windows can more precisely track small, detailed features -->   slow or subtle movements and where fine detail tracking is crucial.
Larger windows is better for larger displacements between frames ,  more robust to noise and small variations in pixel intensity --> require more computations
"""

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(25, 25),  # Window size
                 maxLevel=2,  # Number of pyramid levels
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))


############################ Algorithm ####################################
sift = cv2.SIFT_create()
# Step 3: Track the keypoint for every frame
def track_keypoints(video_path):
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    width = old_frame.shape[1]
    height = old_frame.shape[0]

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    frame_count = 0
    start_time = time.time()



    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #old_frame = low_pass_filter(old_frame, 30)
    #old_frame = cv2.normalize(old_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)



    # Harris Corner detection
    p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
    
    hsv = np.zeros((old_frame.shape[0], old_frame.shape[1], 3), dtype=np.uint8)

    hsv[..., 1] = 255
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # use fft and high pass filter
        
        #frame_gray = low_pass_filter(frame_gray, 40)
        #frame_gray = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        
        # Draw the rectangle on the image
        #cv2.imshow('High Pass Filtered Image', frame_gray)
        flow = cv2.calcOpticalFlowFarneback(old_frame, frame_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=25,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        features.append(flow_to_feature_vector(flow))
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        """bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('Optical Flow', bgr)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            """
        # Update the previous frame and previous points
        old_frame = frame_gray.copy()
        
        
    
    
    cv2.destroyAllWindows()
    cap.release()
    return features

import pandas as pd
from tqdm import tqdm
df= pd.DataFrame()


# structure of the dataframe
data = []

for folder in os.listdir(os.path.dirname(__file__)):
    print(folder)
    if "." in folder:
        continue
    else:
        videos_path = os.path.join(os.path.dirname(__file__), folder)
        
        for video_path in tqdm(os.listdir(videos_path)):
            if video_path.endswith('.mp4') or video_path.endswith('.avi'):
                #print(video_path)
                video_path_full = os.path.join(os.path.dirname(__file__), folder, video_path)
                features = np.array(track_keypoints(video_path_full))
                
                mean_vector = np.mean(features, axis=0)
                std_vector = np.std(features, axis=0)
                max_vector = np.max(features, axis=0)
                min_vector = np.min(features, axis=0)
                median_vector = np.median(features, axis=0)
                # Concatenate all vectors
                features = np.concatenate((mean_vector, std_vector, max_vector, min_vector, median_vector))
                
                            
                
                data.append({'video_name': video_path_full, 'feature_vector': features, 'activity': folder})
                
        

df = pd.DataFrame(data)
# save as pickle
df.to_pickle('features_nolpass.pkl')                

