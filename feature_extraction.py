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
 







sift = cv2.SIFT_create()
def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return hog_features


# Step 3: Track the keypoint for every frame
def track_keypoints(video_path):
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    ##fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    ##out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (old_frame.shape[1], old_frame.shape[0]))
    if not ret:
        print("Error: Cannot read video file.")
        return []

    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros((old_frame.shape[0], old_frame.shape[1], 3), dtype=np.uint8)

    hsv[..., 1] = 255
    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        flow = cv2.calcOpticalFlowFarneback(old_frame, frame_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=25,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        hog_features = extract_hog_features(frame_gray)
        feature_vector = np.hstack([flow_to_feature_vector(flow), hog_features])
        features.append(feature_vector)

        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        """bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        cv2.imshow('Optical Flow', bgr)
        out.write(bgr)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break"""
            
        # Update the previous frame and previous points
        old_frame = frame_gray.copy()
        
        
    
    #out.release()
    #cv2.destroyAllWindows()
    cap.release()
    return features

import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from concurrent.futures import ProcessPoolExecutor, as_completed
df= pd.DataFrame()


# structure of the dataframe
data = []
                
        
def process_video(args):
    folder, video_path = args
    video_path_full = os.path.join(os.path.dirname(__file__), folder, video_path)
    features = np.array(track_keypoints(video_path_full))
    mean_vector = np.mean(features, axis=0)
    std_vector = np.std(features, axis=0)
    max_vector = np.max(features, axis=0)
    min_vector = np.min(features, axis=0)
    median_vector = np.median(features, axis=0)
    feature_vector = np.concatenate((mean_vector, std_vector, max_vector, min_vector, median_vector))
    return {'video_name': video_path_full, 'feature_vector': feature_vector, 'activity': folder}



if __name__ == "__main__":
    video_tasks = []
    for folder in os.listdir(os.path.dirname(__file__)):
        if "." in folder or 'Makefile' in folder  :
            continue
        videos_path = os.path.join(os.path.dirname(__file__), folder)
        for video_path in os.listdir(videos_path):
            if video_path.endswith('.mp4') or video_path.endswith('.avi'):
                video_tasks.append((folder, video_path))
    data = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, args) for args in video_tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                data.append(result)
            except Exception as e:
                print(f"Error processing a video: {e}")


    df = pd.DataFrame(data)
    # save as pickle
    df.to_pickle('features_hog_hof.pkl')

