import glob
import os
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm
import cv2


image_paths = glob.glob("./road-waymo/rgb-images/*/*")
hog_paths = [path.replace(".jpg", ".npy") for path in image_paths]
hog_paths = [path.replace("rgb-images", "hog-images") for path in hog_paths]

for image_path in tqdm(image_paths, desc = "Extract HOG"):

    # reading the image
    img = imread(image_path)
    img = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)
    
    # Extract HOG feature descriptor and visualization
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True, feature_vector=False, channel_axis=-1)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # make path for hog image visualizations
    hog_image_path = image_path.replace("rgb-images", "hog-images")
    hog_image_dirname = os.path.dirname(hog_image_path)
    Path(hog_image_dirname).mkdir(parents=True, exist_ok = True)
    plt.imsave(hog_image_path, hog_image_rescaled, cmap = "gray")

    # make path for hog feature descriptors
    hog_fd_path = image_path.replace(".jpg", ".npy").replace("rgb-images", "hog-fd")
    hog_fd_dirname = os.path.dirname(hog_fd_path)
    Path(hog_fd_dirname).mkdir(parents=True, exist_ok = True)
    # print(fd.shape)
    np.save(hog_fd_path, fd)