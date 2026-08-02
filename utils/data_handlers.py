import numpy as np
from pathlib import Path
import os
import csv

def load_csv_data(path: str):
    """
    Universal loader for csv or npy data.
    Returns: (t_data, Intensity_data)
    """
    try:
        p = Path(path)
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
        else:
            if p.suffix.lower() == ".csv":
                with open(p, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    arr = np.array([row for row in reader], dtype=float)
            else:
                arr = np.loadtxt(p, delimiter=",")
        if arr.ndim == 1:
            return np.arange(len(arr)), arr
        return arr[:, 0], arr[:, 1]
    except Exception as e:
        print(f"File load error: {e}")
        return None, None
    
    
def load_video_images_data(path: str):
    """
    Universal loader for folders with image sequences.
    images are expected to be in .tiff or .png format.
    Returns: 3D array (frame, x, y)
    """
    from PIL import Image
    try:
        p = Path(path)
        if not p.is_dir():
            print(f"Provided path is not a directory: {path}")
            return None
        
        image_files = sorted([f for f in p.iterdir() if f.suffix.lower() in ['.tiff', '.tif', '.png']])
        if not image_files:
            print(f"No image files found in directory: {path}")
            return None
        
        images = []
        for img_file in image_files:
            img = Image.open(img_file)
            images.append(np.array(img))
        
        # Stack images into a 3D numpy array
        video_data = np.stack(images, axis=0)  # Shape: (num_frames, height, width)
        
        return video_data
    except Exception as e:
        print(f"Video load error: {e}")
        return None