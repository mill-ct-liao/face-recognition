import os
import cv2
import numpy as np
import pandas as pd

def load_dataset(dataset_path):
    data = []
    labels = []
    image_names = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                if image_path.endswith('.jpg'):  # Ensure it's a jpg file
                    image = cv2.imread(image_path)
                    if image is not None:
                        data.append(image)
                        labels.append(int(folder_name))
                        image_names.append(image_name)
    return data, labels, image_names

def compute_image_quality_metrics(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    std_dev = np.std(gray) # noisy
    mean = np.mean(gray) # brightness
    height, width = image.shape[:2]
    resolution = (height, width)
    return variance_laplacian, mean, std_dev, resolution

def compute_images_quality_metrics(data, labels, image_names):
    # Initialize a list to store the metrics
    metrics = []
    # Compute metrics for each image
    for img, label, image_name in zip(data, labels, image_names):
        variance_laplacian, mean, std_dev, resolution = compute_image_quality_metrics(img)
        metrics.append({
            'image_name': image_name,
            'label': label,
            'laplacian_variance': variance_laplacian,
            'mean': mean,
            'standard_deviation': std_dev,
            'resolution': f"{resolution[0]}x{resolution[1]}",
        })

    return pd.DataFrame(metrics)

# Function to check quality
def check_image_quality(row, laplacian_variance_low, laplacian_variance_high, mean_intensity_low, mean_intensity_high, standard_deviation_low, standard_deviation_high, min_resolution):
    resolution = int(row['resolution'].split('x')[0]) * int(row['resolution'].split('x')[1])
    if row['laplacian_variance'] < laplacian_variance_low:
        row['too_blurry'] = 1
    if row['laplacian_variance'] > laplacian_variance_high:
        row['too_noisy_laplacian'] = 1
    if row['mean'] < mean_intensity_low:
        row['too_dark'] = 1
    if row['mean'] > mean_intensity_high:
        row['too_bright'] = 1
    if row['standard_deviation'] < standard_deviation_low:
        row['lack_of_texture'] = 1
    if row['standard_deviation'] > standard_deviation_high:
        row['too_noisy_stddev'] = 1
    if resolution < min_resolution:
        row['low_resolution'] = 1
    return row

def check_images_quality(df, laplacian_variance_low, laplacian_variance_high, mean_intensity_low, mean_intensity_high, standard_deviation_low, standard_deviation_high, min_resolution):
    # Initialize new columns with default value 0
    df['too_blurry'] = 0
    df['too_noisy_laplacian'] = 0
    df['too_dark'] = 0
    df['too_bright'] = 0
    df['lack_of_texture'] = 0
    df['too_noisy_stddev'] = 0
    df['low_resolution'] = 0
    return df.apply(check_image_quality, axis=1, args=(laplacian_variance_low, laplacian_variance_high, mean_intensity_low, mean_intensity_high, standard_deviation_low, standard_deviation_high, min_resolution))