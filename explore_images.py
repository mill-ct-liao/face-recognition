import os
import cv2
import numpy as np
import pandas as pd
import json
import argparse
from image_reader import load_dataset, compute_images_quality_metrics, check_images_quality

def main():
    parser = argparse.ArgumentParser(description="Check image quality")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_path = config['dataset_path']
    csv_path = config['csv_path']
    laplacian_variance_low = config['laplacian_variance_low']
    laplacian_variance_high = config['laplacian_variance_high']
    mean_intensity_low = config['mean_intensity_low']
    mean_intensity_high = config['mean_intensity_high']
    standard_deviation_low = config['standard_deviation_low']
    standard_deviation_high = config['standard_deviation_high']
    min_resolution = config['min_resolution']
    
    # Print thresholds
    print("Thresholds:")
    print(f"Laplacian Variance Low: {laplacian_variance_low}")
    print(f"Laplacian Variance High: {laplacian_variance_high}")
    print(f"Mean Intensity Low: {mean_intensity_low}")
    print(f"Mean Intensity High: {mean_intensity_high}")
    print(f"Standard Deviation Low: {standard_deviation_low}")
    print(f"Standard Deviation High: {standard_deviation_high}")
    print(f"Minimum Resolution: {min_resolution}")

    # Load dataset
    data, labels, image_names = load_dataset(dataset_path)
    
    # Compute image quality metrics
    df = compute_images_quality_metrics(data, labels, image_names)
    
    # Check images quality
    df = check_images_quality(df, laplacian_variance_low, laplacian_variance_high, mean_intensity_low, mean_intensity_high, standard_deviation_low, standard_deviation_high, min_resolution)
    
    # Add a new column to indicate low-quality images
    df['low_quality'] = df[['too_blurry', 'too_noisy_laplacian', 'too_dark', 'too_bright', 'lack_of_texture', 'too_noisy_stddev', 'low_resolution']].max(axis=1)

    # Compute the number of low-quality images
    num_low_quality_images = df['low_quality'].sum()

    print("#Low quality images: {}/{}".format(num_low_quality_images, len(df)))
    print("%Low quality images: {:.2f}%".format(num_low_quality_images / len(df) * 100))

    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    print("Save {}".format(csv_path))

if __name__ == "__main__":
    main()