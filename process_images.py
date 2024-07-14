import pandas as pd
import numpy as np
import cv2
import json
import os
import argparse
from tqdm import tqdm
from image_processor import process_image
from utils import ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Image processing")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    dataset_path = config['dataset_path']   
    csv_path = config['csv_path']
    processed_path = config['processed_path']
    laplacian_variance_low = config['laplacian_variance_low']
    laplacian_variance_high = config['laplacian_variance_high']
    mean_intensity_low = config['mean_intensity_low']
    mean_intensity_high = config['mean_intensity_high']
    standard_deviation_low = config['standard_deviation_low']
    standard_deviation_high = config['standard_deviation_high']
    min_resolution = config['min_resolution']
    
    df = pd.read_csv(csv_path)
    # Process and save images
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        label = row['label']
        image_name = row['image_name']
        image_path = os.path.join(dataset_path, str(label), image_name)
        processed_label_path = os.path.join(processed_path, str(label))

        ensure_dir(processed_label_path)

        image = cv2.imread(image_path)
        if image is not None:
            issues = row[['too_blurry', 'too_noisy_laplacian', 'too_dark', 'too_bright', 'lack_of_texture', 'too_noisy_stddev', 'low_resolution']]
            processed_image = process_image(image, issues)
            processed_image_path = os.path.join(processed_label_path, image_name)
            cv2.imwrite(processed_image_path, processed_image)

    print("Image processing and saving completed to {}.".format(processed_path))

if __name__ == "__main__":
    main()
