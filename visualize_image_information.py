import pandas as pd
import json
import os
import argparse
from image_visualizer import visualize_distribution

def main():
    parser = argparse.ArgumentParser(description="Visualize image quality metrics")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    input_csv = config['input_csv']
    output_dir = config['output_dir']
    visualizations = config['visualizations']

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the DataFrame
    df = pd.read_csv(input_csv)

    # Visualize and save distributions
    for viz in visualizations:
        column = viz['column']
        title = viz['title']
        xlabel = viz['xlabel']
        ylabel = viz['ylabel']
        output_file = os.path.join(output_dir, viz['output_file'])

        # Handle resolution separately since it's stored as a string
        if column == 'resolution':
            df['resolution_pixels'] = df[column].apply(lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
            data = df['resolution_pixels']
        else:
            data = df[column]

        visualize_distribution(data, title, xlabel, ylabel, output_file)

if __name__ == "__main__":
    main()