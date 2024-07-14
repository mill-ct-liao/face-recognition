import os

# Function to ensure the target directory exists
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
