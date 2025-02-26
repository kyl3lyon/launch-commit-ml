import os
from lightning_sdk import Studio

# Initialize the Studio
studio = Studio(name="LCC", teamspace="Silicon Mountain", user="kyl3lyon")

# Set the local root directory
local_root = "data/raw"

# Set the remote root directory
remote_root = "/teamspace/uploads"

# Walk through the directory
for root, dirs, files in os.walk(local_root):
    for file in files:
        # Calculate the relative path
        rel_path = os.path.relpath(root, local_root)
        # Construct the remote path
        remote_path = os.path.normpath(os.path.join(remote_root, rel_path, file))
        # Construct the local file path
        local_file_path = os.path.join(root, file)
        
        # Upload the file
        print(f"Uploading {local_file_path} to {remote_path}")
        try:
            studio.upload_file(local_file_path, remote_path)
            print(f"Successfully uploaded {file}")
        except Exception as e:
            print(f"Failed to upload {file}. Error: {str(e)}")

print("Upload process completed!")