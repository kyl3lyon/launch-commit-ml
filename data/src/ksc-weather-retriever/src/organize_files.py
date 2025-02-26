import os
import re
from collections import defaultdict

# Function to extract year from filename
def extract_year(filename):
    match = re.search(r'fieldmill_data_(\d{4})-', filename)
    return match.group(1) if match else None

# Main function to organize files
def organize_files(directory):
    # Get all zip files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.zip')]
    
    # Group files by year
    files_by_year = defaultdict(list)
    for file in files:
        year = extract_year(file)
        if year:
            files_by_year[year].append(file)
    
    # Create year directories and move files
    for year, year_files in files_by_year.items():
        year_dir = os.path.join(directory, year)
        os.makedirs(year_dir, exist_ok=True)
        for file in year_files:
            old_path = os.path.join(directory, file)
            new_path = os.path.join(year_dir, file)
            os.rename(old_path, new_path)
        print(f"Moved {len(year_files)} files to {year_dir}")

# Run the organization script
if __name__ == "__main__":
    directory = "~/data/raw/field_mill_50hz"
    organize_files(os.path.expanduser(directory))
    print("File organization complete.")