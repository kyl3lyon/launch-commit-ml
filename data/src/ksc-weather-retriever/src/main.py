# -- Imports --
import os
from download_ksc_files import download
from open_zip_files import extract_all_zips

# -- Function --
def main():
    # Step 1: Download all files
    print("Downloading files...")
    download()

    # Step 2: Extract all zip files
    print("\nExtracting files...")
    extract_all_zips("data/raw/field_mill_50hz", "data/transformed/field_mill_50hz")

    # Step 3: Print summary
    print("\nSummary:")
    raw_size = sum(os.path.getsize(os.path.join("data/raw/field_mill_50hz", f)) for f in os.listdir("data/raw/field_mill_50hz") if os.path.isfile(os.path.join("data/raw/field_mill_50hz", f)))
    
    transformed_size = 0
    file_count = 0
    for root, dirs, files in os.walk("data/transformed/field_mill_50hz"):
        for file in files:
            file_path = os.path.join(root, file)
            transformed_size += os.path.getsize(file_path)
            file_count += 1
    
    print(f"Total number of raw zip files: {len(os.listdir('data/raw/field_mill_50hz'))}")
    print(f"Total size of raw zip files: {raw_size / (1024*1024*1024):.2f} GB")
    print(f"Total number of extracted files: {file_count}")
    print(f"Total size of extracted data: {transformed_size / (1024*1024*1024):.2f} GB")

# -- Execution --
if __name__ == "__main__":
    main()