# -- Imports --
import os
import zipfile
from tqdm import tqdm
import hashlib

# -- Functions --
def calculate_md5(file_path, chunk_size=8192):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def check_zip_file(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test ZIP file integrity
            test_result = zip_ref.testzip()
            if test_result is not None:
                return f"Corrupt file detected: {test_result}"
            
            # Check if the ZIP file is empty
            if len(zip_ref.namelist()) == 0:
                return "ZIP file is empty"
        
        # Calculate and return MD5 hash
        return calculate_md5(zip_path)
    except zipfile.BadZipFile:
        return "Not a valid ZIP file"
    except Exception as e:
        return f"Error checking ZIP file: {str(e)}"

def extract_nested_zips(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc=f"Extracting {os.path.basename(zip_path)}"):
                zip_ref.extract(file, extract_to)
                if file.endswith('.zip'):
                    nested_zip_path = os.path.join(extract_to, file)
                    nested_extract_to = os.path.join(extract_to, os.path.splitext(file)[0])
                    extract_nested_zips(nested_zip_path, nested_extract_to)
                    os.remove(nested_zip_path)  # Remove the zip file after extraction
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file. Skipping.")
    except Exception as e:
        print(f"Error extracting {zip_path}: {str(e)}. Skipping.")

def extract_all_zips(source_folder, destination_folder):
    error_files = []
    for year_folder in os.listdir(source_folder):
        year_path = os.path.join(source_folder, year_folder)
        if not os.path.isdir(year_path):
            continue
        
        for filename in os.listdir(year_path):
            if filename.endswith('.zip'):
                date = filename.split('_')[-1].split('.')[0]
                year, month = date.split('-')
                
                zip_path = os.path.join(year_path, filename)
                extract_to = os.path.join(destination_folder, year, month)
                
                # Check ZIP file integrity
                zip_check_result = check_zip_file(zip_path)
                if zip_check_result != calculate_md5(zip_path):
                    print(f"Error with {filename}: {zip_check_result}")
                    error_files.append((filename, zip_check_result))
                    continue
                
                # Check if already extracted
                if os.path.exists(extract_to) and os.listdir(extract_to):
                    print(f"Skipping {filename}: Already extracted.")
                    continue
                
                os.makedirs(extract_to, exist_ok=True)
                extract_nested_zips(zip_path, extract_to)
    
    return error_files

if __name__ == "__main__":
    error_files = extract_all_zips("data/raw/field_mill_50hz", "data/transformed/field_mill_50hz")

    # Print summary of extracted files
    print("\nExtracted files:")
    for root, dirs, files in os.walk("data/transformed/field_mill_50hz"):
        for file in files:
            print(os.path.join(root, file))

    # Print summary of error files
    if error_files:
        print("\nFiles with errors:")
        for filename, error in error_files:
            print(f"{filename}: {error}")