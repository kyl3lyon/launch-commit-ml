# -- Imports --
import os
import zipfile
import hashlib
import magic
from tqdm import tqdm

# -- Functions --
def calculate_md5(file_path, chunk_size=8192):
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def get_file_info(file_path):
    try:
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Get file type using magic numbers
        file_type = magic.from_file(file_path)
        
        # Read first few bytes to check file signature
        with open(file_path, 'rb') as f:
            first_bytes = f.read(8).hex()
        
        return f"Size: {file_size} bytes, Type: {file_type}, First bytes: {first_bytes}"
    except Exception as e:
        return f"Error getting file info: {str(e)}"

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
        
        # Calculate MD5 hash
        md5_hash = calculate_md5(zip_path)
        
        return f"Valid ZIP file. MD5: {md5_hash}"
    except zipfile.BadZipFile:
        return f"Not a valid ZIP file. {get_file_info(zip_path)}"
    except Exception as e:
        return f"Error checking ZIP file: {str(e)}. {get_file_info(zip_path)}"

def check_all_zips(folder_path):
    results = {}
    for filename in tqdm(os.listdir(folder_path), desc="Checking ZIP files"):
        if filename.endswith('.zip'):
            file_path = os.path.join(folder_path, filename)
            results[filename] = check_zip_file(file_path)
    return results

def delete_invalid_zips(results, folder_path):
    invalid_files = [filename for filename, result in results.items() if not result.startswith("Valid")]
    
    if not invalid_files:
        print("No invalid files to delete.")
        return

    print(f"\nFound {len(invalid_files)} invalid files.")
    confirm = input("Do you want to delete these files? (yes/no): ").lower().strip()
    
    if confirm != 'yes':
        print("Deletion cancelled.")
        return

    for filename in invalid_files:
        file_path = os.path.join(folder_path, filename)
        
        # Delete the file
        os.remove(file_path)
        print(f"Deleted {filename}")

    print(f"\nAll invalid files have been deleted.")

# -- Main Execution --
if __name__ == "__main__":
    folder_path = "data/raw/field_mill_50hz"
    results = check_all_zips(folder_path)
    
    print("\nResults:")
    for filename, result in results.items():
        print(f"{filename}: {result}")
    
    # Count and display the number of valid and invalid files
    valid_count = sum(1 for result in results.values() if result.startswith("Valid"))
    invalid_count = len(results) - valid_count
    
    print(f"\nTotal files checked: {len(results)}")
    print(f"Valid ZIP files: {valid_count}")
    print(f"Invalid or problematic files: {invalid_count}")

    # Delete invalid files
    delete_invalid_zips(results, folder_path)