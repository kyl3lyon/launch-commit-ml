# -- Imports --
import os
import json
import requests
import time
import streamlit as st
import sys
from tqdm import tqdm
import hashlib
import concurrent.futures

# -- Functions --
def download_large_file(url, filename, chunk_size=8192):
    """
    Download a large file in chunks, with progress bar and resumption capability.

    Args:
    url (str): The URL of the file to download.
    filename (str): The name to save the file as.
    chunk_size (int): Size of chunks to download at a time.

    Returns:
    str: Path to the downloaded file.
    """
    file_path = os.path.join(os.getcwd(), filename)

    # Check if the file already exists and get its size
    if os.path.exists(file_path):
        first_byte = os.path.getsize(file_path)
    else:
        first_byte = 0

    # Set up the request headers for resuming download
    headers = {"Range": f"bytes={first_byte}-"}

    # Start the download
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

    # Get the total file size
    total_size = int(response.headers.get('content-length', 0)) + first_byte

    # Open the file in binary append mode
    mode = 'ab' if first_byte > 0 else 'wb'
    with open(file_path, mode) as file, tqdm(
        desc=filename,
        initial=first_byte,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            progress_bar.update(size)

    return file_path

def calculate_md5(file_path, chunk_size=8192):
    """
    Calculate MD5 hash of a large file.

    Args:
    file_path (str): Path to the file.
    chunk_size (int): Size of chunks to read at a time.

    Returns:
    str: MD5 hash of the file.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def download_file(date, url, base_url):
    filename = f"fieldmill_data_{date}.zip"
    full_url = base_url + url
    file_path = os.path.join("data/raw/field_mill_50hz", filename)

    if os.path.exists(file_path):
        print(f"\n{filename} already exists. Skipping download.")
        return

    print(f"\nDownloading {filename}...")
    file_path = download_large_file(full_url, file_path)

    # Calculate and print MD5 hash
    md5_hash = calculate_md5(file_path)
    print(f"MD5 hash of {filename}: {md5_hash}")

    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")


def download(progress_bar, status_text):
    # Load the JSON data
    with open("data/field_mill.json") as f:
        data = json.load(f)

    # Create a directory to store the downloaded files
    os.makedirs("data/raw/field_mill_50hz", exist_ok=True)

    BASE_URL = "https://kscweather.ksc.nasa.gov"

    total_files = len(data)
    downloaded_files = 0

    # Download files one at a time
    for date, url in data.items():
        filename = f"fieldmill_data_{date}.zip"
        file_path = os.path.join("data/raw/field_mill_50hz", filename)

        if os.path.exists(file_path):
            message = f"\n{filename} already exists. Skipping download."
            status_text.text(message)
            print(message, file=sys.stderr)
        else:
            message = f"Downloading {filename}..."
            status_text.text(message)
            print(message, file=sys.stderr)
            download_file(date, url, BASE_URL)
            message = "Waiting 10 seconds before the next download..."
            status_text.text(message)
            print(message, file=sys.stderr)
            time.sleep(10)

        downloaded_files += 1
        progress = int((downloaded_files / total_files) * 100)
        progress_bar.progress(progress)
        message = f"Processed {downloaded_files} of {total_files} files"
        status_text.text(message)
        print(message, file=sys.stderr)

    message = "Download process complete!"
    status_text.text(message)
    print(message, file=sys.stderr)

if __name__ == "__main__":
    class DummyProgressBar:
        def progress(self, value):
            pass

    class DummyStatusText:
        def text(self, value):
            print(value)

    download(DummyProgressBar(), DummyStatusText())
    download()