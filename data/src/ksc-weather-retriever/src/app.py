# -- Imports --
import streamlit as st
import os
import sys
from download_ksc_files import download
from open_zip_files import extract_all_zips

# -- Functions --
def get_summary():
    raw_size = sum(os.path.getsize(os.path.join("data/raw/field_mill_50hz", f)) for f in os.listdir("data/raw/field_mill_50hz") if os.path.isfile(os.path.join("data/raw/field_mill_50hz", f)))
    
    transformed_size = 0
    file_count = 0
    for root, dirs, files in os.walk("data/transformed/field_mill_50hz"):
        for file in files:
            file_path = os.path.join(root, file)
            transformed_size += os.path.getsize(file_path)
            file_count += 1
    
    return {
        "raw_file_count": len(os.listdir('data/raw/field_mill_50hz')),
        "raw_size_gb": raw_size / (1024*1024*1024),
        "extracted_file_count": file_count,
        "extracted_size_gb": transformed_size / (1024*1024*1024)
    }

# -- Streamlit Execution --
st.title("KSC Field Mill Data Processor")

if st.button("Download Files"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    print("Starting download process...", file=sys.stderr)
    download(progress_bar, status_text)
    print("Download process completed.", file=sys.stderr)
    st.success("Download complete!")

if st.button("Extract Files"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    print("Starting extraction process...", file=sys.stderr)
    
    files = os.listdir("data/raw/field_mill_50hz")
    total_files = len(files)
    
    for i, file in enumerate(files):
        message = f"Extracting {file}..."
        status_text.text(message)
        print(message, file=sys.stderr)
        extract_all_zips("data/raw/field_mill_50hz", "data/transformed/field_mill_50hz")
        progress = int(((i+1) / total_files) * 100)
        progress_bar.progress(progress)
    
    message = "Extraction complete!"
    status_text.text(message)
    print(message, file=sys.stderr)
    st.success("Extraction complete!")

if st.button("Show Summary"):
    summary = get_summary()
    st.write("Summary:")
    st.write(f"Total number of raw zip files: {summary['raw_file_count']}")
    st.write(f"Total size of raw zip files: {summary['raw_size_gb']:.2f} GB")
    st.write(f"Total number of extracted files: {summary['extracted_file_count']}")
    st.write(f"Total size of extracted data: {summary['extracted_size_gb']:.2f} GB")