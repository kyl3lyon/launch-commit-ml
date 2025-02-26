import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob

def load_and_process_json(file_path):
    """Load and process a single JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract data and units
    hourly_data = data.get('hourly', {})
    minutely_data = data.get('minutely_15', {})
    hourly_units = data.get('hourly_units', {})
    minutely_units = data.get('minutely_15_units', {})

    if not hourly_data and not minutely_data:
        print(f"Warning: No data found in {file_path}")
        return None, None

    # Process hourly data
    hourly_df = pd.DataFrame(hourly_data)
    if 'time' in hourly_df.columns:
        hourly_df['time'] = pd.to_datetime(hourly_df['time'])
        hourly_df = hourly_df.set_index('time')

    # Process minutely data if it exists
    if minutely_data:
        minutely_df = pd.DataFrame(minutely_data)
        if 'time' in minutely_df.columns:
            minutely_df['time'] = pd.to_datetime(minutely_df['time'])

            # Group minutely data by hour and convert to arrays
            grouped_minutely = {}
            for col in minutely_df.columns:
                if col != 'time':
                    grouped = minutely_df.groupby(minutely_df['time'].dt.floor('H'))[col].agg(list)
                    grouped_minutely[f"{col}_15min"] = grouped

            # Convert grouped data to DataFrame
            minutely_grouped_df = pd.DataFrame(grouped_minutely)
            minutely_grouped_df.index.name = 'time'

            # Merge hourly and minutely data
            combined_df = pd.concat([hourly_df, minutely_grouped_df], axis=1)
        else:
            combined_df = hourly_df
    else:
        combined_df = hourly_df

    # Handle special case for 'cape' which exists in both granularities
    if 'cape' in combined_df.columns and 'cape_15min' in combined_df.columns:
        combined_df = combined_df.rename(columns={
            'cape': 'cape',
            'cape_15min': 'cape_15min'
        })

    return combined_df, {**hourly_units, **minutely_units}

def combine_open_meteo_data(base_dir):
    """Combine all Open-Meteo JSON files into a single DataFrame."""
    json_files = glob.glob(os.path.join(base_dir, '**/*.json'), recursive=True)
    json_files.sort()

    all_data = []
    units = None

    print(f"Processing {len(json_files)} files...")

    for file_path in json_files:
        try:
            df, file_units = load_and_process_json(file_path)
            if df is not None:
                all_data.append(df)
                if units is None:
                    units = file_units
                print(f"✓ Processed {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(file_path)}: {e}")

    if not all_data:
        raise ValueError("No valid data found in any of the files")

    print("\nCombining all data...")
    combined_df = pd.concat(all_data)

    # Sort by time and remove duplicates
    combined_df = combined_df.sort_index().loc[~combined_df.index.duplicated(keep='first')]

    # Save units to a separate JSON file
    units_path = os.path.join(os.path.dirname(base_dir), 'open_meteo_units.json')
    with open(units_path, 'w') as f:
        json.dump(units, f, indent=2)

    return combined_df

def main():
    base_dir = "data/raw/weather/open-meteo"
    output_dir = "data/transformed/weather"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting data processing...")
    combined_df = combine_open_meteo_data(base_dir)

    # Save to CSV format
    output_path = os.path.join(output_dir, 'open_meteo_combined.csv')
    combined_df.to_csv(output_path)
    print(f"\nData saved to {output_path}")

    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total records: {len(combined_df):,}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Number of columns: {len(combined_df.columns)}")

    # Print column types summary
    hourly_cols = [col for col in combined_df.columns if not col.endswith('_15min')]
    minutely_cols = [col for col in combined_df.columns if col.endswith('_15min')]
    print(f"\nHourly features: {len(hourly_cols)}")
    print(f"15-minute features: {len(minutely_cols)}")

if __name__ == "__main__":
    main()