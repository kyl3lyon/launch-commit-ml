import requests
import os
from datetime import datetime, timedelta
import json

def fetch_weather_data():
    print("Starting weather data collection...")
    # Base URL and parameters
    base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 28.4058,
        "longitude": -80.6048,
        "minutely_15": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,visibility,cape,lightning_potential,is_day",
        "hourly": "uv_index,sunshine_duration,cape,lifted_index,convective_inhibition,freezing_level_height,"
                 "temperature_1000hPa,temperature_975hPa,temperature_950hPa,temperature_925hPa,temperature_900hPa,"
                 "temperature_850hPa,temperature_800hPa,temperature_700hPa,temperature_600hPa,temperature_500hPa,"
                 "temperature_400hPa,temperature_300hPa,temperature_250hPa,temperature_200hPa,temperature_150hPa,"
                 "temperature_100hPa,temperature_70hPa,temperature_50hPa,temperature_30hPa,relative_humidity_1000hPa,"
                 "relative_humidity_975hPa,relative_humidity_950hPa,relative_humidity_925hPa,relative_humidity_900hPa,"
                 "relative_humidity_850hPa,relative_humidity_800hPa,relative_humidity_700hPa,relative_humidity_600hPa,"
                 "relative_humidity_500hPa,relative_humidity_400hPa,relative_humidity_300hPa,relative_humidity_250hPa,"
                 "relative_humidity_200hPa,relative_humidity_150hPa,relative_humidity_100hPa,relative_humidity_70hPa,"
                 "relative_humidity_50hPa,relative_humidity_30hPa,cloud_cover_1000hPa,cloud_cover_975hPa,cloud_cover_950hPa,"
                 "cloud_cover_925hPa,cloud_cover_900hPa,cloud_cover_850hPa,cloud_cover_800hPa,cloud_cover_700hPa,"
                 "cloud_cover_600hPa,cloud_cover_500hPa,cloud_cover_400hPa,cloud_cover_300hPa,cloud_cover_250hPa,"
                 "cloud_cover_200hPa,cloud_cover_150hPa,cloud_cover_100hPa,cloud_cover_70hPa,cloud_cover_50hPa,"
                 "cloud_cover_30hPa,wind_speed_1000hPa,wind_speed_975hPa,wind_speed_950hPa,wind_speed_925hPa,"
                 "wind_speed_900hPa,wind_speed_850hPa,wind_speed_800hPa,wind_speed_700hPa,wind_speed_600hPa,"
                 "wind_speed_500hPa,wind_speed_400hPa,wind_speed_300hPa,wind_speed_250hPa,wind_speed_200hPa,"
                 "wind_speed_150hPa,wind_speed_100hPa,wind_speed_70hPa,wind_speed_50hPa,wind_speed_30hPa,"
                 "wind_direction_1000hPa,wind_direction_975hPa,wind_direction_950hPa,wind_direction_925hPa,"
                 "wind_direction_900hPa,wind_direction_850hPa,wind_direction_800hPa,wind_direction_700hPa,"
                 "wind_direction_600hPa,wind_direction_500hPa,wind_direction_400hPa,wind_direction_300hPa,"
                 "wind_direction_250hPa,wind_direction_200hPa,wind_direction_150hPa,wind_direction_100hPa,"
                 "wind_direction_70hPa,wind_direction_50hPa,wind_direction_30hPa",

        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }

    print("Creating directory structure...")
    # Create base directory for data storage
    base_dir = "data/raw/weather/open-meteo"
    os.makedirs(base_dir, exist_ok=True)

    # Start date and end date
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)

    print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    current_date = start_date
    while current_date <= end_date:
        # Calculate the first day of next month
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)

        # Adjust end date to be the last day of current month
        month_end = next_month - timedelta(days=1)

        # Create year and month directories
        year_dir = os.path.join(base_dir, str(current_date.year))
        os.makedirs(year_dir, exist_ok=True)

        # Check if file already exists
        filename = f"{current_date.strftime('%Y_%m')}.json"
        filepath = os.path.join(year_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Skipping existing file for {current_date.strftime('%Y-%m')}")
        else:
            print(f"\nProcessing month: {current_date.strftime('%Y-%m')}")
            # Update parameters with current date range
            params["start_date"] = current_date.strftime("%Y-%m-%d")
            params["end_date"] = month_end.strftime("%Y-%m-%d")

            # Make API request
            try:
                print("Making API request...")
                response = requests.get(base_url, params=params)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Save the response to a file
                with open(filepath, 'w') as f:
                    json.dump(response.json(), f, indent=2)

                print(f"✓ Successfully saved data for {current_date.strftime('%Y-%m')}")

            except requests.exceptions.RequestException as e:
                print(f"✗ Error fetching data for {current_date.strftime('%Y-%m')}: {e}")

        # Move to next month
        current_date = next_month

    print("\nWeather data collection completed!")

if __name__ == "__main__":
    fetch_weather_data()
