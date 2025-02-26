import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ['PATH'] = f"{os.environ['JAVA_HOME']}/bin:{os.environ['PATH']}"

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime, timedelta, time

def create_spark_session():
    return SparkSession.builder \
        .appName("FieldMillDataProcessing") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.session.timeZone", "UTC") \
        .config("spark.master", "local[*]") \
        .config("spark.sql.shuffle.partitions", "32") \
        .getOrCreate()

def process_file(spark, file_path):
    try:
        # Read the file content
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Parse the base date and time from the header
        date_time = datetime.strptime(lines[0].strip(), "%m/%d/%Y %H:%M")
        base_hour = date_time.hour
        base_date = date_time.date()
        print(f"Base date and time: {date_time}")
        
        # Process the data lines
        data = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    hour_index = int(parts[0])
                    minute = int(parts[1])
                    second = int(parts[2])
                    electric_field = float(parts[3])
                    
                    # Calculate the actual hour
                    actual_hour = base_hour + (hour_index - 1)
                    
                    # Handle day rollover if hour exceeds 23
                    timestamp = datetime.combine(base_date, time(0, 0, 0)) + timedelta(
                        hours=actual_hour, minutes=minute, seconds=second
                    )
                    
                    data.append((timestamp, electric_field))
                except ValueError as ve:
                    print(f"Skipping line due to ValueError: {line.strip()} - {ve}")
            else:
                print(f"Skipping line due to insufficient data: {line.strip()}")
        
        if not data:
            print(f"No valid data in file {file_path}")
            return None
    
        # Create a DataFrame from the processed data
        df = spark.createDataFrame(data, ["DATETIME", "ELECTRIC_FIELD"])
        
        return df
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_month(spark, year, month):
    base_path = f"data/transformed/field_mill_50hz/{year}/{month:02d}"
    
    if not os.path.exists(base_path):
        print(f"No data found for {year}-{month:02d}")
        return None
    
    all_data = []
    total_files = 0
    processed_files = 0

    for day in sorted(os.listdir(base_path)):
        day_path = os.path.join(base_path, day)
        for hour_folder in sorted(os.listdir(day_path)):
            file_path = os.path.join(day_path, hour_folder, f"{hour_folder}.txt")
            total_files += 1
            df = process_file(spark, file_path)
            if df is not None:
                all_data.append(df)
                processed_files += 1
                print(f"Completed processing file: {file_path}")

    if not all_data:
        print(f"No data was successfully processed for {year}-{month:02d}")
        return None

    print(f"\nProcessed {processed_files} out of {total_files} files for {year}-{month:02d}.")

    # Union all DataFrames
    monthly_df = all_data[0]
    for df in all_data[1:]:
        monthly_df = monthly_df.union(df)
    
    # Create 20-minute intervals (1200 seconds)
    monthly_df = monthly_df.withColumn("interval_start",
        from_unixtime(floor(unix_timestamp("DATETIME") / 1200) * 1200).cast("timestamp")
    )

    # Aggregate data
    aggregated_df = monthly_df.groupBy("interval_start").agg(
        mean("ELECTRIC_FIELD").alias("MEAN_ELECTRIC_FIELD")
    )

    aggregated_df = aggregated_df.orderBy("interval_start")
    
    return aggregated_df


def process_year(spark, year):
    try:
        # Create the process folder if it doesn't exist
        os.makedirs(f"data/process/field_mill_data_{year}", exist_ok=True)
        
        for month in range(1, 13):
            aggregated_df = process_month(spark, year, month)
            if aggregated_df is not None:
                # Save the DataFrame for this month
                output_path = f"data/process/field_mill_data_{year}/month_{month:02d}"
                # Allow Spark to manage the number of output files
                aggregated_df.write.csv(output_path, header=True, mode="overwrite")
                print(f"Data for {year}-{month:02d} has been processed and saved to {output_path}")
            else:
                print(f"Skipping {year}-{month:02d} due to no data")
            
            # Clear cache to free up memory
            spark.catalog.clearCache()

        print(f"Processing complete for year {year}")
    except Exception as e:
        print(f"Error processing year {year}: {str(e)}")
        import traceback
        traceback.print_exc()

def get_available_years():
    base_path = "data/transformed/field_mill_50hz"
    return sorted([int(year) for year in os.listdir(base_path) if year.isdigit()])

# Create Spark session
spark = create_spark_session()

# Get all available years
years = get_available_years()
print("Years:", years)

# Process the data for all available years
for year in years:
    print(f"\nProcessing year {year}")
    process_year(spark, year)

# Stop the Spark session
spark.stop()