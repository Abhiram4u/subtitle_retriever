import os
import sqlite3
import zipfile
import io
import pandas as pd
import re

# Connect to the SQLite database
db_path = "eng_subtitles_database.db"  # Ensure this file exists in the same directory
con = sqlite3.connect(db_path)

# Read the zipfile table from the database
df = pd.read_sql_query("SELECT * FROM zipfiles", con)

# Function to decode subtitle content from binary zip data
def decode_method(binary_data):
    with io.BytesIO(binary_data) as f:
        with zipfile.ZipFile(f, "r") as zip_file:
            subtitle_content = zip_file.read(zip_file.namelist()[0])  # Extract the first file
    
    return subtitle_content.decode("latin-1")

# Function to clean subtitle text (removes timestamps and unwanted text)
def clean_subtitles(subtitle_text):
    # Remove timestamps (format: 00:00:00,000 --> 00:00:05,000)
    subtitle_text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", subtitle_text)
    # Remove empty lines and extra spaces
    subtitle_text = "\n".join([line.strip() for line in subtitle_text.split("\n") if line.strip()])
    return subtitle_text

# Decode and clean subtitles
df["file_content"] = df["content"].apply(decode_method).apply(clean_subtitles)

# Create dataset folder if not exists
os.makedirs("dataset", exist_ok=True)

# Function to save cleaned subtitles to .srt files
def generate_subtitles(start_range, end_range):
    file_no = start_range
    for data, movie_name in zip(df["file_content"][start_range:end_range], df["name"][start_range:end_range]):
        file_path = f"dataset/{movie_name}_subtitle_cleaned.srt"
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(data)
        print(f"File {file_no} written: {file_path}")
        file_no += 1
    return True

# Example: Extract and clean the first 10 subtitles
generate_subtitles(0, 10)
