#!/usr/bin/env python3
"""
Script to format time values in CSV files to have exactly 5 decimal places.
"""

import os
import csv
import re
from pathlib import Path

def is_time_value(value):
    """Check if a value looks like a time measurement (float)."""
    try:
        float_val = float(value)
        return True
    except ValueError:
        return False

def format_time_in_row(row, headers):
    """Format time values in a CSV row to 5 decimal places."""
    formatted_row = []
    
    for i, cell in enumerate(row):
        # Check if this column might contain time data
        if headers and i < len(headers):
            header = headers[i].lower()
            # Look for time-related column names
            if any(keyword in header for keyword in ['time', 'seconds', 'duration']):
                if is_time_value(cell):
                    try:
                        time_val = float(cell)
                        formatted_row.append(f"{time_val:.5f}")
                    except ValueError:
                        formatted_row.append(cell)
                else:
                    formatted_row.append(cell)
            else:
                formatted_row.append(cell)
        else:
            # For files without headers, check if the last column looks like time
            # and if it's a float value
            if i == len(row) - 1 and is_time_value(cell):
                try:
                    time_val = float(cell)
                    formatted_row.append(f"{time_val:.5f}")
                except ValueError:
                    formatted_row.append(cell)
            else:
                formatted_row.append(cell)
    
    return formatted_row

def process_csv_file(file_path):
    """Process a single CSV file to format time values."""
    print(f"Processing: {file_path}")
    
    # Read the original file
    rows = []
    headers = None
    
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Try to detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.reader(csvfile, delimiter=delimiter)
            
            for i, row in enumerate(reader):
                if i == 0:
                    # Check if first row contains headers
                    if any(keyword in str(cell).lower() for cell in row 
                          for keyword in ['time', 'filename', 'instance', 'variable', 'constraint']):
                        headers = row
                        rows.append(row)  # Keep headers as-is
                    else:
                        # No headers, process as data
                        formatted_row = format_time_in_row(row, None)
                        rows.append(formatted_row)
                else:
                    # Process data rows
                    formatted_row = format_time_in_row(row, headers)
                    rows.append(formatted_row)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Write the updated data back
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        print(f"✓ Successfully updated {file_path}")
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def main():
    """Main function to process all CSV files."""
    # Find all CSV files in the workspace
    base_path = Path("/home/nguyenchiphong2909/powerpeak/SAML3P")
    csv_files = list(base_path.glob("**/*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file_path in csv_files:
        print(f"  - {file_path}")
    
    print("\nStarting processing...")
    
    success_count = 0
    for file_path in csv_files:
        if process_csv_file(file_path):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(csv_files)} files")

if __name__ == "__main__":
    main()
