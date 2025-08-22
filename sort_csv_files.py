#!/usr/bin/env python3
"""
Script to sort all CSV files by column 'n' (ascending) then by column 'c' (ascending).
"""

import csv
import os
from pathlib import Path

def sort_csv_file(file_path):
    """Sort a CSV file by column 'n' then by column 'c'."""
    print(f"Processing: {file_path}")
    
    try:
        # Read the CSV file
        rows = []
        headers = None
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            for i, row in enumerate(reader):
                if i == 0:
                    # Check if first row contains headers
                    if any(col.strip().lower() in ['n', 'c', 'instance', 'filename'] for col in row):
                        headers = row
                        continue
                    else:
                        # No headers, treat as data
                        rows.append(row)
                else:
                    rows.append(row)
        
        if not rows:
            print(f"  ⚠️ No data rows found in {file_path}")
            return True
        
        # Find column indices for 'n' and 'c'
        n_index = None
        c_index = None
        
        if headers:
            for i, header in enumerate(headers):
                if header.strip().lower() == 'n':
                    n_index = i
                elif header.strip().lower() == 'c':
                    c_index = i
        else:
            # For files without headers, determine by file structure
            print(f"  ⚠️ No headers found, analyzing file structure...")
            sample_row = rows[0] if rows else []
            
            # Check file path to determine format
            file_name = os.path.basename(file_path)
            
            if 'output.csv' in file_name:
                # Format: filename,m,c,#Var,#Cons,value,#sol,#solbb,Time
                # No 'n' column in this file, only 'm' and 'c'
                n_index = 1  # m column (treating as n for sorting)
                c_index = 2  # c column
            elif any(name in file_name.lower() for name in ['binary', 'saml3p', 'staircase', 'incremental', 'pb_cadical']):
                # Format: INSTANCE,n,m,c,... (Binary.csv and others)
                n_index = 1  # n column
                c_index = 3  # c column (skip m at index 2)
            else:
                # Default guess
                if len(sample_row) >= 4:
                    n_index = 1
                    c_index = 3
        
        if n_index is None and c_index is None:
            print(f"  ⚠️ Could not find 'n' and 'c' columns in {file_path}")
            return True
        
        print(f"  📍 Found n at index {n_index}, c at index {c_index}")
        
        # Sort function
        def sort_key(row):
            try:
                n_val = int(row[n_index]) if n_index is not None and n_index < len(row) else 0
            except (ValueError, IndexError):
                n_val = 0
            
            try:
                c_val = int(row[c_index]) if c_index is not None and c_index < len(row) else 0
            except (ValueError, IndexError):
                c_val = 0
            
            return (n_val, c_val)
        
        # Sort the rows
        sorted_rows = sorted(rows, key=sort_key)
        
        # Write back to file
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers if they exist
            if headers:
                writer.writerow(headers)
            
            # Write sorted data
            writer.writerows(sorted_rows)
        
        print(f"  ✓ Successfully sorted {file_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all CSV files."""
    # Find all CSV files in the workspace
    base_path = Path("/home/nguyenchiphong2909/powerpeak/SAML3P")
    csv_files = list(base_path.glob("**/*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    print(f"Found {len(csv_files)} CSV files to sort:")
    for file_path in csv_files:
        print(f"  - {file_path}")
    
    print("\nStarting sorting process...")
    
    success_count = 0
    for file_path in csv_files:
        if sort_csv_file(file_path):
            success_count += 1
    
    print(f"\nSorting complete!")
    print(f"Successfully processed: {success_count}/{len(csv_files)} files")

if __name__ == "__main__":
    main()
