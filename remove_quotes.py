#!/usr/bin/env python3
"""
Script to remove all quotation marks (") from CSV files.
"""

import os
from pathlib import Path

def remove_quotes_from_file(file_path):
    """Remove all quotation marks from a file."""
    print(f"Processing: {file_path}")
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove all quotation marks
        cleaned_content = content.replace('"', '')
        
        # Write back the cleaned content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"✓ Successfully cleaned {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
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
        if remove_quotes_from_file(file_path):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(csv_files)} files")

if __name__ == "__main__":
    main()
