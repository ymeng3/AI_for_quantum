#!/usr/bin/env python3
"""
Fetch pairwise comparison data from Google Sheets.

Usage:
    # Set environment variables first:
    export GOOGLE_SHEETS_CREDENTIALS_JSON='{"type": "service_account", ...}'
    export GOOGLE_SHEETS_SPREADSHEET_ID='your-spreadsheet-id'

    # Then run:
    python fetch_pairwise_data.py

    # Or use a credentials file:
    export GOOGLE_SHEETS_CREDENTIALS_JSON='/path/to/credentials.json'
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("Error: gspread not installed. Install with: pip install gspread google-auth")
    sys.exit(1)

# Configuration
GOOGLE_SHEETS_CREDENTIALS = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_JSON')
GOOGLE_SHEETS_SPREADSHEET_ID = os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID')
PAIRWISE_TAB = 'Pairwise_Comparison'

# Output paths
OUTPUT_DIR = Path(__file__).parent / 'data'
OUTPUT_CSV = OUTPUT_DIR / 'pairwise_data.csv'


def get_google_sheets_client():
    """Get authenticated Google Sheets client."""
    if not GOOGLE_SHEETS_CREDENTIALS:
        print("Error: GOOGLE_SHEETS_CREDENTIALS_JSON not set")
        print("Set it to either a JSON string or path to credentials file")
        return None

    try:
        # Parse credentials from JSON string or file path
        if GOOGLE_SHEETS_CREDENTIALS.startswith('{'):
            creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
        else:
            with open(GOOGLE_SHEETS_CREDENTIALS, 'r') as f:
                creds_dict = json.load(f)

        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"Error authenticating with Google Sheets: {e}")
        return None


def fetch_pairwise_data():
    """Fetch pairwise comparison data from Google Sheets."""
    print("Connecting to Google Sheets...")
    client = get_google_sheets_client()
    if not client:
        return None

    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        print("Error: GOOGLE_SHEETS_SPREADSHEET_ID not set")
        return None

    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        print(f"Connected to spreadsheet: {spreadsheet.title}")

        # Get pairwise comparison tab
        try:
            worksheet = spreadsheet.worksheet(PAIRWISE_TAB)
        except gspread.exceptions.WorksheetNotFound:
            print(f"Error: Worksheet '{PAIRWISE_TAB}' not found")
            print("Available worksheets:", [ws.title for ws in spreadsheet.worksheets()])
            return None

        # Fetch all data
        print(f"Fetching data from '{PAIRWISE_TAB}' tab...")
        records = worksheet.get_all_records()

        if not records:
            print("Warning: No data found in worksheet")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        print(f"Fetched {len(df)} pairwise comparisons")

        # Standardize column names
        column_map = {
            'Image1_Path': 'image1_path',
            'Image1_Name': 'image1_name',
            'Image2_Path': 'image2_path',
            'Image2_Name': 'image2_name',
            'Reconstruction_Type': 'reconstruction_type',
            'Winner': 'winner',
            'Labeler_Name': 'labeler_name',
            'Notes': 'notes',
            'Created_At': 'created_at'
        }
        df = df.rename(columns=column_map)

        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None


def summarize_data(df):
    """Print summary statistics of the pairwise data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal comparisons: {len(df)}")

    if 'reconstruction_type' in df.columns:
        print("\nComparisons per reconstruction type:")
        for rt, count in df['reconstruction_type'].value_counts().items():
            print(f"  {rt}: {count}")

    if 'winner' in df.columns:
        print("\nWinner distribution:")
        for winner, count in df['winner'].value_counts().items():
            print(f"  {winner}: {count}")

    if 'labeler_name' in df.columns:
        print("\nLabelers:")
        for labeler, count in df['labeler_name'].value_counts().items():
            if labeler:  # Skip empty names
                print(f"  {labeler}: {count}")

    # Count unique images
    if 'image1_path' in df.columns and 'image2_path' in df.columns:
        all_images = set(df['image1_path'].tolist() + df['image2_path'].tolist())
        print(f"\nUnique images compared: {len(all_images)}")


def main():
    """Main function to fetch and save pairwise data."""
    print("=" * 60)
    print("PAIRWISE DATA FETCHER")
    print("=" * 60)

    # Fetch data
    df = fetch_pairwise_data()

    if df is None:
        print("\nFailed to fetch data. Check your credentials and spreadsheet ID.")
        sys.exit(1)

    if len(df) == 0:
        print("\nNo data found in the spreadsheet.")
        sys.exit(0)

    # Print summary
    summarize_data(df)

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nData saved to: {OUTPUT_CSV}")

    return df


if __name__ == '__main__':
    main()
