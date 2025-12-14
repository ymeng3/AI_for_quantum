#!/usr/bin/env python3
"""
One-time script to sync existing database data to Google Sheets.
Run this to migrate your existing 71 pairwise comparisons to Google Sheets.

Usage:
    python3 sync_existing_to_sheets.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=" * 60)
    print("Syncing existing database data to Google Sheets")
    print("=" * 60)
    print()
    
    # Check if Google Sheets is configured
    if not os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID'):
        print("❌ ERROR: GOOGLE_SHEETS_SPREADSHEET_ID not set!")
        print("   Please set this environment variable in Render.")
        return
    
    if not os.environ.get('GOOGLE_SHEETS_CREDENTIALS_JSON'):
        print("❌ ERROR: GOOGLE_SHEETS_CREDENTIALS_JSON not set!")
        print("   Please set this environment variable in Render.")
        return
    
    print("✅ Google Sheets configuration found")
    print()
    
    try:
        from google_sheets_sync import sync_from_database_to_sheets
        sync_from_database_to_sheets()
        print()
        print("=" * 60)
        print("✅ Sync complete!")
        print("=" * 60)
    except Exception as e:
        import traceback
        print()
        print("=" * 60)
        print("❌ ERROR during sync:")
        print("=" * 60)
        print(str(e))
        print()
        print("Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()

