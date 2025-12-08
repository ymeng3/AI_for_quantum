#!/usr/bin/env python3
"""
One-time script to sync existing database data to Google Sheets.
Run this to migrate your existing 71 pairwise comparisons to Google Sheets.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from google_sheets_sync import sync_from_database_to_sheets

if __name__ == '__main__':
    print("=" * 60)
    print("Syncing existing database data to Google Sheets")
    print("=" * 60)
    print()
    sync_from_database_to_sheets()
    print()
    print("=" * 60)
    print("Sync complete!")
    print("=" * 60)

