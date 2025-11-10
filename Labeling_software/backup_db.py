#!/usr/bin/env python3
"""
Simple script to backup the labels database.
Usage: python backup_db.py
"""
import shutil
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "labels.db"
BACKUP_DIR = Path(__file__).parent / "backups"

def backup_database():
    if not DB_PATH.exists():
        print("No database file found. Nothing to backup.")
        return
    
    # Create backups directory if it doesn't exist
    BACKUP_DIR.mkdir(exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"labels_backup_{timestamp}.db"
    backup_path = BACKUP_DIR / backup_filename
    
    # Copy the database file
    shutil.copy2(DB_PATH, backup_path)
    
    print(f"✓ Database backed up to: {backup_path}")
    print(f"  Original size: {DB_PATH.stat().st_size / 1024:.2f} KB")
    print(f"  Backup size: {backup_path.stat().st_size / 1024:.2f} KB")
    
    # Keep only the last 10 backups
    backups = sorted(BACKUP_DIR.glob("labels_backup_*.db"), reverse=True)
    if len(backups) > 10:
        for old_backup in backups[10:]:
            old_backup.unlink()
            print(f"  Removed old backup: {old_backup.name}")

if __name__ == "__main__":
    backup_database()

