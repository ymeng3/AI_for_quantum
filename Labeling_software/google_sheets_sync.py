#!/usr/bin/env python3
"""
Google Sheets synchronization for labeling data.
Uses Google Sheets as the single source of truth and syncs to both SQLite and PostgreSQL.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("Warning: gspread not installed. Install with: pip install gspread google-auth")

# Configuration
GOOGLE_SHEETS_CREDENTIALS = os.environ.get('GOOGLE_SHEETS_CREDENTIALS_JSON')
GOOGLE_SHEETS_SPREADSHEET_ID = os.environ.get('GOOGLE_SHEETS_SPREADSHEET_ID')
GOOGLE_SHEETS_ABSOLUTE_TAB = os.environ.get('GOOGLE_SHEETS_ABSOLUTE_TAB', 'Absolute_Scoring')
GOOGLE_SHEETS_PAIRWISE_TAB = os.environ.get('GOOGLE_SHEETS_PAIRWISE_TAB', 'Pairwise_Comparison')

def get_google_sheets_client():
    """Get authenticated Google Sheets client"""
    if not GSPREAD_AVAILABLE:
        return None
    
    if not GOOGLE_SHEETS_CREDENTIALS:
        return None
    
    try:
        # Parse credentials from JSON string or file path
        if GOOGLE_SHEETS_CREDENTIALS.startswith('{'):
            # JSON string
            creds_dict = json.loads(GOOGLE_SHEETS_CREDENTIALS)
        else:
            # File path
            with open(GOOGLE_SHEETS_CREDENTIALS, 'r') as f:
                creds_dict = json.load(f)
        
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"Error authenticating with Google Sheets: {e}")
        return None

def sync_pairwise_to_sheets(comparison_data, client=None):
    """Sync a pairwise comparison to Google Sheets"""
    if not client:
        client = get_google_sheets_client()
    if not client:
        return False
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        return False
    
    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        
        # Get or create the pairwise tab
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_PAIRWISE_TAB)
            # Check if headers need updating (add new columns to the right)
            headers = worksheet.row_values(1)
            expected_headers = ['Image1_Path', 'Image1_Name', 'Image2_Path', 'Image2_Name',
                              'Reconstruction_Type', 'Winner', 'Labeler_Name', 'Notes', 'Created_At',
                              'Brightness_1', 'Brightness_2', 'Confidence']
            if not headers:
                # No headers, add all
                worksheet.append_row(expected_headers)
            elif len(headers) < 12:
                # Old headers without new columns - add new columns to the right
                # Update header row to include new columns
                worksheet.update('J1:L1', [['Brightness_1', 'Brightness_2', 'Confidence']])
        except:
            worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_PAIRWISE_TAB, rows=1000, cols=15)
            # Add headers with new columns
            worksheet.append_row([
                'Image1_Path', 'Image1_Name', 'Image2_Path', 'Image2_Name',
                'Reconstruction_Type', 'Winner', 'Labeler_Name', 'Notes', 'Created_At',
                'Brightness_1', 'Brightness_2', 'Confidence'
            ])
        
        # Append the new row (ensure all 12 columns)
        brightness_1 = comparison_data.get('brightness_1')
        brightness_2 = comparison_data.get('brightness_2')
        row = [
            str(comparison_data.get('image1_path', '')),
            str(comparison_data.get('image1_name', '')),
            str(comparison_data.get('image2_path', '')),
            str(comparison_data.get('image2_name', '')),
            str(comparison_data.get('reconstruction_type', '')),
            str(comparison_data.get('winner', '')),
            str(comparison_data.get('labeler_name', '')),
            str(comparison_data.get('notes', '')),
            str(comparison_data.get('created_at', datetime.now().isoformat())),
            str(brightness_1) if brightness_1 is not None else '',
            str(brightness_2) if brightness_2 is not None else '',
            str(comparison_data.get('confidence', 'Confident'))
        ]
        # Ensure row has exactly 12 values
        while len(row) < 12:
            row.append('')
        worksheet.append_row(row[:12])
        return True
    except Exception as e:
        import traceback
        print(f"Error syncing to Google Sheets: {e}\n{traceback.format_exc()}")
        return False

def sync_absolute_to_sheets(label_data, client=None):
    """Sync an absolute scoring label to Google Sheets"""
    if not client:
        client = get_google_sheets_client()
    if not client:
        return False
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        return False
    
    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        
        # Get or create the absolute tab
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_ABSOLUTE_TAB)
            # Ensure headers exist and are correct
            headers = worksheet.row_values(1)
            expected_headers = ['File_Path', 'File_Name', 'Quality', 'Reconstruction', 
                               'Reconstruction_Scores', 'Labeler_Name', 'Notes', 'Created_At', 'Updated_At']
            if not headers or headers != expected_headers:
                # Clear and add correct headers
                worksheet.clear()
                worksheet.append_row(expected_headers)
        except:
            worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_ABSOLUTE_TAB, rows=1000, cols=10)
            # Add headers
            worksheet.append_row([
                'File_Path', 'File_Name', 'Quality', 'Reconstruction', 
                'Reconstruction_Scores', 'Labeler_Name', 'Notes', 'Created_At', 'Updated_At'
            ])
        
        file_path = str(label_data.get('file_path', ''))
        
        # Check if row exists (by file_path) and update, or append
        try:
            # Try to find existing row - search in column A (File_Path)
            all_values = worksheet.get_all_values()
            row_num = None
            for i, row in enumerate(all_values, start=1):
                if i == 1:  # Skip header
                    continue
                if len(row) > 0 and row[0] == file_path:
                    row_num = i
                    break
            
            # Prepare row data (ensure all 9 columns)
            row = [
                file_path,
                str(label_data.get('file_name', '')),
                str(label_data.get('quality', '')),
                str(label_data.get('reconstruction', '')),
                str(label_data.get('reconstruction_scores', '')),
                str(label_data.get('labeler_name', '')),
                str(label_data.get('notes', '')),
                str(label_data.get('created_at', '') or datetime.now().isoformat()),
                str(label_data.get('updated_at', '') or datetime.now().isoformat())
            ]
            # Ensure row has exactly 9 values
            while len(row) < 9:
                row.append('')
            
            if row_num:
                # Update existing row
                worksheet.update(f'A{row_num}:I{row_num}', [row[:9]])
            else:
                # Row doesn't exist, append new one
                worksheet.append_row(row[:9])
        except Exception as e:
            import traceback
            print(f"Error updating row in Google Sheets: {e}\n{traceback.format_exc()}")
            return False
        return True
    except Exception as e:
        import traceback
        print(f"Error syncing to Google Sheets: {e}\n{traceback.format_exc()}")
        return False

def delete_pairwise_from_sheets(image1_path, image2_path, reconstruction_type, client=None):
    """Delete a pairwise comparison from Google Sheets"""
    if not client:
        client = get_google_sheets_client()
    if not client:
        return False
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        return False
    
    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_PAIRWISE_TAB)
        except:
            # Tab doesn't exist, nothing to delete
            return True
        
        # Find the row(s) matching the criteria
        try:
            # Search for matching rows
            all_values = worksheet.get_all_values()
            rows_to_delete = []
            
            for i, row in enumerate(all_values, start=1):
                # Skip header row
                if i == 1:
                    continue
                # Check if this row matches
                if (len(row) >= 5 and 
                    row[0] == image1_path and 
                    row[2] == image2_path and 
                    row[4] == reconstruction_type):
                    rows_to_delete.append(i)
            
            # Delete rows from bottom to top to maintain indices
            for row_num in reversed(rows_to_delete):
                worksheet.delete_rows(row_num)
            
            return len(rows_to_delete) > 0
        except Exception as e:
            print(f"Error finding/deleting row in Google Sheets: {e}")
            return False
    except Exception as e:
        print(f"Error deleting from Google Sheets: {e}")
        return False

def delete_absolute_from_sheets(file_path, client=None):
    """Delete an absolute scoring label from Google Sheets"""
    if not client:
        client = get_google_sheets_client()
    if not client:
        return False
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        return False
    
    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_ABSOLUTE_TAB)
        except:
            # Tab doesn't exist, nothing to delete
            return True
        
        # Find the row matching the file_path (search in column A)
        try:
            all_values = worksheet.get_all_values()
            rows_to_delete = []
            
            for i, row in enumerate(all_values, start=1):
                if i == 1:  # Skip header
                    continue
                if len(row) > 0 and row[0] == str(file_path):
                    rows_to_delete.append(i)
            
            # Delete rows from bottom to top to maintain indices
            for row_num in reversed(rows_to_delete):
                worksheet.delete_rows(row_num)
            
            return len(rows_to_delete) > 0
        except Exception as e:
            import traceback
            print(f"Error finding/deleting row in Google Sheets: {e}\n{traceback.format_exc()}")
            return False
    except Exception as e:
        import traceback
        print(f"Error deleting from Google Sheets: {e}\n{traceback.format_exc()}")
        return False

def sync_from_database_to_sheets():
    """Sync existing data from database to Google Sheets (one-time migration)"""
    print("Connecting to Google Sheets...")
    client = get_google_sheets_client()
    if not client:
        print("❌ ERROR: Could not authenticate with Google Sheets")
        print("   Check your GOOGLE_SHEETS_CREDENTIALS_JSON")
        return
    
    if not GOOGLE_SHEETS_SPREADSHEET_ID:
        print("❌ ERROR: GOOGLE_SHEETS_SPREADSHEET_ID not set")
        return
    
    print(f"✅ Connected to Google Sheets (Spreadsheet ID: {GOOGLE_SHEETS_SPREADSHEET_ID[:20]}...)")
    print()
    
    # Check DATABASE_URL explicitly and connect directly to PostgreSQL
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("❌ ERROR: DATABASE_URL not set!")
        print("   This script needs to connect to PostgreSQL on Render.")
        return
    
    print(f"✅ DATABASE_URL found: {database_url[:30]}...")
    
    # Connect directly to PostgreSQL (don't rely on app.py's connection)
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        print("✅ Connecting directly to PostgreSQL...")
        USE_POSTGRES = True
    except ImportError:
        print("❌ ERROR: psycopg2 not installed!")
        return
    
    def get_db_connection():
        """Get PostgreSQL connection directly"""
        return psycopg2.connect(database_url, sslmode='require')
    
    print("✅ Using PostgreSQL database")
    print()
    
    try:
        print("Opening spreadsheet...")
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        print("✅ Spreadsheet opened")
        print()
        
        # Sync pairwise comparisons
        print("=== Syncing Pairwise Comparisons ===")
        try:
            print("Connecting to database...")
            conn = get_db_connection()
            print(f"✅ Connected to {'PostgreSQL' if USE_POSTGRES else 'SQLite'} database")
            
            print("Fetching pairwise comparisons from database...")
            if USE_POSTGRES:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
                comparisons = [dict(row) for row in cursor.fetchall()]
            else:
                cursor = conn.cursor()
                cursor.execute('PRAGMA table_info(pairwise_comparisons)')
                columns = [row[1] for row in cursor.fetchall()]
                cursor.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
                comparisons = []
                for row in cursor.fetchall():
                    comp_dict = {}
                    for i, col in enumerate(columns):
                        comp_dict[col] = row[i]
                    comparisons.append(comp_dict)
            conn.close()
            print(f"✅ Found {len(comparisons)} pairwise comparisons in database")
            
            if comparisons:
                # Get or create worksheet
                try:
                    worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_PAIRWISE_TAB)
                    # Ensure headers are correct
                    headers = worksheet.row_values(1)
                    expected_headers = ['Image1_Path', 'Image1_Name', 'Image2_Path', 'Image2_Name',
                                      'Reconstruction_Type', 'Winner', 'Labeler_Name', 'Notes', 'Created_At']
                    if not headers or headers != expected_headers:
                        # Clear and add correct headers
                        worksheet.clear()
                        worksheet.append_row(expected_headers)
                    # Get existing data to avoid duplicates
                    existing = worksheet.get_all_records()
                    existing_keys = set()
                    for row in existing:
                        key = (str(row.get('Image1_Path', '')), str(row.get('Image2_Path', '')), str(row.get('Reconstruction_Type', '')))
                        existing_keys.add(key)
                except:
                    worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_PAIRWISE_TAB, rows=1000, cols=10)
                    worksheet.append_row([
                        'Image1_Path', 'Image1_Name', 'Image2_Path', 'Image2_Name',
                        'Reconstruction_Type', 'Winner', 'Labeler_Name', 'Notes', 'Created_At'
                    ])
                    existing_keys = set()
                
                # Add new rows
                added = 0
                for comp in comparisons:
                    key = (str(comp.get('image1_path', '')), str(comp.get('image2_path', '')), str(comp.get('reconstruction_type', '')))
                    if key not in existing_keys:
                        row = [
                            str(comp.get('image1_path', '')),
                            str(comp.get('image1_name', '')),
                            str(comp.get('image2_path', '')),
                            str(comp.get('image2_name', '')),
                            str(comp.get('reconstruction_type', '')),
                            str(comp.get('winner', '')),
                            str(comp.get('labeler_name', '')),
                            str(comp.get('notes', '')),
                            str(comp.get('created_at', '') or datetime.now().isoformat())
                        ]
                        # Ensure row has exactly 9 values
                        while len(row) < 9:
                            row.append('')
                        worksheet.append_row(row[:9])
                        added += 1
                        existing_keys.add(key)
                
                print(f"✅ Synced {added} new pairwise comparisons to Google Sheets (total in DB: {len(comparisons)})")
            else:
                print("No pairwise comparisons in database to sync")
        except Exception as e:
            import traceback
            print(f"Error syncing pairwise data: {e}\n{traceback.format_exc()}")
        
        # Sync absolute scoring
        print()
        print("=== Syncing Absolute Scoring Labels ===")
        try:
            print("Connecting to database...")
            conn = get_db_connection()
            print(f"✅ Connected to {'PostgreSQL' if USE_POSTGRES else 'SQLite'} database")
            
            print("Fetching absolute labels from database...")
            if USE_POSTGRES:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute('SELECT * FROM labels ORDER BY created_at DESC')
                labels = [dict(row) for row in cursor.fetchall()]
            else:
                cursor = conn.cursor()
                cursor.execute('PRAGMA table_info(labels)')
                columns = [row[1] for row in cursor.fetchall()]
                cursor.execute('SELECT * FROM labels ORDER BY created_at DESC')
                labels = []
                for row in cursor.fetchall():
                    label_dict = {}
                    for i, col in enumerate(columns):
                        label_dict[col] = row[i]
                    labels.append(label_dict)
            conn.close()
            print(f"✅ Found {len(labels)} absolute labels in database")
            
            if labels:
                # Get or create worksheet
                try:
                    worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_ABSOLUTE_TAB)
                    # Ensure headers are correct
                    headers = worksheet.row_values(1)
                    expected_headers = ['File_Path', 'File_Name', 'Quality', 'Reconstruction', 
                                       'Reconstruction_Scores', 'Labeler_Name', 'Notes', 'Created_At', 'Updated_At']
                    if not headers or headers != expected_headers:
                        # Clear and add correct headers
                        worksheet.clear()
                        worksheet.append_row(expected_headers)
                    existing = worksheet.get_all_records()
                    existing_paths = {str(row.get('File_Path', '')) for row in existing}
                except:
                    worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_ABSOLUTE_TAB, rows=1000, cols=10)
                    worksheet.append_row([
                        'File_Path', 'File_Name', 'Quality', 'Reconstruction', 
                        'Reconstruction_Scores', 'Labeler_Name', 'Notes', 'Created_At', 'Updated_At'
                    ])
                    existing_paths = set()
                
                # Add new rows
                added = 0
                for label in labels:
                    file_path = str(label.get('file_path', ''))
                    if file_path and file_path not in existing_paths:
                        row = [
                            file_path,
                            str(label.get('file_name', '')),
                            str(label.get('quality', '')),
                            str(label.get('reconstruction', '')),
                            str(label.get('reconstruction_scores', '')),
                            str(label.get('labeler_name', '')),
                            str(label.get('notes', '')),
                            str(label.get('created_at', '') or datetime.now().isoformat()),
                            str(label.get('updated_at', '') or datetime.now().isoformat())
                        ]
                        # Ensure row has exactly 9 values
                        while len(row) < 9:
                            row.append('')
                        worksheet.append_row(row[:9])
                        added += 1
                        existing_paths.add(file_path)
                
                print(f"✅ Synced {added} new absolute labels to Google Sheets (total in DB: {len(labels)})")
            else:
                print("No absolute labels in database to sync")
        except Exception as e:
            import traceback
            print(f"Error syncing absolute data: {e}\n{traceback.format_exc()}")
        
    except Exception as e:
        import traceback
        print(f"Error syncing from database: {e}\n{traceback.format_exc()}")

def restore_from_sheets_if_empty(get_db_connection_func, use_postgres):
    """Restore data from Google Sheets if database is empty (called on app startup)"""
    client = get_google_sheets_client()
    if not client or not GOOGLE_SHEETS_SPREADSHEET_ID:
        print("Google Sheets not configured, skipping restore check")
        return

    try:
        conn = get_db_connection_func()
        cursor = conn.cursor()

        # Check if database has any data
        if use_postgres:
            cursor.execute('SELECT COUNT(*) FROM labels')
            labels_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM pairwise_comparisons')
            pairwise_count = cursor.fetchone()[0]
        else:
            cursor.execute('SELECT COUNT(*) FROM labels')
            labels_count = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM pairwise_comparisons')
            pairwise_count = cursor.fetchone()[0]

        conn.close()

        if labels_count > 0 or pairwise_count > 0:
            print(f"Database has data (labels: {labels_count}, pairwise: {pairwise_count}), skipping restore")
            return

        print("Database is empty! Restoring from Google Sheets...")

        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)

        # Restore pairwise comparisons
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_PAIRWISE_TAB)
            rows = worksheet.get_all_records()

            if rows:
                conn = get_db_connection_func()
                cursor = conn.cursor()

                restored = 0
                for row in rows:
                    try:
                        if use_postgres:
                            cursor.execute("""
                                INSERT INTO pairwise_comparisons
                                (image1_path, image1_name, image2_path, image2_name,
                                 reconstruction_type, winner, labeler_name, notes, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (row.get('Image1_Path', ''), row.get('Image1_Name', ''),
                                  row.get('Image2_Path', ''), row.get('Image2_Name', ''),
                                  row.get('Reconstruction_Type', ''), row.get('Winner', ''),
                                  row.get('Labeler_Name', ''), row.get('Notes', ''),
                                  row.get('Created_At', '') or datetime.now().isoformat()))
                        else:
                            cursor.execute("""
                                INSERT INTO pairwise_comparisons
                                (image1_path, image1_name, image2_path, image2_name,
                                 reconstruction_type, winner, labeler_name, notes, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (row.get('Image1_Path', ''), row.get('Image1_Name', ''),
                                  row.get('Image2_Path', ''), row.get('Image2_Name', ''),
                                  row.get('Reconstruction_Type', ''), row.get('Winner', ''),
                                  row.get('Labeler_Name', ''), row.get('Notes', ''),
                                  row.get('Created_At', '') or datetime.now().isoformat()))
                        restored += 1
                    except Exception as e:
                        print(f"Error restoring pairwise row: {e}")

                conn.commit()
                conn.close()
                print(f"✅ Restored {restored} pairwise comparisons from Google Sheets")
        except Exception as e:
            print(f"Error restoring pairwise data: {e}")

        # Restore absolute labels
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_ABSOLUTE_TAB)
            rows = worksheet.get_all_records()

            if rows:
                conn = get_db_connection_func()
                cursor = conn.cursor()

                restored = 0
                for row in rows:
                    try:
                        file_path = row.get('File_Path', '')
                        if not file_path:
                            continue

                        if use_postgres:
                            cursor.execute("""
                                INSERT INTO labels
                                (file_path, file_name, quality, reconstruction,
                                 reconstruction_scores, labeler_name, notes, created_at, updated_at)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (file_path) DO NOTHING
                            """, (file_path, row.get('File_Name', ''),
                                  row.get('Quality', ''), row.get('Reconstruction', ''),
                                  row.get('Reconstruction_Scores', ''), row.get('Labeler_Name', ''),
                                  row.get('Notes', ''),
                                  row.get('Created_At', '') or datetime.now().isoformat(),
                                  row.get('Updated_At', '') or datetime.now().isoformat()))
                        else:
                            cursor.execute("""
                                INSERT OR IGNORE INTO labels
                                (file_path, file_name, quality, reconstruction,
                                 reconstruction_scores, labeler_name, notes, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (file_path, row.get('File_Name', ''),
                                  row.get('Quality', ''), row.get('Reconstruction', ''),
                                  row.get('Reconstruction_Scores', ''), row.get('Labeler_Name', ''),
                                  row.get('Notes', ''),
                                  row.get('Created_At', '') or datetime.now().isoformat(),
                                  row.get('Updated_At', '') or datetime.now().isoformat()))
                        restored += 1
                    except Exception as e:
                        print(f"Error restoring label row: {e}")

                conn.commit()
                conn.close()
                print(f"✅ Restored {restored} absolute labels from Google Sheets")
        except Exception as e:
            print(f"Error restoring absolute labels: {e}")

    except Exception as e:
        import traceback
        print(f"Error in restore_from_sheets_if_empty: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    print("Google Sheets sync utility")
    print("=" * 60)
    sync_from_sheets_to_databases()

