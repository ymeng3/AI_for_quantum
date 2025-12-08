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
        except:
            worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_PAIRWISE_TAB, rows=1000, cols=10)
            # Add headers
            worksheet.append_row([
                'Image1_Path', 'Image1_Name', 'Image2_Path', 'Image2_Name',
                'Reconstruction_Type', 'Winner', 'Labeler_Name', 'Notes', 'Created_At'
            ])
        
        # Append the new row
        row = [
            comparison_data.get('image1_path', ''),
            comparison_data.get('image1_name', ''),
            comparison_data.get('image2_path', ''),
            comparison_data.get('image2_name', ''),
            comparison_data.get('reconstruction_type', ''),
            comparison_data.get('winner', ''),
            comparison_data.get('labeler_name', ''),
            comparison_data.get('notes', ''),
            comparison_data.get('created_at', datetime.now().isoformat())
        ]
        worksheet.append_row(row)
        return True
    except Exception as e:
        print(f"Error syncing to Google Sheets: {e}")
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
        except:
            worksheet = spreadsheet.add_worksheet(title=GOOGLE_SHEETS_ABSOLUTE_TAB, rows=1000, cols=10)
            # Add headers
            worksheet.append_row([
                'File_Path', 'File_Name', 'Quality', 'Reconstruction', 
                'Reconstruction_Scores', 'Labeler_Name', 'Notes', 'Created_At', 'Updated_At'
            ])
        
        # Check if row exists (by file_path) and update, or append
        try:
            # Try to find existing row
            cell = worksheet.find(label_data.get('file_path', ''))
            # Update existing row
            row_num = cell.row
            row = [
                label_data.get('file_path', ''),
                label_data.get('file_name', ''),
                label_data.get('quality', ''),
                label_data.get('reconstruction', ''),
                label_data.get('reconstruction_scores', ''),
                label_data.get('labeler_name', ''),
                label_data.get('notes', ''),
                label_data.get('created_at', ''),
                label_data.get('updated_at', datetime.now().isoformat())
            ]
            worksheet.update(f'A{row_num}:I{row_num}', [row])
        except:
            # Row doesn't exist, append new one
            row = [
                label_data.get('file_path', ''),
                label_data.get('file_name', ''),
                label_data.get('quality', ''),
                label_data.get('reconstruction', ''),
                label_data.get('reconstruction_scores', ''),
                label_data.get('labeler_name', ''),
                label_data.get('notes', ''),
                label_data.get('created_at', datetime.now().isoformat()),
                label_data.get('updated_at', datetime.now().isoformat())
            ]
            worksheet.append_row(row)
        return True
    except Exception as e:
        print(f"Error syncing to Google Sheets: {e}")
        return False

def sync_from_sheets_to_databases():
    """Sync data from Google Sheets to both SQLite and PostgreSQL databases"""
    client = get_google_sheets_client()
    if not client or not GOOGLE_SHEETS_SPREADSHEET_ID:
        print("Google Sheets not configured")
        return
    
    from app import get_db_connection, USE_POSTGRES
    
    try:
        spreadsheet = client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        
        # Sync pairwise comparisons
        try:
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_PAIRWISE_TAB)
            rows = worksheet.get_all_records()
            
            conn = get_db_connection()
            if USE_POSTGRES:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cursor = conn.cursor()
            
            for row in rows:
                # Check if exists
                if USE_POSTGRES:
                    cursor.execute("""
                        SELECT id FROM pairwise_comparisons 
                        WHERE image1_path = %s AND image2_path = %s AND reconstruction_type = %s
                    """, (row['Image1_Path'], row['Image2_Path'], row['Reconstruction_Type']))
                else:
                    cursor.execute("""
                        SELECT id FROM pairwise_comparisons 
                        WHERE image1_path = ? AND image2_path = ? AND reconstruction_type = ?
                    """, (row['Image1_Path'], row['Image2_Path'], row['Reconstruction_Type']))
                
                exists = cursor.fetchone()
                
                if not exists:
                    # Insert new record
                    if USE_POSTGRES:
                        cursor.execute("""
                            INSERT INTO pairwise_comparisons 
                            (image1_path, image1_name, image2_path, image2_name, 
                             reconstruction_type, winner, labeler_name, notes, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (row['Image1_Path'], row['Image1_Name'], row['Image2_Path'], 
                              row['Image2_Name'], row['Reconstruction_Type'], row['Winner'],
                              row.get('Labeler_Name', ''), row.get('Notes', ''), 
                              row.get('Created_At', datetime.now().isoformat())))
                    else:
                        cursor.execute("""
                            INSERT INTO pairwise_comparisons 
                            (image1_path, image1_name, image2_path, image2_name, 
                             reconstruction_type, winner, labeler_name, notes, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (row['Image1_Path'], row['Image1_Name'], row['Image2_Path'], 
                              row['Image2_Name'], row['Reconstruction_Type'], row['Winner'],
                              row.get('Labeler_Name', ''), row.get('Notes', ''), 
                              row.get('Created_At', datetime.now().isoformat())))
            
            conn.commit()
            conn.close()
            print(f"Synced {len(rows)} pairwise comparisons from Google Sheets")
        except Exception as e:
            print(f"Error syncing pairwise data: {e}")
        
        # Sync absolute scoring (similar logic)
        # ... (implement similar to pairwise)
        
    except Exception as e:
        print(f"Error syncing from Google Sheets: {e}")

if __name__ == '__main__':
    print("Google Sheets sync utility")
    print("=" * 60)
    sync_from_sheets_to_databases()

