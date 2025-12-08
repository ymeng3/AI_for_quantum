#!/usr/bin/env python3
"""
Import AJ's pairwise comparison data from the exported CSV format.
This handles the export format with full paths already included.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent))

# Import database connection from app
from app import get_db_connection, USE_POSTGRES

def convert_winner(winner_str):
    """Convert winner string to database format"""
    winner_lower = str(winner_str).strip().lower()
    if winner_lower in ["1", "image 1"]:
        return "1"
    elif winner_lower in ["2", "image 2"]:
        return "2"
    elif winner_lower == "tie":
        return "tie"
    elif winner_lower in ["not apply", "not_apply"]:
        return "not_apply"
    else:
        print(f"Warning: Unknown winner value '{winner_str}', defaulting to 'tie'")
        return "tie"

def import_aj_data():
    """Import AJ's pairwise comparison data from CSV"""
    # CSV is in Labeled_data folder (parent directory)
    csv_path = Path(__file__).parent.parent / 'Labeled_data' / 'labeled_data_AJ.csv'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Use the same database connection as the app
    try:
        conn = get_db_connection()
        if USE_POSTGRES:
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            cursor = conn.cursor()
        print("✅ Connected to database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("\nMake sure DATABASE_URL is set in Render environment variables")
        return
    
    # Verify table exists
    try:
        if USE_POSTGRES:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'pairwise_comparisons'
                );
            """)
            table_exists = cursor.fetchone()[0]
        else:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='pairwise_comparisons'
            """)
            table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            print("Error: pairwise_comparisons table does not exist!")
            conn.close()
            return
    except Exception as e:
        print(f"Error checking table: {e}")
        conn.close()
        return
    
    # Read CSV and import
    imported_count = 0
    skipped_count = 0
    errors = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, start=2):
            try:
                # Handle both formats: export format (with paths) or simple format
                if 'Image1_Path' in row:
                    # Export format
                    image1_path = row['Image1_Path'].strip()
                    image1_name = row['Image1_Name'].strip()
                    image2_path = row['Image2_Path'].strip()
                    image2_name = row['Image2_Name'].strip()
                    reconstruction = row['Reconstruction_Type'].strip()
                    winner_str = row['Winner'].strip()
                    labeler = row['Labeler_Name'].strip() if row.get('Labeler_Name') else 'AJ'
                    notes = row['Notes'].strip() if row.get('Notes') and row['Notes'].strip() else ''
                    created_at = row.get('Created_At', '').strip() if row.get('Created_At') else None
                else:
                    # Simple format (like SY's data)
                    image1_name = row['IMAGE 1'].strip()
                    image2_name = row['IMAGE 2'].strip()
                    reconstruction = row['RECONSTRUCTION'].strip()
                    winner_str = row['WINNER'].strip()
                    labeler = row['LABELER'].strip() if row.get('LABELER') else 'AJ'
                    notes = row['NOTES'].strip() if row.get('NOTES') and row['NOTES'].strip() != '-' else ''
                    created_at = None
                    
                    # Construct paths
                    if "RR220204" in image1_name:
                        image1_path = f"Trajectories/2022-02-04/{image1_name}"
                    elif "RR220206" in image1_name:
                        image1_path = f"Trajectories/2022-02-06/{image1_name}"
                    elif "RR220411" in image1_name:
                        image1_path = f"Trajectories/2022-04-11/{image1_name}"
                    else:
                        image1_path = f"Trajectories/2022-02-04/{image1_name}"
                    
                    if "RR220204" in image2_name:
                        image2_path = f"Trajectories/2022-02-04/{image2_name}"
                    elif "RR220206" in image2_name:
                        image2_path = f"Trajectories/2022-02-06/{image2_name}"
                    elif "RR220411" in image2_name:
                        image2_path = f"Trajectories/2022-04-11/{image2_name}"
                    else:
                        image2_path = f"Trajectories/2022-02-04/{image2_name}"
                
                winner = convert_winner(winner_str)
                
                # Use provided created_at or current time
                if created_at:
                    try:
                        # Try to parse the timestamp
                        from dateutil import parser
                        created_at_parsed = parser.parse(created_at).isoformat()
                    except:
                        created_at_parsed = datetime.now().isoformat()
                else:
                    created_at_parsed = datetime.now().isoformat()
                
                # Insert into database
                if USE_POSTGRES:
                    cursor.execute('''
                        INSERT INTO pairwise_comparisons 
                        (image1_path, image1_name, image2_path, image2_name, 
                         reconstruction_type, winner, labeler_name, notes, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (image1_path, image1_name, image2_path, image2_name, 
                          reconstruction, winner, labeler, notes, created_at_parsed))
                else:
                    cursor.execute('''
                        INSERT INTO pairwise_comparisons 
                        (image1_path, image1_name, image2_path, image2_name, 
                         reconstruction_type, winner, labeler_name, notes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (image1_path, image1_name, image2_path, image2_name, 
                          reconstruction, winner, labeler, notes, created_at_parsed))
                
                imported_count += 1
                
            except Exception as e:
                error_msg = f"Row {row_num}: {str(e)}"
                errors.append(error_msg)
                print(f"Error processing row {row_num}: {e}")
                skipped_count += 1
    
    try:
        conn.commit()
        print(f"\n✅ Import complete!")
        print(f"   - Successfully imported: {imported_count} comparisons")
        print(f"   - Skipped due to errors: {skipped_count}")
    except Exception as e:
        print(f"\n❌ Error committing to database: {e}")
        conn.rollback()
    
    if errors:
        print(f"\n⚠️  Errors encountered:")
        for error in errors[:10]:
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    
    conn.close()

if __name__ == '__main__':
    print("Importing AJ's pairwise comparison data...")
    print("=" * 60)
    import_aj_data()

