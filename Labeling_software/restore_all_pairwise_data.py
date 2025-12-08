#!/usr/bin/env python3
"""
Restore all pairwise comparison data - imports both SY and AJ data
"""

import csv
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app
sys.path.insert(0, str(Path(__file__).parent))

# Import database connection from app
from app import get_db_connection, USE_POSTGRES

def find_image_path(image_name):
    """Find the full path for an image"""
    if "RR220204" in image_name:
        return f"Trajectories/2022-02-04/{image_name}"
    elif "RR220206" in image_name:
        return f"Trajectories/2022-02-06/{image_name}"
    elif "RR220411" in image_name:
        return f"Trajectories/2022-04-11/{image_name}"
    else:
        return f"Trajectories/2022-02-04/{image_name}"

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
        return "tie"

def restore_all_data():
    """Restore both SY and AJ data"""
    conn = get_db_connection()
    if USE_POSTGRES:
        from psycopg2.extras import RealDictCursor
        cursor = conn.cursor(cursor_factory=RealDictCursor)
    else:
        cursor = conn.cursor()
    
    print("✅ Connected to database")
    
    # Check current count
    cursor.execute('SELECT COUNT(*) FROM pairwise_comparisons')
    current_count = cursor.fetchone()[0] if USE_POSTGRES else cursor.fetchone()[0]
    print(f"Current comparisons in database: {current_count}")
    
    total_imported = 0
    
    # Import SY's data
    print("\n=== Importing SY's data ===")
    sy_csv = Path(__file__).parent / 'label_pairs.csv'
    if sy_csv.exists():
        with open(sy_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            sy_count = 0
            for row in reader:
                image1_name = row['IMAGE 1'].strip()
                image2_name = row['IMAGE 2'].strip()
                reconstruction = row['RECONSTRUCTION'].strip()
                winner_str = row['WINNER'].strip()
                labeler = row['LABELER'].strip() if row.get('LABELER') else 'SY'
                notes = row['NOTES'].strip() if row.get('NOTES') and row['NOTES'].strip() != '-' else ''
                
                winner = convert_winner(winner_str)
                image1_path = find_image_path(image1_name)
                image2_path = find_image_path(image2_name)
                
                if USE_POSTGRES:
                    cursor.execute('''
                        INSERT INTO pairwise_comparisons 
                        (image1_path, image1_name, image2_path, image2_name, 
                         reconstruction_type, winner, labeler_name, notes, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (image1_path, image1_name, image2_path, image2_name, 
                          reconstruction, winner, labeler, notes, datetime.now().isoformat()))
                else:
                    cursor.execute('''
                        INSERT INTO pairwise_comparisons 
                        (image1_path, image1_name, image2_path, image2_name, 
                         reconstruction_type, winner, labeler_name, notes, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (image1_path, image1_name, image2_path, image2_name, 
                          reconstruction, winner, labeler, notes, datetime.now().isoformat()))
                sy_count += 1
        print(f"   Imported {sy_count} SY comparisons")
        total_imported += sy_count
    else:
        print("   SY CSV not found")
    
    # Import AJ's data
    print("\n=== Importing AJ's data ===")
    aj_csv_paths = [
        Path(__file__).parent.parent / 'Labeled_data' / 'labeled_data_AJ.csv',
        Path('/opt/render/project/src/Labeled_data/labeled_data_AJ.csv'),
    ]
    
    aj_csv = None
    for path in aj_csv_paths:
        if path.exists():
            aj_csv = path
            break
    
    if aj_csv:
        with open(aj_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            aj_count = 0
            for row in reader:
                if 'Image1_Path' in row:
                    image1_path = row['Image1_Path'].strip()
                    image1_name = row['Image1_Name'].strip()
                    image2_path = row['Image2_Path'].strip()
                    image2_name = row['Image2_Name'].strip()
                    reconstruction = row['Reconstruction_Type'].strip()
                    winner_str = row['Winner'].strip()
                    labeler = row['Labeler_Name'].strip() if row.get('Labeler_Name') else 'AJ'
                    notes = row['Notes'].strip() if row.get('Notes') else ''
                    created_at = row.get('Created_At', '').strip() if row.get('Created_At') else None
                    
                    winner = convert_winner(winner_str)
                    
                    if created_at:
                        created_at_parsed = created_at.replace(' ', 'T')
                        if '+' not in created_at_parsed and 'Z' not in created_at_parsed:
                            created_at_parsed += '+00:00'
                    else:
                        created_at_parsed = datetime.now().isoformat()
                    
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
                    aj_count += 1
        print(f"   Imported {aj_count} AJ comparisons")
        total_imported += aj_count
    else:
        print("   AJ CSV not found")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Restore complete!")
    print(f"   - Total imported: {total_imported} comparisons")
    
    # Verify final count
    conn = get_db_connection()
    if USE_POSTGRES:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
    else:
        cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM pairwise_comparisons')
    final_count = cursor.fetchone()[0] if USE_POSTGRES else cursor.fetchone()[0]
    print(f"   - Total in database now: {final_count} comparisons")
    conn.close()

if __name__ == '__main__':
    print("Restoring all pairwise comparison data...")
    print("=" * 60)
    restore_all_data()

