#!/usr/bin/env python3
"""
Script to import pairwise comparison data from CSV into Render PostgreSQL database.
This connects to your Render database using the DATABASE_URL environment variable.
"""

import csv
import os
from pathlib import Path
from datetime import datetime

def find_image_path(image_name):
    """Find the full path for an image by searching in Trajectories folders"""
    # Files seem to be from 2022-02-04, 2022-02-06, 2022-04-11
    if "RR220204" in image_name:
        return f"Trajectories/2022-02-04/{image_name}"
    elif "RR220206" in image_name:
        return f"Trajectories/2022-02-06/{image_name}"
    elif "RR220411" in image_name:
        return f"Trajectories/2022-04-11/{image_name}"
    else:
        # Default to 2022-02-04 if we can't determine
        return f"Trajectories/2022-02-04/{image_name}"

def convert_winner(winner_str):
    """Convert winner string to database format"""
    winner_lower = winner_str.strip().lower()
    if winner_lower == "image 1":
        return "1"
    elif winner_lower == "image 2":
        return "2"
    elif winner_lower == "tie":
        return "tie"
    elif winner_lower == "not apply":
        return "not_apply"
    else:
        print(f"Warning: Unknown winner value '{winner_str}', defaulting to 'tie'")
        return "tie"

def import_pairwise_to_render(csv_file='label_pairs.csv'):
    """Import pairwise comparison data from CSV to Render PostgreSQL database"""
    csv_path = Path(__file__).parent / csv_file
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Get DATABASE_URL from environment
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        # Try to get it from Render's environment or check if we're in Render
        # Render sets DATABASE_URL automatically, but in shell it might not be loaded
        print("Error: DATABASE_URL environment variable not set!")
        print("\nTo get your DATABASE_URL:")
        print("1. Go to Render Dashboard -> Your Web Service -> Environment")
        print("2. Look for DATABASE_URL in the environment variables")
        print("3. Or go to your PostgreSQL database service -> 'Connections' tab")
        print("4. Copy the 'Internal Database URL' or 'External Database URL'")
        print("\nThen run:")
        print("  export DATABASE_URL='your-postgresql-url'")
        print("  python3 import_pairwise_to_render.py")
        return
    
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        print("Error: psycopg2 not installed. Install it with: pip install psycopg2-binary")
        return
    
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(database_url, sslmode='require')
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to Render PostgreSQL database")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
    # Verify table exists
    try:
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'pairwise_comparisons'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Error: pairwise_comparisons table does not exist!")
            print("Please run the app first to create the table.")
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
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
            try:
                image1_name = row['IMAGE 1'].strip()
                image2_name = row['IMAGE 2'].strip()
                reconstruction = row['RECONSTRUCTION'].strip()
                winner_str = row['WINNER'].strip()
                labeler = row['LABELER'].strip() if row.get('LABELER') else ''
                notes = row['NOTES'].strip() if row.get('NOTES') and row['NOTES'].strip() != '-' else ''
                
                # Convert winner
                winner = convert_winner(winner_str)
                
                # Find image paths
                image1_path = find_image_path(image1_name)
                image2_path = find_image_path(image2_name)
                
                # Insert into database
                cursor.execute('''
                    INSERT INTO pairwise_comparisons 
                    (image1_path, image1_name, image2_path, image2_name, 
                     reconstruction_type, winner, labeler_name, notes, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (image1_path, image1_name, image2_path, image2_name, 
                      reconstruction, winner, labeler, notes, datetime.now().isoformat()))
                
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
        for error in errors[:10]:  # Show first 10 errors
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
    
    conn.close()

if __name__ == '__main__':
    print("Importing pairwise comparison data to Render PostgreSQL database...")
    print("=" * 60)
    import_pairwise_to_render()



