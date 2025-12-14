#!/usr/bin/env python3
"""
Restore all pairwise comparison data - imports both SY and AJ data
"""

import csv
import sys
import io
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

# AJ's data embedded directly in the script
AJ_DATA_CSV = """Image1_Path,Image1_Name,Image2_Path,Image2_Name,Reconstruction_Type,Winner,Labeler_Name,Notes,Created_At
Trajectories/2022-02-06/123_RR220206A_894C_1923.bmp,123_RR220206A_894C_1923.bmp,Trajectories/2022-02-04/117_RR220204A_445C_1740.bmp,117_RR220204A_445C_1740.bmp,c(6 x 2),2,AJ,"",2025-12-06 01:24:55
Trajectories/2022-02-06/123_RR220206A_894C_1923.bmp,123_RR220206A_894C_1923.bmp,Trajectories/2022-02-04/117_RR220204A_445C_1740.bmp,117_RR220204A_445C_1740.bmp,(1 x 1),1,AJ,"",2025-12-06 01:24:55
Trajectories/2022-02-04/034_RR220204A_655C_0142.bmp,034_RR220204A_655C_0142.bmp,Trajectories/2022-02-06/272_RR220206A_424C_2242.bmp,272_RR220206A_424C_2242.bmp,HTR,2,AJ,"",2025-12-06 01:24:31
Trajectories/2022-02-04/034_RR220204A_655C_0142.bmp,034_RR220204A_655C_0142.bmp,Trajectories/2022-02-06/272_RR220206A_424C_2242.bmp,272_RR220206A_424C_2242.bmp,c(6 x 2),1,AJ,"",2025-12-06 01:24:31
Trajectories/2022-04-11/359_RR220411A_675C_1926.bmp,359_RR220411A_675C_1926.bmp,Trajectories/2022-02-04/262_RR220204A_793C_1932.bmp,262_RR220204A_793C_1932.bmp,c(6 x 2),2,AJ,"",2025-12-06 01:24:10
Trajectories/2022-04-11/359_RR220411A_675C_1926.bmp,359_RR220411A_675C_1926.bmp,Trajectories/2022-02-04/262_RR220204A_793C_1932.bmp,262_RR220204A_793C_1932.bmp,(√13 x √13),1,AJ,"",2025-12-06 01:24:10
Trajectories/2022-02-04/309_RR220204A_911C_2027.bmp,309_RR220204A_911C_2027.bmp,Trajectories/2022-02-04/361_RR220204A_947C_2156.bmp,361_RR220204A_947C_2156.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:23:38
Trajectories/2022-02-06/098_RR220206A_656C_1838.bmp,098_RR220206A_656C_1838.bmp,Trajectories/2022-02-04/192_RR220204A_607C_1831.bmp,192_RR220204A_607C_1831.bmp,(1 x 1),1,AJ,"",2025-12-06 01:23:17
Trajectories/2022-02-06/098_RR220206A_656C_1838.bmp,098_RR220206A_656C_1838.bmp,Trajectories/2022-02-04/192_RR220204A_607C_1831.bmp,192_RR220204A_607C_1831.bmp,(√13 x √13),2,AJ,"",2025-12-06 01:23:17
Trajectories/2022-02-06/207_RR220206A_875C_2114.bmp,207_RR220206A_875C_2114.bmp,Trajectories/2022-02-06/104_RR220206A_734C_1847.bmp,104_RR220206A_734C_1847.bmp,HTR,1,AJ,"",2025-12-06 01:22:53
Trajectories/2022-02-06/207_RR220206A_875C_2114.bmp,207_RR220206A_875C_2114.bmp,Trajectories/2022-02-06/104_RR220206A_734C_1847.bmp,104_RR220206A_734C_1847.bmp,(1 x 1),2,AJ,"",2025-12-06 01:22:53
Trajectories/2022-04-11/120_RR220411A_555C_1553.bmp,120_RR220411A_555C_1553.bmp,Trajectories/2022-02-04/215_RR220204A_669C_1848.bmp,215_RR220204A_669C_1848.bmp,(1 x 1),1,AJ,"",2025-12-06 01:22:37
Trajectories/2022-04-11/120_RR220411A_555C_1553.bmp,120_RR220411A_555C_1553.bmp,Trajectories/2022-02-04/215_RR220204A_669C_1848.bmp,215_RR220204A_669C_1848.bmp,(√13 x √13),2,AJ,"",2025-12-06 01:22:37
Trajectories/2022-04-11/332_RR220411A_756C_1911.bmp,332_RR220411A_756C_1911.bmp,Trajectories/2022-04-11/320_RR220411A_824C_1901.bmp,320_RR220411A_824C_1901.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:22:26
Trajectories/2022-04-11/258_RR220411A_932C_1751.bmp,258_RR220411A_932C_1751.bmp,Trajectories/2022-04-11/246_RR220411A_915C_1741.bmp,246_RR220411A_915C_1741.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:22:11
Trajectories/2022-02-04/063_RR220204A_165C_0215-03.bmp,063_RR220204A_165C_0215-03.bmp,Trajectories/2022-04-11/376_RR220411A_620C_1938.bmp,376_RR220411A_620C_1938.bmp,Other,1,AJ,"",2025-12-06 01:21:25
Trajectories/2022-02-04/063_RR220204A_165C_0215-03.bmp,063_RR220204A_165C_0215-03.bmp,Trajectories/2022-04-11/376_RR220411A_620C_1938.bmp,376_RR220411A_620C_1938.bmp,HTR,2,AJ,"",2025-12-06 01:21:25
Trajectories/2022-02-04/382_RR220204A_935C_2330.bmp,382_RR220204A_935C_2330.bmp,Trajectories/2022-02-04/085_RR220204A_378C_1720.bmp,085_RR220204A_378C_1720.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:21:05
Trajectories/2022-02-04/074_RR220204A_297C_1711.bmp,074_RR220204A_297C_1711.bmp,Trajectories/2022-02-04/037_RR220204A_579C_0146.bmp,037_RR220204A_579C_0146.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:20:43
Trajectories/2022-02-04/287_RR220204A_851C_2008.bmp,287_RR220204A_851C_2008.bmp,Trajectories/2022-04-11/409_RR220411A_474C_1953.bmp,409_RR220411A_474C_1953.bmp,(√13 x √13),2,AJ,"",2025-12-06 01:20:27
Trajectories/2022-02-04/287_RR220204A_851C_2008.bmp,287_RR220204A_851C_2008.bmp,Trajectories/2022-04-11/409_RR220411A_474C_1953.bmp,409_RR220411A_474C_1953.bmp,c(6 x 2),1,AJ,"",2025-12-06 01:20:27
Trajectories/2022-02-06/163_RR220206A_1105C_2015.bmp,163_RR220206A_1105C_2015.bmp,Trajectories/2022-02-04/341_RR220204A_937C_2114.bmp,341_RR220204A_937C_2114.bmp,(√13 x √13),2,AJ,"",2025-12-06 01:19:50
Trajectories/2022-02-06/163_RR220206A_1105C_2015.bmp,163_RR220206A_1105C_2015.bmp,Trajectories/2022-02-04/341_RR220204A_937C_2114.bmp,341_RR220204A_937C_2114.bmp,HTR,1,AJ,"",2025-12-06 01:19:50
Trajectories/2022-04-11/379_RR220411A_614C_1940.bmp,379_RR220411A_614C_1940.bmp,Trajectories/2022-02-06/273_RR220206A_423C_2243.bmp,273_RR220206A_423C_2243.bmp,HTR,2,AJ,"",2025-12-06 01:19:39
Trajectories/2022-04-11/379_RR220411A_614C_1940.bmp,379_RR220411A_614C_1940.bmp,Trajectories/2022-02-06/273_RR220206A_423C_2243.bmp,273_RR220206A_423C_2243.bmp,(√13 x √13),1,AJ,"",2025-12-06 01:19:39
Trajectories/2022-04-11/363_RR220411A_661C_1929.bmp,363_RR220411A_661C_1929.bmp,Trajectories/2022-02-04/147_RR220204A_486C_1804.bmp,147_RR220204A_486C_1804.bmp,(√13 x √13),tie,AJ,"",2025-12-06 01:19:25
Trajectories/2022-02-06/058_RR220206A_249C_1745.bmp,058_RR220206A_249C_1745.bmp,Trajectories/2022-04-11/178_RR220411A_774C_1649.bmp,178_RR220411A_774C_1649.bmp,(1 x 1),2,AJ,"",2025-12-06 01:18:28
Trajectories/2022-02-06/058_RR220206A_249C_1745.bmp,058_RR220206A_249C_1745.bmp,Trajectories/2022-04-11/178_RR220411A_774C_1649.bmp,178_RR220411A_774C_1649.bmp,Other,1,AJ,"",2025-12-06 01:18:28
Trajectories/2022-02-06/171_RR220206A_1085C_2027.bmp,171_RR220206A_1085C_2027.bmp,Trajectories/2022-02-04/038_RR220204A_576C_0147.bmp,038_RR220204A_576C_0147.bmp,(√13 x √13),2,AJ,"",2025-12-06 01:18:06
Trajectories/2022-02-06/171_RR220206A_1085C_2027.bmp,171_RR220206A_1085C_2027.bmp,Trajectories/2022-02-04/038_RR220204A_576C_0147.bmp,038_RR220204A_576C_0147.bmp,HTR,1,AJ,"",2025-12-06 01:18:05
Trajectories/2022-02-06/057_RR220206A_249C_1744.bmp,057_RR220206A_249C_1744.bmp,Trajectories/2022-02-06/154_RR220206A_1092C_2003-02.bmp,154_RR220206A_1092C_2003-02.bmp,(1 x 1),1,AJ,"",2025-12-06 01:17:47
Trajectories/2022-02-06/057_RR220206A_249C_1744.bmp,057_RR220206A_249C_1744.bmp,Trajectories/2022-02-06/154_RR220206A_1092C_2003-02.bmp,154_RR220206A_1092C_2003-02.bmp,HTR,2,AJ,"",2025-12-06 01:17:47"""

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
    
    # Import AJ's data (embedded in script)
    print("\n=== Importing AJ's data ===")
    reader = csv.DictReader(io.StringIO(AJ_DATA_CSV))
    aj_count = 0
    for row in reader:
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
