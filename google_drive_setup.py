"""
Script to generate a file mapping for Google Drive images.
This creates a JSON file mapping image paths to Google Drive file IDs.

Usage:
1. Make sure your Google Drive folder is public
2. Get a Google Drive API key (optional but recommended)
3. Run this script to generate image_list.json
4. The app will use this file to serve images
"""

import os
import json
from pathlib import Path
from googleapiclient.discovery import build

# Configuration
GOOGLE_DRIVE_FOLDER_ID = '1lRe2jJhgC8Pd6RsCKEirNK4NlWmC1QJU'
API_KEY = os.environ.get('GOOGLE_DRIVE_API_KEY', '')
OUTPUT_FILE = Path(__file__).parent / 'google_drive_images.json'

def get_all_images_from_drive(folder_id, api_key=None):
    """Recursively get all image files from Google Drive folder"""
    if not api_key:
        print("No API key provided. Using public folder access method.")
        return []
    
    try:
        service = build('drive', 'v3', developerKey=api_key)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        all_files = []
        
        def get_files_recursive(folder_id, parent_path=''):
            files = []
            query = f"'{folder_id}' in parents and trashed=false"
            page_token = None
            
            # Handle pagination - Google Drive API returns max 100 items per page
            while True:
                request_params = {
                    'q': query,
                    'fields': 'nextPageToken, files(id, name, mimeType)',
                    'pageSize': 1000  # Maximum allowed by API
                }
                if page_token:
                    request_params['pageToken'] = page_token
                
                results = service.files().list(**request_params).execute()
                items = results.get('files', [])
                
                for item in items:
                    mime_type = item.get('mimeType', '')
                    
                    if 'image' in mime_type.lower():
                        # It's an image file
                        ext = Path(item['name']).suffix.lower()
                        if ext in image_extensions:
                            rel_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
                            files.append({
                                'path': rel_path,
                                'name': item['name'],
                                'file_id': item['id'],
                                'mime_type': mime_type
                            })
                    elif mime_type == 'application/vnd.google-apps.folder':
                        # It's a folder, recurse
                        folder_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
                        subfolder_files = get_files_recursive(item['id'], folder_path)
                        files.extend(subfolder_files)
                
                # Check if there are more pages
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            return files
        
        all_files = get_files_recursive(folder_id)
        return sorted(all_files, key=lambda x: x['name'])
        
    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        return []

def main():
    print("Generating Google Drive image list...")
    
    if not API_KEY:
        print("\n⚠️  No GOOGLE_DRIVE_API_KEY environment variable set.")
        print("You can get an API key from: https://console.cloud.google.com/apis/credentials")
        print("Or set it with: export GOOGLE_DRIVE_API_KEY='your-key-here'")
        print("\nContinuing without API key (will create empty list)...")
    
    images = get_all_images_from_drive(GOOGLE_DRIVE_FOLDER_ID, API_KEY)
    
    # Create mapping: path -> file_id for quick lookup
    image_map = {}
    for img in images:
        image_map[img['path']] = {
            'file_id': img['file_id'],
            'name': img['name'],
            'mime_type': img.get('mime_type', '')
        }
    
    # Save to JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({
            'folder_id': GOOGLE_DRIVE_FOLDER_ID,
            'images': image_map,
            'total_count': len(images)
        }, f, indent=2)
    
    print(f"\n✅ Generated {len(images)} image mappings")
    print(f"📁 Saved to: {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"1. Commit this file to your repo")
    print(f"2. Set USE_GOOGLE_DRIVE=true in Render environment variables")
    print(f"3. Deploy!")

if __name__ == '__main__':
    main()

