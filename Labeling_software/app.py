from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import mimetypes

app = Flask(__name__)
CORS(app)

# App version - increment to force Render cache refresh
APP_VERSION = "2.2.0"
print(f"=== RHEED Labeling Software v{APP_VERSION} starting ===")

# Configuration
# For local development, use local data directory
# For cloud, images can be in Google Drive or cloud storage
LOCAL_DATA_DIR = Path(__file__).parent.parent / "data"
GOOGLE_DRIVE_FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', '1lRe2jJhgC8Pd6RsCKEirNK4NlWmC1QJU')
USE_GOOGLE_DRIVE = os.environ.get('USE_GOOGLE_DRIVE', 'false').lower() == 'true'
GOOGLE_DRIVE_IMAGES_FILE = Path(__file__).parent / 'google_drive_images.json'

# Use local data if available, otherwise will use Google Drive
if LOCAL_DATA_DIR.exists() and list(LOCAL_DATA_DIR.glob('*')):
    DATA_DIR = LOCAL_DATA_DIR
    USE_GOOGLE_DRIVE = False
else:
    DATA_DIR = None

# Load Google Drive image mapping if available
GOOGLE_DRIVE_IMAGE_MAP = {}
# Cache for file IDs fetched via API (path -> file_id)
GOOGLE_DRIVE_FILE_ID_CACHE = {}

# Always load the image map if the file exists (needed for cloud deployment)
if GOOGLE_DRIVE_IMAGES_FILE.exists():
    try:
        import json
        with open(GOOGLE_DRIVE_IMAGES_FILE, 'r') as f:
            data = json.load(f)
            GOOGLE_DRIVE_IMAGE_MAP = data.get('images', {})
            print(f"Loaded {len(GOOGLE_DRIVE_IMAGE_MAP)} images from Google Drive mapping")
    except Exception as e:
        print(f"Error loading Google Drive image map: {e}")

# If no local data, enable Google Drive mode
if DATA_DIR is None and GOOGLE_DRIVE_IMAGE_MAP:
    USE_GOOGLE_DRIVE = True
    print("No local data directory - using Google Drive mode")

# Database configuration - use PostgreSQL in cloud, SQLite locally
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # Cloud deployment - use PostgreSQL
    import psycopg2
    from psycopg2.extras import RealDictCursor
    USE_POSTGRES = True
    # Parse DATABASE_URL (format: postgresql://user:pass@host:port/dbname)
    DB_PATH = DATABASE_URL
else:
    # Local development - use SQLite
    import sqlite3
    USE_POSTGRES = False
    DB_PATH = Path(__file__).parent / "labels.db"

# Database connection helper
def get_db_connection():
    if USE_POSTGRES:
        conn = psycopg2.connect(DB_PATH, sslmode='require')
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

# Initialize database
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    if USE_POSTGRES:
        c.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id SERIAL PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                quality TEXT,
                reconstruction TEXT,
                reconstruction_scores TEXT,
                labeler_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add new columns if they don't exist (for existing databases)
        try:
            c.execute('ALTER TABLE labels ADD COLUMN IF NOT EXISTS labeler_name TEXT')
            c.execute('ALTER TABLE labels ADD COLUMN IF NOT EXISTS reconstruction_scores TEXT')
            c.execute('ALTER TABLE labels ADD COLUMN IF NOT EXISTS notes TEXT')
        except:
            pass  # Columns might already exist
        
        # Create pairwise comparisons table
        c.execute('''
            CREATE TABLE IF NOT EXISTS pairwise_comparisons (
                id SERIAL PRIMARY KEY,
                image1_path TEXT NOT NULL,
                image1_name TEXT NOT NULL,
                image2_path TEXT NOT NULL,
                image2_name TEXT NOT NULL,
                reconstruction_type TEXT NOT NULL,
                winner TEXT NOT NULL,
                labeler_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add new columns for brightness and confidence (for existing databases)
        try:
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN IF NOT EXISTS brightness_1 REAL')
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN IF NOT EXISTS brightness_2 REAL')
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN IF NOT EXISTS confidence TEXT')
        except:
            pass  # Columns might already exist
    else:
        c.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                quality TEXT,
                reconstruction TEXT,
                reconstruction_scores TEXT,
                labeler_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add new columns if they don't exist (for existing databases)
        try:
            c.execute('ALTER TABLE labels ADD COLUMN labeler_name TEXT')
        except:
            pass
        try:
            c.execute('ALTER TABLE labels ADD COLUMN reconstruction_scores TEXT')
        except:
            pass
        try:
            c.execute('ALTER TABLE labels ADD COLUMN notes TEXT')
        except:
            pass
        
        # Create pairwise comparisons table
        c.execute('''
            CREATE TABLE IF NOT EXISTS pairwise_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image1_path TEXT NOT NULL,
                image1_name TEXT NOT NULL,
                image2_path TEXT NOT NULL,
                image2_name TEXT NOT NULL,
                reconstruction_type TEXT NOT NULL,
                winner TEXT NOT NULL,
                labeler_name TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add new columns for brightness and confidence (for existing databases)
        try:
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN brightness_1 REAL')
        except:
            pass
        try:
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN brightness_2 REAL')
        except:
            pass
        try:
            c.execute('ALTER TABLE pairwise_comparisons ADD COLUMN confidence TEXT')
        except:
            pass

    conn.commit()
    conn.close()

# Get all image files from data directory or Google Drive
# Only include images from specific trajectory folders
ALLOWED_TRAJECTORY_FOLDERS = ['2022-02-04', '2022-02-06', '2022-04-11', '2025-10-04', '2025-10-05']
# Folders that should use Google Drive (even if local data exists)
GOOGLE_DRIVE_ONLY_FOLDERS = ['2025-10-04', '2025-10-05']
# Folders that can use local data
LOCAL_DATA_FOLDERS = ['2022-02-04', '2022-02-06', '2022-04-11']

def get_image_files():
    """
    Hybrid approach:
    - 2022 folders: Use local data if available, otherwise Google Drive
    - 2025 folders: Always use Google Drive (has correct filenames)
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_files = []

    # Get 2022 images from local data if available
    if DATA_DIR and DATA_DIR.exists():
        trajectories_dir = DATA_DIR / 'Trajectories'
        if trajectories_dir.exists():
            for folder_name in LOCAL_DATA_FOLDERS:
                folder_path = trajectories_dir / folder_name
                if folder_path.exists():
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            if Path(file).suffix.lower() in image_extensions:
                                full_path = os.path.join(root, file)
                                rel_path = os.path.relpath(full_path, DATA_DIR)
                                rel_path = rel_path.replace('\\', '/')
                                image_files.append({
                                    'path': rel_path,
                                    'name': file,
                                    'full_path': full_path,
                                    'source': 'local'
                                })
    elif GOOGLE_DRIVE_IMAGE_MAP:
        # No local data - get 2022 from Google Drive
        for path, info in GOOGLE_DRIVE_IMAGE_MAP.items():
            if any(folder in path for folder in LOCAL_DATA_FOLDERS):
                file_id = info.get('file_id')
                if file_id:
                    GOOGLE_DRIVE_FILE_ID_CACHE[path] = file_id
                image_files.append({
                    'path': path,
                    'name': info.get('name', Path(path).name),
                    'file_id': file_id,
                    'full_path': path,
                    'source': 'google_drive'
                })

    # Always get 2025 images from Google Drive (has correct filenames)
    if GOOGLE_DRIVE_IMAGE_MAP:
        for path, info in GOOGLE_DRIVE_IMAGE_MAP.items():
            if any(folder in path for folder in GOOGLE_DRIVE_ONLY_FOLDERS):
                file_id = info.get('file_id')
                if file_id:
                    GOOGLE_DRIVE_FILE_ID_CACHE[path] = file_id
                image_files.append({
                    'path': path,
                    'name': info.get('name', Path(path).name),
                    'file_id': file_id,
                    'full_path': path,
                    'source': 'google_drive'
                })

    return sorted(image_files, key=lambda x: x['name'])

def get_google_drive_images():
    """Fetch image list from Google Drive using cached mapping file"""
    # Use the pre-generated mapping file for fast access
    if GOOGLE_DRIVE_IMAGE_MAP:
        image_files = []
        for path, info in GOOGLE_DRIVE_IMAGE_MAP.items():
            file_id = info.get('file_id')
            # Cache file_id for later use
            if file_id:
                GOOGLE_DRIVE_FILE_ID_CACHE[path] = file_id
            image_files.append({
                'path': path,
                'name': info.get('name', Path(path).name),
                'file_id': file_id,
                'full_path': path
            })
        return sorted(image_files, key=lambda x: x['name'])
    
    # If no mapping file, try to generate it on the fly (slower)
    images = get_google_drive_images_api()
    # Cache file_ids from API response
    for img in images:
        if img.get('file_id'):
            GOOGLE_DRIVE_FILE_ID_CACHE[img['path']] = img['file_id']
    return images

def get_google_drive_images_api():
    """Fetch image list from Google Drive API (requires API key)"""
    try:
        from googleapiclient.discovery import build
        
        API_KEY = os.environ.get('GOOGLE_DRIVE_API_KEY', '')
        if not API_KEY:
            print("No Google Drive API key. Use google_drive_setup.py to generate image mapping.")
            return []
        
        service = build('drive', 'v3', developerKey=API_KEY)
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        image_files = []
        
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
                        ext = Path(item['name']).suffix.lower()
                        if ext in image_extensions:
                            rel_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
                            files.append({
                                'path': rel_path,
                                'name': item['name'],
                                'file_id': item['id'],
                                'full_path': rel_path
                            })
                    elif mime_type == 'application/vnd.google-apps.folder':
                        folder_path = f"{parent_path}/{item['name']}" if parent_path else item['name']
                        files.extend(get_files_recursive(item['id'], folder_path))
                
                # Check if there are more pages
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            return files
        
        image_files = get_files_recursive(GOOGLE_DRIVE_FOLDER_ID)
        return sorted(image_files, key=lambda x: x['name'])
        
    except Exception as e:
        print(f"Google Drive API error: {e}")
        return []

def get_file_id_from_api(image_path):
    """Fetch file ID for a specific image path from Google Drive API"""
    try:
        from googleapiclient.discovery import build
        
        API_KEY = os.environ.get('GOOGLE_DRIVE_API_KEY', '')
        if not API_KEY:
            return None
        
        service = build('drive', 'v3', developerKey=API_KEY)
        
        # Extract filename from path
        filename = Path(image_path).name
        
        # Search for file by name in the folder
        query = f"name='{filename}' and '{GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            # Return first match (or could match by full path if needed)
            return items[0]['id']
        
        return None
    except Exception as e:
        print(f"Error fetching file ID from API: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check deployment status"""
    sample_2025_images = [k for k in GOOGLE_DRIVE_IMAGE_MAP.keys() if '2025-10-04' in k][:5]
    # Count images by source
    all_images = get_image_files()
    local_count = sum(1 for img in all_images if img.get('source') == 'local')
    gdrive_count = sum(1 for img in all_images if img.get('source') == 'google_drive')
    return jsonify({
        'version': APP_VERSION,
        'mode': 'hybrid',
        'local_folders': LOCAL_DATA_FOLDERS,
        'google_drive_folders': GOOGLE_DRIVE_ONLY_FOLDERS,
        'data_dir': str(DATA_DIR) if DATA_DIR else None,
        'total_images': len(all_images),
        'local_images': local_count,
        'google_drive_images': gdrive_count,
        'total_in_gdrive_map': len(GOOGLE_DRIVE_IMAGE_MAP),
        'sample_2025_images': sample_2025_images
    })

@app.route('/api/images')
def get_images():
    """Get list of all available images"""
    images = get_image_files()
    return jsonify(images)

@app.route('/api/images/<path:image_path>')
def serve_image(image_path):
    """Serve image files from local storage or Google Drive (hybrid mode)"""
    # Flask automatically URL-decodes the path, but we need to handle it properly
    # Normalize path separators
    image_path = image_path.replace('\\', '/')
    try:
        # Decode URL encoding if needed
        import urllib.parse
        image_path = urllib.parse.unquote(image_path)

        # Check if this is a 2025 folder - always use Google Drive for these
        is_2025_folder = any(folder in image_path for folder in GOOGLE_DRIVE_ONLY_FOLDERS)

        if is_2025_folder and GOOGLE_DRIVE_IMAGE_MAP:
            # Always serve 2025 images from Google Drive
            return serve_google_drive_image(image_path)
        elif DATA_DIR and DATA_DIR.exists():
            # Try to serve from local directory first (for 2022 folders)
            full_path = os.path.join(DATA_DIR, image_path)
            if os.path.exists(full_path):
                # Determine MIME type
                mime_type, _ = mimetypes.guess_type(full_path)
                if not mime_type:
                    # Default MIME types for common image formats
                    ext = Path(full_path).suffix.lower()
                    mime_map = {'.bmp': 'image/bmp', '.png': 'image/png', '.jpg': 'image/jpeg',
                               '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
                    mime_type = mime_map.get(ext, 'application/octet-stream')

                with open(full_path, 'rb') as f:
                    image_data = f.read()

                response = Response(image_data, mimetype=mime_type)
                # Add aggressive cache control headers for better performance
                response.headers['Cache-Control'] = 'public, max-age=86400, immutable'  # Cache for 1 day
                response.headers['Expires'] = (datetime.now() + timedelta(days=1)).strftime('%a, %d %b %Y %H:%M:%S GMT')
                return response
            elif GOOGLE_DRIVE_IMAGE_MAP:
                # Fallback to Google Drive if local file not found
                return serve_google_drive_image(image_path)
            else:
                return f"Image not found: {image_path}", 404
        elif GOOGLE_DRIVE_IMAGE_MAP:
            # No local data - serve everything from Google Drive
            return serve_google_drive_image(image_path)
        else:
            return "No image source configured", 404
    except Exception as e:
        import traceback
        return f"Error serving image: {str(e)}\n{traceback.format_exc()}", 404

def serve_google_drive_image(image_path):
    """Serve image from Google Drive using direct download link"""
    try:
        import requests
        
        # Look up file ID from multiple sources
        file_id = None
        
        # 1. Check cache (populated from API or mapping file)
        if image_path in GOOGLE_DRIVE_FILE_ID_CACHE:
            file_id = GOOGLE_DRIVE_FILE_ID_CACHE[image_path]
        # 2. Check mapping file
        elif image_path in GOOGLE_DRIVE_IMAGE_MAP:
            file_id = GOOGLE_DRIVE_IMAGE_MAP[image_path].get('file_id')
            if file_id:
                GOOGLE_DRIVE_FILE_ID_CACHE[image_path] = file_id
        # 3. Direct file ID provided
        elif image_path.startswith('id:'):
            file_id = image_path[3:]
        # 4. Try to fetch from API on demand (slower)
        else:
            file_id = get_file_id_from_api(image_path)
            if file_id:
                GOOGLE_DRIVE_FILE_ID_CACHE[image_path] = file_id
        
        if not file_id:
            return f"Image not found: {image_path}. File ID not available.", 404
        
        # Use direct download link for public files
        # Format: https://drive.google.com/uc?export=download&id=FILE_ID
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Fetch image from Google Drive
        response = requests.get(download_url, allow_redirects=True, timeout=30, stream=True)
        
        if response.status_code == 200:
            # Determine MIME type from extension or content
            ext = Path(image_path).suffix.lower()
            mime_map = {'.bmp': 'image/bmp', '.png': 'image/png', '.jpg': 'image/jpeg', 
                       '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
            mime_type = mime_map.get(ext, response.headers.get('Content-Type', 'image/png'))
            
            # Stream the response for better performance
            def generate():
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            
            flask_response = Response(generate(), mimetype=mime_type)
            # Add aggressive cache control headers for Google Drive images
            flask_response.headers['Cache-Control'] = 'public, max-age=86400, immutable'  # Cache for 1 day
            flask_response.headers['Expires'] = (datetime.now() + timedelta(days=1)).strftime('%a, %d %b %Y %H:%M:%S GMT')
            return flask_response
        else:
            return f"Failed to fetch image from Google Drive: {response.status_code}", 404
            
    except Exception as e:
        import traceback
        return f"Error serving Google Drive image: {str(e)}\n{traceback.format_exc()}", 500

@app.route('/api/labels', methods=['GET'])
def get_labels():
    """Get all labels"""
    conn = get_db_connection()
    
    if USE_POSTGRES:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT * FROM labels ORDER BY updated_at DESC')
        labels = [dict(row) for row in c.fetchall()]
    else:
        c = conn.cursor()
        c.execute('SELECT * FROM labels ORDER BY updated_at DESC')
        labels = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return jsonify(labels)

@app.route('/api/labels', methods=['POST'])
def save_label():
    """Save or update a label"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        file_path = data.get('file_path')
        file_name = data.get('file_name')
        quality = data.get('quality')
        reconstruction = data.get('reconstruction')  # Can be a list or single value
        reconstruction_scores = data.get('reconstruction_scores')  # Dict mapping reconstruction to score
        labeler_name = data.get('labeler_name', '').strip()
        notes = data.get('notes', '').strip() if data.get('notes') else ''
        
        if not file_path or not file_name:
            return jsonify({'error': 'file_path and file_name are required'}), 400
        
        # Convert reconstruction to JSON string if it's a list
        if isinstance(reconstruction, list):
            reconstruction_json = json.dumps(reconstruction)
        elif reconstruction:
            reconstruction_json = json.dumps([reconstruction])  # Single value as list
        else:
            reconstruction_json = None
        
        # Convert reconstruction_scores to JSON string
        if isinstance(reconstruction_scores, dict):
            scores_json = json.dumps(reconstruction_scores)
        elif reconstruction_scores:
            # Try to convert if it's a string
            try:
                if isinstance(reconstruction_scores, str):
                    scores_json = reconstruction_scores
                else:
                    scores_json = json.dumps(reconstruction_scores)
            except:
                scores_json = None
        else:
            scores_json = None
        
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            # Check if label exists
            if USE_POSTGRES:
                c.execute('SELECT id FROM labels WHERE file_path = %s', (file_path,))
            else:
                c.execute('SELECT id FROM labels WHERE file_path = ?', (file_path,))
            existing = c.fetchone()
            
            if existing:
                # Update existing label
                if USE_POSTGRES:
                    c.execute('''
                        UPDATE labels 
                        SET quality = %s, reconstruction = %s, reconstruction_scores = %s, 
                            labeler_name = %s, notes = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE file_path = %s
                    ''', (quality, reconstruction_json, scores_json, labeler_name, notes, file_path))
                else:
                    c.execute('''
                        UPDATE labels 
                        SET quality = ?, reconstruction = ?, reconstruction_scores = ?,
                            labeler_name = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE file_path = ?
                    ''', (quality, reconstruction_json, scores_json, labeler_name, notes, file_path))
            else:
                # Insert new label
                if USE_POSTGRES:
                    c.execute('''
                        INSERT INTO labels (file_path, file_name, quality, reconstruction, reconstruction_scores, labeler_name, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (file_path, file_name, quality, reconstruction_json, scores_json, labeler_name, notes))
                else:
                    c.execute('''
                        INSERT INTO labels (file_path, file_name, quality, reconstruction, reconstruction_scores, labeler_name, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (file_path, file_name, quality, reconstruction_json, scores_json, labeler_name, notes))
            
            conn.commit()
            
            # Sync to Google Sheets if configured (before closing connection)
            try:
                from google_sheets_sync import sync_absolute_to_sheets
                # Get the saved record to sync
                if USE_POSTGRES:
                    c.execute('SELECT * FROM labels WHERE file_path = %s', (file_path,))
                else:
                    c.execute('SELECT * FROM labels WHERE file_path = ?', (file_path,))
                record = c.fetchone()
                
                if record:
                    if USE_POSTGRES:
                        from psycopg2.extras import RealDictCursor
                        # Re-fetch with dict cursor for easier access
                        dict_cursor = conn.cursor(cursor_factory=RealDictCursor)
                        dict_cursor.execute('SELECT * FROM labels WHERE file_path = %s', (file_path,))
                        record_dict = dict(dict_cursor.fetchone())
                        dict_cursor.close()
                    else:
                        # Convert SQLite row to dict
                        columns = [desc[0] for desc in c.description]
                        record_dict = dict(zip(columns, record))
                    
                    sync_absolute_to_sheets(record_dict)
            except ImportError:
                # Google Sheets sync not available, skip silently
                pass
            except Exception as sync_error:
                # Don't fail if Google Sheets sync fails
                import traceback
                print(f"Warning: Google Sheets sync failed: {sync_error}\n{traceback.format_exc()}")
            
            return jsonify({'success': True})
        except Exception as db_error:
            conn.rollback()
            import traceback
            error_msg = str(db_error)
            print(f"Database error in save_label: {error_msg}\n{traceback.format_exc()}")
            return jsonify({'error': f'Database error: {error_msg}'}), 500
        finally:
            conn.close()
    except Exception as e:
        import traceback
        print(f"Error in save_label: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/labels/<path:file_path>', methods=['GET'])
def get_label(file_path):
    """Get label for a specific file"""
    import urllib.parse
    file_path = urllib.parse.unquote(file_path)
    
    conn = get_db_connection()
    
    if USE_POSTGRES:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT * FROM labels WHERE file_path = %s', (file_path,))
        row = c.fetchone()
        if row:
            label_data = dict(row)
            # Parse JSON fields
            if label_data.get('reconstruction'):
                try:
                    label_data['reconstruction'] = json.loads(label_data['reconstruction'])
                except:
                    pass
            if label_data.get('reconstruction_scores'):
                try:
                    label_data['reconstruction_scores'] = json.loads(label_data['reconstruction_scores'])
                except:
                    pass
            conn.close()
            return jsonify(label_data)
    else:
        c = conn.cursor()
        c.execute('SELECT * FROM labels WHERE file_path = ?', (file_path,))
        row = c.fetchone()
        if row:
            label_data = dict(row)
            # Parse JSON fields
            if label_data.get('reconstruction'):
                try:
                    label_data['reconstruction'] = json.loads(label_data['reconstruction'])
                except:
                    pass
            if label_data.get('reconstruction_scores'):
                try:
                    label_data['reconstruction_scores'] = json.loads(label_data['reconstruction_scores'])
                except:
                    pass
            conn.close()
            return jsonify(label_data)
    
    conn.close()
    return jsonify({'quality': None, 'reconstruction': None, 'reconstruction_scores': None, 'labeler_name': None, 'notes': None})

@app.route('/api/labels/<path:file_path>', methods=['DELETE'])
def delete_label(file_path):
    """Delete a label"""
    import urllib.parse
    file_path = urllib.parse.unquote(file_path)
    
    conn = get_db_connection()
    c = conn.cursor()
    
    if USE_POSTGRES:
        c.execute('DELETE FROM labels WHERE file_path = %s', (file_path,))
    else:
        c.execute('DELETE FROM labels WHERE file_path = ?', (file_path,))
    
    deleted = c.rowcount
    conn.commit()
    
    # Delete from Google Sheets if configured
    if deleted > 0:
        try:
            from google_sheets_sync import delete_absolute_from_sheets
            delete_absolute_from_sheets(file_path)
        except ImportError:
            pass
        except Exception as sync_error:
            import traceback
            print(f"Warning: Google Sheets delete failed: {sync_error}\n{traceback.format_exc()}")
    
    conn.close()
    
    if deleted > 0:
        return jsonify({'success': True, 'message': 'Label deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Label not found'}), 404

@app.route('/api/labels/export', methods=['GET'])
def export_labels():
    """Export all labels as CSV - matching Google Sheets format"""
    conn = get_db_connection()

    try:
        # Get all labels
        if USE_POSTGRES:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute('SELECT * FROM labels ORDER BY created_at DESC')
            labels = [dict(row) for row in c.fetchall()]
        else:
            c = conn.cursor()
            c.execute('PRAGMA table_info(labels)')
            columns = [row[1] for row in c.fetchall()]
            c.execute('SELECT * FROM labels ORDER BY created_at DESC')
            labels = []
            for row in c.fetchall():
                label_dict = {}
                for i, col in enumerate(columns):
                    label_dict[col] = row[i]
                labels.append(label_dict)

        conn.close()

        if not labels:
            return 'File_Path,File_Name,Quality,Reconstruction,Reconstruction_Scores,Labeler_Name,Notes,Created_At,Updated_At\nNo labels found', 200, {'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': 'attachment; filename="labels_export.csv"'}

        # CSV header matching Google Sheets exactly
        csv_lines = ['File_Path,File_Name,Quality,Reconstruction,Reconstruction_Scores,Labeler_Name,Notes,Created_At,Updated_At']

        for label in labels:
            file_path = label.get('file_path', '') or ''
            file_name = label.get('file_name', '') or ''
            quality = label.get('quality', '') or ''
            reconstruction = label.get('reconstruction', '') or ''
            reconstruction_scores = label.get('reconstruction_scores', '') or ''
            labeler_name = label.get('labeler_name', '') or ''
            notes = label.get('notes', '') or ''
            created_at = str(label.get('created_at', '') or '')
            updated_at = str(label.get('updated_at', '') or '')

            # Escape quotes for CSV
            notes_escaped = notes.replace('"', '""')
            reconstruction_escaped = reconstruction.replace('"', '""')
            reconstruction_scores_escaped = reconstruction_scores.replace('"', '""')

            csv_lines.append(f'"{file_path}","{file_name}","{quality}","{reconstruction_escaped}",'
                           f'"{reconstruction_scores_escaped}","{labeler_name}","{notes_escaped}",'
                           f'"{created_at}","{updated_at}"')

        return '\n'.join(csv_lines), 200, {'Content-Type': 'text/csv; charset=utf-8',
                                            'Content-Disposition': 'attachment; filename="labels_export.csv"'}
    except Exception as e:
        conn.close()
        import traceback
        return f'Error exporting labels: {str(e)}\n{traceback.format_exc()}', 500

@app.route('/api/pairwise', methods=['POST'])
def save_pairwise_comparison():
    """Save a pairwise comparison"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        image1_path = data.get('image1_path')
        image1_name = data.get('image1_name')
        image2_path = data.get('image2_path')
        image2_name = data.get('image2_name')
        reconstruction_type = data.get('reconstruction_type')
        winner = data.get('winner')  # '1', '2', 'tie', or 'not_apply'
        labeler_name = data.get('labeler_name', '').strip()
        notes = data.get('notes', '').strip() if data.get('notes') else ''
        # New fields for brightness and confidence
        brightness_1 = data.get('brightness_1')  # float or None
        brightness_2 = data.get('brightness_2')  # float or None
        confidence = data.get('confidence', 'Confident')  # Default to 'Confident'
        
        # Validate required fields
        if not all([image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner]):
            missing = []
            if not image1_path: missing.append('image1_path')
            if not image1_name: missing.append('image1_name')
            if not image2_path: missing.append('image2_path')
            if not image2_name: missing.append('image2_name')
            if not reconstruction_type: missing.append('reconstruction_type')
            if not winner: missing.append('winner')
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
        
        # Validate winner value
        if winner not in ['1', '2', 'tie', 'not_apply']:
            return jsonify({'error': f'Invalid winner value: {winner}. Must be "1", "2", "tie", or "not_apply"'}), 400
        
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            if USE_POSTGRES:
                c.execute('''
                    INSERT INTO pairwise_comparisons (image1_path, image1_name, image2_path, image2_name,
                                                     reconstruction_type, winner, labeler_name, notes,
                                                     brightness_1, brightness_2, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner, labeler_name, notes,
                      brightness_1, brightness_2, confidence))
            else:
                c.execute('''
                    INSERT INTO pairwise_comparisons (image1_path, image1_name, image2_path, image2_name,
                                                     reconstruction_type, winner, labeler_name, notes,
                                                     brightness_1, brightness_2, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner, labeler_name, notes,
                      brightness_1, brightness_2, confidence))
            
            conn.commit()
            
            # Sync to Google Sheets if configured (before closing connection)
            try:
                from google_sheets_sync import sync_pairwise_to_sheets
                from datetime import datetime
                sync_pairwise_to_sheets({
                    'image1_path': image1_path,
                    'image1_name': image1_name,
                    'image2_path': image2_path,
                    'image2_name': image2_name,
                    'reconstruction_type': reconstruction_type,
                    'winner': winner,
                    'labeler_name': labeler_name,
                    'notes': notes,
                    'created_at': datetime.now().isoformat(),
                    'brightness_1': brightness_1,
                    'brightness_2': brightness_2,
                    'confidence': confidence
                })
            except ImportError:
                # Google Sheets sync not available, skip silently
                pass
            except Exception as sync_error:
                # Don't fail if Google Sheets sync fails
                import traceback
                print(f"Warning: Google Sheets sync failed: {sync_error}\n{traceback.format_exc()}")
            
            return jsonify({'success': True})
        except Exception as db_error:
            conn.rollback()
            return jsonify({'error': f'Database error: {str(db_error)}'}), 500
        finally:
            conn.close()
    except Exception as e:
        import traceback
        print(f"Error in save_pairwise_comparison: {e}\n{traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/pairwise', methods=['GET'])
def get_pairwise_comparisons():
    """Get all pairwise comparisons"""
    try:
        conn = get_db_connection()
        
        # Check if table exists first
        if USE_POSTGRES:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'pairwise_comparisons'
                );
            """)
            table_exists = c.fetchone()[0]
        else:
            c = conn.cursor()
            c.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='pairwise_comparisons'
            """)
            table_exists = c.fetchone() is not None
        
        if not table_exists:
            conn.close()
            # Table doesn't exist - return empty list (init_db should create it, but handle gracefully)
            return jsonify([])
        
        # Table exists, query it
        if USE_POSTGRES:
            c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
            comparisons = [dict(row) for row in c.fetchall()]
        else:
            c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
            # Convert to list of dicts
            c.execute('PRAGMA table_info(pairwise_comparisons)')
            columns = [row[1] for row in c.fetchall()]
            c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
            comparisons = []
            for row in c.fetchall():
                comp_dict = {}
                for i, col in enumerate(columns):
                    comp_dict[col] = row[i]
                comparisons.append(comp_dict)
        
        conn.close()
        print(f"DEBUG: Returning {len(comparisons)} pairwise comparisons")
        return jsonify(comparisons)
    except Exception as e:
        import traceback
        error_msg = f"Error in get_pairwise_comparisons: {e}\n{traceback.format_exc()}"
        print(error_msg)
        # Log to stderr so it shows up in Flask logs
        import sys
        sys.stderr.write(error_msg + "\n")
        # Return empty list on error rather than crashing
        return jsonify([])

@app.route('/api/pairwise/<int:comp_id>', methods=['DELETE'])
def delete_pairwise_comparison(comp_id):
    """Delete a pairwise comparison"""
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get the record before deleting (for Google Sheets sync)
    if USE_POSTGRES:
        from psycopg2.extras import RealDictCursor
        dict_cursor = conn.cursor(cursor_factory=RealDictCursor)
        dict_cursor.execute('SELECT * FROM pairwise_comparisons WHERE id = %s', (comp_id,))
        record = dict_cursor.fetchone()
        dict_cursor.close()
    else:
        c.execute('PRAGMA table_info(pairwise_comparisons)')
        columns = [row[1] for row in c.fetchall()]
        c.execute('SELECT * FROM pairwise_comparisons WHERE id = ?', (comp_id,))
        row = c.fetchone()
        if row:
            record = dict(zip(columns, row))
        else:
            record = None
    
    if USE_POSTGRES:
        c.execute('DELETE FROM pairwise_comparisons WHERE id = %s', (comp_id,))
    else:
        c.execute('DELETE FROM pairwise_comparisons WHERE id = ?', (comp_id,))
    
    deleted = c.rowcount
    conn.commit()
    
    # Delete from Google Sheets if configured
    if deleted > 0 and record:
        try:
            from google_sheets_sync import delete_pairwise_from_sheets
            delete_pairwise_from_sheets(
                record.get('image1_path', ''),
                record.get('image2_path', ''),
                record.get('reconstruction_type', '')
            )
        except ImportError:
            pass
        except Exception as sync_error:
            import traceback
            print(f"Warning: Google Sheets delete failed: {sync_error}\n{traceback.format_exc()}")
    
    conn.close()
    
    if deleted > 0:
        return jsonify({'success': True, 'message': 'Comparison deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Comparison not found'}), 404

@app.route('/api/pairwise/export', methods=['GET'])
def export_pairwise_comparisons():
    """Export all pairwise comparisons as CSV - matching Google Sheets format"""
    conn = get_db_connection()

    try:
        # Get all comparisons
        if USE_POSTGRES:
            c = conn.cursor(cursor_factory=RealDictCursor)
            c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
            comparisons = [dict(row) for row in c.fetchall()]
        else:
            c = conn.cursor()
            c.execute('PRAGMA table_info(pairwise_comparisons)')
            columns = [row[1] for row in c.fetchall()]
            c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
            comparisons = []
            for row in c.fetchall():
                comp_dict = {}
                for i, col in enumerate(columns):
                    comp_dict[col] = row[i]
                comparisons.append(comp_dict)

        conn.close()

        if not comparisons:
            return 'Image1_Path,Image1_Name,Image2_Path,Image2_Name,Reconstruction_Type,Winner,Labeler_Name,Notes,Created_At\nNo pairwise comparisons found', 200, {'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': 'attachment; filename="pairwise_comparisons_export.csv"'}

        # CSV header matching Google Sheets exactly
        csv_lines = ['Image1_Path,Image1_Name,Image2_Path,Image2_Name,Reconstruction_Type,Winner,Labeler_Name,Notes,Created_At']

        for comp in comparisons:
            image1_path = comp.get('image1_path', '') or ''
            image1_name = comp.get('image1_name', '') or ''
            image2_path = comp.get('image2_path', '') or ''
            image2_name = comp.get('image2_name', '') or ''
            reconstruction_type = comp.get('reconstruction_type', '') or ''
            winner = comp.get('winner', '') or ''
            labeler_name = comp.get('labeler_name', '') or ''
            notes = comp.get('notes', '') or ''
            created_at = str(comp.get('created_at', '') or '')

            # Escape quotes for CSV
            notes_escaped = notes.replace('"', '""')

            csv_lines.append(f'"{image1_path}","{image1_name}","{image2_path}","{image2_name}",'
                           f'"{reconstruction_type}","{winner}","{labeler_name}",'
                           f'"{notes_escaped}","{created_at}"')

        return '\n'.join(csv_lines), 200, {'Content-Type': 'text/csv; charset=utf-8',
                                            'Content-Disposition': 'attachment; filename="pairwise_comparisons_export.csv"'}
    except Exception as e:
        conn.close()
        import traceback
        return f'Error exporting pairwise comparisons: {str(e)}\n{traceback.format_exc()}', 500

# Initialize database on startup (for both local and cloud)
init_db()

# Auto-restore from Google Sheets if database is empty
try:
    from google_sheets_sync import restore_from_sheets_if_empty
    restore_from_sheets_if_empty(get_db_connection, USE_POSTGRES)
except Exception as e:
    print(f"Note: Could not check/restore from Google Sheets: {e}")

if __name__ == '__main__':
    # Get port from environment (for cloud) or use default
    port = int(os.environ.get('PORT', 5001))
    # Use 0.0.0.0 to allow access from other devices on the network
    # Use 127.0.0.1 for localhost-only access
    app.run(debug=True, host='0.0.0.0', port=port)

