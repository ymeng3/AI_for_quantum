from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import mimetypes

app = Flask(__name__)
CORS(app)

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

if USE_GOOGLE_DRIVE and GOOGLE_DRIVE_IMAGES_FILE.exists():
    try:
        import json
        with open(GOOGLE_DRIVE_IMAGES_FILE, 'r') as f:
            data = json.load(f)
            GOOGLE_DRIVE_IMAGE_MAP = data.get('images', {})
            print(f"Loaded {len(GOOGLE_DRIVE_IMAGE_MAP)} images from Google Drive mapping")
    except Exception as e:
        print(f"Error loading Google Drive image map: {e}")

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
    
    conn.commit()
    conn.close()

# Get all image files from data directory or Google Drive
def get_image_files():
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_files = []
    
    if USE_GOOGLE_DRIVE and DATA_DIR is None:
        # Fetch from Google Drive
        return get_google_drive_images()
    elif DATA_DIR and DATA_DIR.exists():
        # Use local directory
        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    rel_path = os.path.relpath(os.path.join(root, file), DATA_DIR)
                    # Normalize path separators to forward slashes
                    rel_path = rel_path.replace('\\', '/')
                    image_files.append({
                        'path': rel_path,
                        'name': file,
                        'full_path': os.path.join(root, file)
                    })
        return sorted(image_files, key=lambda x: x['name'])
    else:
        # No data source available
        return []

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
            results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
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

@app.route('/api/images')
def get_images():
    """Get list of all available images"""
    images = get_image_files()
    return jsonify(images)

@app.route('/api/images/<path:image_path>')
def serve_image(image_path):
    """Serve image files from local storage or Google Drive"""
    # Flask automatically URL-decodes the path, but we need to handle it properly
    # Normalize path separators
    image_path = image_path.replace('\\', '/')
    try:
        # Decode URL encoding if needed
        import urllib.parse
        image_path = urllib.parse.unquote(image_path)
        
        if USE_GOOGLE_DRIVE and DATA_DIR is None:
            # Serve from Google Drive
            return serve_google_drive_image(image_path)
        elif DATA_DIR and DATA_DIR.exists():
            # Serve from local directory
            full_path = os.path.join(DATA_DIR, image_path)
            if not os.path.exists(full_path):
                return f"Image not found: {image_path}", 404
            
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
    data = request.json
    file_path = data.get('file_path')
    file_name = data.get('file_name')
    quality = data.get('quality')
    reconstruction = data.get('reconstruction')  # Can be a list or single value
    reconstruction_scores = data.get('reconstruction_scores')  # Dict mapping reconstruction to score
    labeler_name = data.get('labeler_name', '').strip()
    notes = data.get('notes', '').strip()
    
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
    else:
        scores_json = None
    
    conn = get_db_connection()
    c = conn.cursor()
    
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
    conn.close()
    
    return jsonify({'success': True})

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
    conn.close()
    
    if deleted > 0:
        return jsonify({'success': True, 'message': 'Label deleted successfully'})
    else:
        return jsonify({'success': False, 'message': 'Label not found'}), 404

@app.route('/api/labels/export', methods=['GET'])
def export_labels():
    """Export all labels as CSV"""
    conn = get_db_connection()
    
    if USE_POSTGRES:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT file_name, quality, reconstruction, reconstruction_scores, labeler_name, notes FROM labels ORDER BY file_name')
        labels = [dict(row) for row in c.fetchall()]
    else:
        c = conn.cursor()
        c.execute('SELECT file_name, quality, reconstruction, reconstruction_scores, labeler_name, notes FROM labels ORDER BY file_name')
        labels = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Generate CSV
    csv_lines = ['File,Quality,Reconstruction,Reconstruction_Scores,Labeler_Name,Notes']
    for label in labels:
        file_name = label['file_name']
        quality = label.get('quality') or '-'
        
        # Parse reconstruction (can be JSON list or single value)
        reconstruction = label.get('reconstruction') or '-'
        if reconstruction and reconstruction != '-':
            try:
                recon_list = json.loads(reconstruction) if isinstance(reconstruction, str) else reconstruction
                if isinstance(recon_list, list):
                    reconstruction = '; '.join(recon_list)
            except:
                pass
        
        # Parse reconstruction scores
        scores = label.get('reconstruction_scores') or '-'
        if scores and scores != '-':
            try:
                scores_dict = json.loads(scores) if isinstance(scores, str) else scores
                if isinstance(scores_dict, dict):
                    scores = '; '.join([f"{k}: {v}" for k, v in scores_dict.items()])
            except:
                pass
        
        labeler_name = label.get('labeler_name') or '-'
        notes = label.get('notes') or '-'
        # Escape quotes in notes for CSV
        notes_escaped = notes.replace('"', '""') if notes != '-' else '-'
        
        csv_lines.append(f'{file_name},{quality},"{reconstruction}","{scores}",{labeler_name},"{notes_escaped}"')
    
    return '\n'.join(csv_lines), 200, {'Content-Type': 'text/csv'}

@app.route('/api/pairwise', methods=['POST'])
def save_pairwise_comparison():
    """Save a pairwise comparison"""
    data = request.json
    image1_path = data.get('image1_path')
    image1_name = data.get('image1_name')
    image2_path = data.get('image2_path')
    image2_name = data.get('image2_name')
    reconstruction_type = data.get('reconstruction_type')
    winner = data.get('winner')  # '1', '2', or 'tie'
    labeler_name = data.get('labeler_name', '').strip()
    notes = data.get('notes', '').strip()
    
    if not all([image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    conn = get_db_connection()
    c = conn.cursor()
    
    if USE_POSTGRES:
        c.execute('''
            INSERT INTO pairwise_comparisons (image1_path, image1_name, image2_path, image2_name, 
                                             reconstruction_type, winner, labeler_name, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ''', (image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner, labeler_name, notes))
    else:
        c.execute('''
            INSERT INTO pairwise_comparisons (image1_path, image1_name, image2_path, image2_name, 
                                             reconstruction_type, winner, labeler_name, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner, labeler_name, notes))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/pairwise/export', methods=['GET'])
def export_pairwise_comparisons():
    """Export all pairwise comparisons as CSV"""
    conn = get_db_connection()
    
    if USE_POSTGRES:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
        comparisons = [dict(row) for row in c.fetchall()]
    else:
        c = conn.cursor()
        c.execute('SELECT * FROM pairwise_comparisons ORDER BY created_at DESC')
        comparisons = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Generate CSV
    csv_lines = ['Image1_Path,Image1_Name,Image2_Path,Image2_Name,Reconstruction_Type,Winner,Labeler_Name,Notes,Created_At']
    for comp in comparisons:
        csv_lines.append(f'{comp["image1_path"]},{comp["image1_name"]},{comp["image2_path"]},{comp["image2_name"]},'
                         f'{comp["reconstruction_type"]},{comp["winner"]},{comp.get("labeler_name", "-")},'
                         f'"{comp.get("notes", "-").replace('"', '""')}",{comp.get("created_at", "-")}')
    
    return '\n'.join(csv_lines), 200, {'Content-Type': 'text/csv'}

# Initialize database on startup (for both local and cloud)
init_db()

if __name__ == '__main__':
    # Get port from environment (for cloud) or use default
    port = int(os.environ.get('PORT', 5001))
    # Use 0.0.0.0 to allow access from other devices on the network
    # Use 127.0.0.1 for localhost-only access
    app.run(debug=True, host='0.0.0.0', port=port)

