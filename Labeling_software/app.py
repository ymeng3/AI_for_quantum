from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import os
from pathlib import Path
from datetime import datetime
import json
import mimetypes

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    conn.commit()
    conn.close()

# Get all image files from data directory
def get_image_files():
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    image_files = []
    
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
    """Serve image files"""
    # Flask automatically URL-decodes the path, but we need to handle it properly
    # Normalize path separators
    image_path = image_path.replace('\\', '/')
    try:
        # Decode URL encoding if needed
        import urllib.parse
        image_path = urllib.parse.unquote(image_path)
        
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
        # Add cache control headers
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        import traceback
        return f"Error serving image: {str(e)}\n{traceback.format_exc()}", 404

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
    reconstruction = data.get('reconstruction')
    
    if not file_path or not file_name:
        return jsonify({'error': 'file_path and file_name are required'}), 400
    
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
                SET quality = %s, reconstruction = %s, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = %s
            ''', (quality, reconstruction, file_path))
        else:
            c.execute('''
                UPDATE labels 
                SET quality = ?, reconstruction = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
            ''', (quality, reconstruction, file_path))
    else:
        # Insert new label
        if USE_POSTGRES:
            c.execute('''
                INSERT INTO labels (file_path, file_name, quality, reconstruction)
                VALUES (%s, %s, %s, %s)
            ''', (file_path, file_name, quality, reconstruction))
        else:
            c.execute('''
                INSERT INTO labels (file_path, file_name, quality, reconstruction)
                VALUES (?, ?, ?, ?)
            ''', (file_path, file_name, quality, reconstruction))
    
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
            conn.close()
            return jsonify(dict(row))
    else:
        c = conn.cursor()
        c.execute('SELECT * FROM labels WHERE file_path = ?', (file_path,))
        row = c.fetchone()
        if row:
            conn.close()
            return jsonify(dict(row))
    
    conn.close()
    return jsonify({'quality': None, 'reconstruction': None})

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
        c.execute('SELECT file_name, quality, reconstruction FROM labels ORDER BY file_name')
        labels = [dict(row) for row in c.fetchall()]
    else:
        c = conn.cursor()
        c.execute('SELECT file_name, quality, reconstruction FROM labels ORDER BY file_name')
        labels = [dict(row) for row in c.fetchall()]
    
    conn.close()
    
    # Generate CSV
    csv_lines = ['File,Quality,Label']
    for label in labels:
        file_name = label['file_name']
        quality = label['quality'] or '-'
        reconstruction = label['reconstruction'] or '-'
        csv_lines.append(f'{file_name},{quality},{reconstruction}')
    
    return '\n'.join(csv_lines), 200, {'Content-Type': 'text/csv'}

if __name__ == '__main__':
    init_db()
    # Get port from environment (for cloud) or use default
    port = int(os.environ.get('PORT', 5001))
    # Use 0.0.0.0 to allow access from other devices on the network
    # Use 127.0.0.1 for localhost-only access
    app.run(debug=True, host='0.0.0.0', port=port)

