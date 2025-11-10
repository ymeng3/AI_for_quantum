# RHEED Image Classification Labeling Software

A web-based tool for classifying RHEED (Reflection High-Energy Electron Diffraction) images by quality and reconstruction type.

## Features

- **Three-Panel Interface:**
  - Left: Single image labeling with quality and reconstruction options
  - Middle: Grid view of all images with filtering options
  - Right: Database table showing all collected labels

- **Labeling Options:**
  - Quality: Ideal, Good/Usable, Bad/Unusable, Unsure/Skip
  - Reconstruction: HTR, √13 x √13 R33.7°, (2 x 2), Other

- **Data Management:**
  - SQLite database for storing labels
  - Export labels as CSV
  - Filter images by labeled/unlabeled status
  - Visual indicators for labeling status

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5001
```

## Network Access (Sharing with Team)

The server is configured to accept connections from other devices on your network.

**To share with others:**

1. **Find your computer's IP address:**
   - **Mac/Linux:** Run `ifconfig` or `ip addr` in terminal, look for your local IP (usually starts with 192.168.x.x or 10.x.x.x)
   - **Windows:** Run `ipconfig` in Command Prompt, look for "IPv4 Address"

2. **Share the URL with your team:**
   ```
   http://YOUR_IP_ADDRESS:5001
   ```
   For example: `http://192.168.1.100:5001`

3. **Firewall:** Make sure your firewall allows incoming connections on port 5001

4. **Keep the server running:** The server must be running on your machine for others to access it

**Important Notes:**
- All team members will share the same database (`labels.db`)
- SQLite handles concurrent access, but for heavy use, consider a cloud database
- The server must stay running on the host machine
- For production use, consider deploying to a cloud service (Heroku, AWS, etc.)

## Usage

1. **Select an image** from the middle grid by clicking on it
2. **Label the image** using the quality and reconstruction buttons
3. **Save the label** using the "Save Label" button
4. **View all labels** in the right panel table
5. **Export data** using the "Export CSV" button

## Data Storage

Labels are stored in a SQLite database (`labels.db`) in the same directory as the application. The database schema includes:
- File path and name
- Quality label
- Reconstruction label
- Timestamps for creation and updates

## Database Management

The labels are stored in a SQLite database file (`labels.db`) in the same directory as the application.

**Backup:**
- Use the "Export CSV" button in the UI to export all labels as CSV
- Run `python backup_db.py` to create a timestamped backup of the database file
- Backups are stored in the `backups/` directory (last 10 backups are kept)

**Persistence:**
- The database file persists on your disk even after stopping the server
- All labels are automatically loaded when you restart the server
- The database file is a regular file - you can copy, move, or delete it like any other file

## Cloud Deployment

The application is now ready for cloud deployment! See `DEPLOYMENT.md` for detailed instructions.

**Quick Summary:**
- The app automatically uses PostgreSQL when `DATABASE_URL` environment variable is set (cloud)
- Uses SQLite locally when `DATABASE_URL` is not set
- Ready to deploy to Render, Railway, Heroku, or other cloud platforms
- All team members can access from anywhere once deployed

**See `DEPLOYMENT.md` for step-by-step cloud deployment instructions.**

