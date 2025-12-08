# Google Sheets as Single Source of Truth

## Problem
- **Root Cause**: Two separate databases (local SQLite vs Render PostgreSQL) caused data loss and sync issues
- **Solution**: Use Google Sheets as the single source of truth, sync to both databases

## Setup

### 1. Create Google Sheets Spreadsheet
1. Create a new Google Sheets spreadsheet
2. Share it with a service account email (see step 2)
3. Copy the Spreadsheet ID from the URL: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`

### 2. Create Google Service Account
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Google Sheets API" and "Google Drive API"
4. Create a Service Account
5. Download the JSON credentials file
6. Share your Google Sheet with the service account email (found in the JSON file)

### 3. Set Environment Variables

**For Render:**
- `GOOGLE_SHEETS_SPREADSHEET_ID`: Your spreadsheet ID
- `GOOGLE_SHEETS_CREDENTIALS_JSON`: The entire JSON credentials as a string (or use a file path)

**For Local:**
- Same environment variables, or use a credentials file path

### 4. Install Dependencies
```bash
pip install gspread google-auth
```

## How It Works

1. **When you save data**: 
   - Saves to Google Sheets first (source of truth)
   - Then saves to current database (SQLite or PostgreSQL)
   - Both stay in sync

2. **When you load data**:
   - Can load from Google Sheets or database
   - Sync script can pull from Sheets to update databases

3. **Backup/Restore**:
   - Google Sheets is always the backup
   - Can restore to any database from Sheets

## Benefits

✅ Single source of truth (Google Sheets)
✅ Both databases stay in sync
✅ Easy backup (just export from Google Sheets)
✅ Can access data from anywhere
✅ No more data loss between local and cloud

