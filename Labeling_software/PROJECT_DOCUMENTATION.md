# Labeling Software - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [File Structure](#file-structure)
5. [Database Schema](#database-schema)
6. [API Endpoints](#api-endpoints)
7. [Frontend Architecture](#frontend-architecture)
8. [Key Functions](#key-functions)
9. [UI Components](#ui-components)
10. [Configuration](#configuration)
11. [Workflows](#workflows)
12. [Known Issues & Fixes](#known-issues--fixes)

---

## Project Overview 

This is a web-based image labeling application built with Flask (Python backend) and vanilla JavaScript (frontend). The application allows users to label images from Google Drive with reconstruction types and quality scores. It supports two labeling modes:

1. **Absolute Scoring Mode**: Label individual images with reconstruction types and quality scores
2. **Pairwise Comparison Mode**: Compare two images side-by-side and select which one better represents each reconstruction type

The application integrates with:
- **Google Drive API**: For fetching images from trajectory folders
- **SQLite/PostgreSQL**: For storing labels and comparisons
- **Google Sheets**: For syncing data to a cloud-based spreadsheet

---

## Features

### Core Features

1. **Dual Labeling Modes**
   - Absolute Scoring: Label individual images with reconstruction types and quality scores
   - Pairwise Comparison: Compare two images and select winners for each reconstruction type

2. **Image Management**
   - Load images from Google Drive (filtered to specific trajectory folders: `2022-02-04`, `2022-02-06`, `2022-04-11`)
   - Lazy loading with Intersection Observer for performance
   - Image Bank with pagination (50 images per page)
   - Filter options: All, Labeled, Unlabeled

3. **Bad Image Marking**
   - Mark images as "Bad" in pairwise mode
   - Bad images show a red badge in the Image Bank
   - Bad images are automatically filtered out from random pair selection
   - Bad images are excluded from pairwise comparisons but still visible in the grid

4. **Data Management**
   - Save labels with labeler name and notes
   - Export data to CSV
   - Delete labels with confirmation
   - View collected labels in tabbed interface (Absolute Scoring / Pairwise Comparison)

5. **Google Sheets Integration**
   - Automatic sync when labels/comparisons are saved
   - Automatic sync when labels/comparisons are deleted
   - One-time migration script for existing data

6. **UI Features**
   - Collapsible sections (Image Labeling, Image Bank, Collected Labels)
   - Random image/pair selection buttons
   - Brightness adjustment (0-400%)
   - Mode switching between Absolute and Pairwise
   - Visual indicators for selected images
   - Status badges on images (✓ for labeled, "Bad" for bad images)

---

## Architecture

### Backend (Flask)
- **Framework**: Flask
- **Database**: SQLite (local) or PostgreSQL (Render deployment)
- **Image Source**: Google Drive API
- **External Services**: Google Sheets API (gspread)

### Frontend
- **Technology**: Vanilla JavaScript (no frameworks)
- **Styling**: CSS with Flexbox layout
- **Image Loading**: Lazy loading with Intersection Observer
- **State Management**: Global variables and DOM manipulation

### Deployment
- **Platform**: Render
- **Database**: PostgreSQL (via DATABASE_URL environment variable)
- **Environment Variables**: 
  - `DATABASE_URL`: PostgreSQL connection string
  - `GOOGLE_SHEETS_CREDENTIALS`: JSON credentials for Google Sheets
  - `GOOGLE_SHEETS_SPREADSHEET_ID`: ID of the Google Sheet

---

## File Structure

```
Labeling_software/
├── app.py                          # Main Flask application
├── google_drive_setup.py           # Google Drive API setup script
├── google_sheets_sync.py           # Google Sheets synchronization
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Main HTML template
├── static/
│   ├── app.js                     # Frontend JavaScript
│   └── style.css                  # CSS styles
├── extract_labeled_data.py        # Script to extract data to CSV
├── import_pairwise_simple.py       # Import pairwise data script
├── restore_all_pairwise_data.py    # Restore historical data
├── sync_existing_to_sheets.py      # One-time sync to Google Sheets
└── GOOGLE_SHEETS_SYNC.md          # Google Sheets setup instructions
```

---

## Database Schema

### `labels` Table
Stores absolute scoring labels.

```sql
CREATE TABLE labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    quality INTEGER,
    reconstruction TEXT,  -- JSON array or single string
    reconstruction_scores TEXT,  -- JSON object
    labeler_name TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Reconstruction Types:**
- `(1 x 1)`
- `Twinned(2 x 1)`
- `c(6 x 2)`
- `(√13 x √13)`
- `HTR`
- `Bad`

### `pairwise_comparisons` Table
Stores pairwise comparison data.

```sql
CREATE TABLE pairwise_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image1_path TEXT NOT NULL,
    image1_name TEXT NOT NULL,
    image2_path TEXT NOT NULL,
    image2_name TEXT NOT NULL,
    reconstruction_type TEXT NOT NULL,
    winner TEXT NOT NULL,  -- 'image1', 'image2', 'tie', 'not_apply'
    labeler_name TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Winner Values:**
- `'image1'`: Image 1 is better
- `'image2'`: Image 2 is better
- `'tie'`: Both images are equal
- `'not_apply'`: Neither image shows this reconstruction type

---

## API Endpoints

### Image Endpoints

**GET `/api/images`**
- Returns list of all images from Google Drive
- Filtered to specific trajectory folders
- Response: `[{path, name}, ...]`

**GET `/api/images/<path>`**
- Serves image file from Google Drive
- Path segments are URL-encoded
- Returns image with cache headers

### Label Endpoints

**GET `/api/labels`**
- Returns all labels
- Response: Array of label objects

**POST `/api/labels`**
- Creates or updates a label
- Body: `{file_path, file_name, quality, reconstruction, reconstruction_scores, labeler_name, notes}`
- Syncs to Google Sheets automatically
- Response: Created/updated label object

**DELETE `/api/labels/<path>`**
- Deletes a label by file path
- Syncs deletion to Google Sheets
- Response: `{success: true}`

**GET `/api/labels/export`**
- Exports all labels to CSV
- Returns CSV file download

### Pairwise Comparison Endpoints

**GET `/api/pairwise`**
- Returns all pairwise comparisons
- Response: Array of comparison objects

**POST `/api/pairwise`**
- Creates a pairwise comparison
- Body: `{image1_path, image1_name, image2_path, image2_name, reconstruction_type, winner, labeler_name, notes}`
- Syncs to Google Sheets automatically
- Response: Created comparison object

**DELETE `/api/pairwise/<id>`**
- Deletes a pairwise comparison by ID
- Syncs deletion to Google Sheets
- Response: `{success: true}`

**GET `/api/pairwise/export`**
- Exports all pairwise comparisons to CSV
- Returns CSV file download

---

## Frontend Architecture

### Global State Variables

```javascript
let images = [];                    // All images from API
let labels = {};                    // Labels indexed by file_path
let currentMode = 'pairwise';       // 'absolute' or 'pairwise'
let currentImage = null;            // Currently selected image (absolute mode)
let pairwiseImage1 = null;          // First image in pairwise comparison
let pairwiseImage2 = null;          // Second image in pairwise comparison
let pairwiseComparisons = {};       // Current comparison selections {reconstruction_type: winner}
let pairwiseComparisonsList = [];   // All saved comparisons from API
let currentPage = 1;                // Current page in image grid
let imagesPerPage = 50;             // Images per page
let currentLabelerName = '';        // Labeler name (absolute mode)
let pairwiseLabelerName = '';       // Labeler name (pairwise mode)
let pairwiseNotes = '';             // Notes for pairwise comparison
```

### Key Functions

#### Image Loading & Display

**`loadImages()`**
- Fetches images from `/api/images`
- Calls `renderImageGrid()` to display

**`renderImageGrid(filter = 'all')`**
- Renders image grid with pagination
- Filters images based on mode and filter option
- In pairwise mode, shows Bad images but prevents selection
- Creates image items with lazy loading
- Calls `updateImageGridStatus()` after rendering

**`updateImageGridStatus()`**
- Updates status badges on all image items
- Shows ✓ for labeled images
- Shows "Bad" badge for Bad images

#### Mode Management

**`switchMode(mode)`**
- Switches between 'absolute' and 'pairwise' modes
- Updates UI visibility
- Updates button states
- Re-renders image grid

**`initializePairwiseMode()`**
- Sets up event listeners for pairwise mode
- Handles labeler name input
- Handles comparison buttons
- Handles brightness sliders
- Handles Bad checkbox events
- Handles save/clear buttons

#### Image Selection

**`selectImage(img)`** (Absolute Mode)
- Sets `currentImage`
- Loads image in main display
- Loads existing labels if any

**`selectImageForPairwise(img, itemElement)`**
- Prevents selection of Bad images
- Handles toggle/deselect if clicking already selected image
- Logic:
  - If both empty → fill Image 1
  - If Image 1 empty, Image 2 filled → fill Image 1
  - If Image 1 filled, Image 2 empty → fill Image 2
  - If both filled → replace Image 2

**`setPairwiseImage(slot, img, itemElement)`**
- Sets image in specified slot (1 or 2)
- Updates DOM elements
- Updates visual indicators
- Enables/disables Bad checkbox

**`clearPairwiseImage(slot)`**
- Clears image from specified slot
- Resets DOM elements
- Disables Bad checkbox

#### Labeling

**`saveLabel()`** (Absolute Mode)
- Saves label to `/api/labels`
- Includes reconstruction types and scores
- Reloads labels after save

**`savePairwiseComparison()`**
- Saves all comparison selections to `/api/pairwise`
- One API call per reconstruction type
- Clears current comparison after save
- Loads new random pair automatically
- Reloads pairwise comparisons list

**`markImageAsBad(img, slot)`**
- Saves image as "Bad" label
- Updates local labels cache
- Re-renders grid to show badge
- Replaces image with random non-Bad image
- Shows notification

#### Bad Image Handling

**`isImageBad(imgPath)`**
- Checks if image is labeled as "Bad"
- Handles both array and string reconstruction formats

**`loadRandomPair()`**
- Filters out Bad images
- Selects two random different images
- Sets them as Image 1 and Image 2

#### Data Management

**`loadLabels()`**
- Fetches labels from `/api/labels`
- Updates `labels` object
- Calls `renderLabelsTable()`
- Calls `updateImageGridStatus()`

**`loadPairwiseComparisons()`**
- Fetches comparisons from `/api/pairwise`
- Updates `pairwiseComparisonsList`
- Calls `renderPairwiseTable()`

**`exportLabels()`**
- Exports absolute labels to CSV
- Triggers browser download

**`exportPairwiseComparisons()`**
- Exports pairwise comparisons to CSV
- Triggers browser download

**`deleteLabel(filePath)`**
- Deletes label from `/api/labels`
- Reloads labels
- Updates grid

**`deletePairwiseComparison(id)`**
- Deletes comparison from `/api/pairwise`
- Reloads comparisons

#### UI Utilities

**`showNotification(message, duration, targetElement)`**
- Shows notification message
- If `targetElement` provided, shows inline next to element
- Otherwise shows in top-right corner
- Auto-removes after duration

**`toggleSection(sectionName)`**
- Toggles collapse state of section
- Saves state to localStorage

**`setupLabelsTabs()`**
- Sets up tabs for Absolute/Pairwise labels
- Switches between label tables

---

## UI Components

### Mode Selector
- Two buttons: "Pairwise Comparison" (left) and "Absolute Scoring" (right)
- Active button highlighted
- Default: Pairwise Comparison

### Image Labeling Section
- **Absolute Mode:**
  - Image display area
  - Reconstruction type buttons
  - Quality slider
  - Labeler name input
  - Notes input
  - Save button
  - Random Image button (in header)

- **Pairwise Mode:**
  - Two image containers side-by-side
  - "Image 1" and "Image 2" captions
  - Brightness sliders for each image
  - Reconstruction type buttons (Image 1 / Tie / Image 2 / Not Apply)
  - "Mark this image as 'Bad'" checkboxes
  - Labeler name input
  - Notes input
  - Save, Clear, Random Pair buttons

### Image Bank Section
- Filter dropdown: All, Labeled, Unlabeled
- Image grid with pagination
- Status badges:
  - Green ✓ for labeled images
  - Red "Bad" for Bad images
- Click behavior:
  - Absolute mode: Selects image for labeling
  - Pairwise mode: Selects image for comparison

### Collected Labels Section
- Two tabs: "Pairwise Comparison" and "Absolute Scoring"
- Tables showing:
  - File name
  - Reconstruction type(s)
  - Scores (absolute mode)
  - Labeler name
  - Notes
  - Delete button
- Google Sheets link next to header

---

## Configuration

### Trajectory Folders
Only images from these folders are loaded:
- `2022-02-04`
- `2022-02-06`
- `2022-04-11`
- `2025-10-04` (contains subfolders A and B)
- `2025-10-05`

Defined in `app.py`:
```python
ALLOWED_TRAJECTORY_FOLDERS = ['2022-02-04', '2022-02-06', '2022-04-11', '2025-10-04', '2025-10-05']
```

**Note:** The 2025-10-04 folder has subfolders (A and B) which are automatically traversed.

### Reconstruction Types
**Absolute Scoring:**
- `(1 x 1)`
- `Twinned(2 x 1)`
- `c(6 x 2)`
- `(√13 x √13)`
- `HTR`
- `Bad`

**Pairwise Comparison:**
- Same types, but "Bad" is not a comparison option (it's a checkbox for marking images)

### Image Loading
- **Images per page**: 50
- **Lazy loading**: Intersection Observer with 200px rootMargin
- **First 9 images**: Load immediately for better UX

### Brightness Range
- **Min**: 0%
- **Max**: 400%

---

## Workflows

### Absolute Scoring Workflow
1. Switch to "Absolute Scoring" mode
2. Click "Random Image" or select from Image Bank
3. Select reconstruction type(s)
4. Adjust quality slider if needed
5. Enter labeler name
6. Add notes (optional)
7. Click "Save"
8. Label appears in "Collected Labels" → "Absolute Scoring" tab

### Pairwise Comparison Workflow
1. Ensure in "Pairwise Comparison" mode (default)
2. Click "Random Pair" or manually select two images from Image Bank
3. Adjust brightness if needed
4. For each reconstruction type, select:
   - Image 1 (if Image 1 is better)
   - Tie (if equal)
   - Image 2 (if Image 2 is better)
   - Not Apply (if neither shows this type)
5. Optionally mark images as "Bad" (replaces with random image)
6. Enter labeler name
7. Add notes (optional)
8. Click "Save"
9. New random pair loads automatically
10. Comparison appears in "Collected Labels" → "Pairwise Comparison" tab

### Marking Image as Bad
1. Select two images in pairwise mode
2. Check "Mark this image as 'Bad'" checkbox below an image
3. Image is saved as "Bad" label
4. Image is automatically replaced with random non-Bad image
5. Bad image shows red "Bad" badge in Image Bank
6. Bad images are filtered out from random pair selection

---

## Known Issues & Fixes

### Image Loading
**Issue**: Images stuck on loading spinner
**Fix**: Implemented Intersection Observer with proper rootMargin and threshold. Force load first 9 images immediately.

### Bad Badge Not Appearing
**Issue**: Bad badge not showing after marking image as Bad
**Fix**: 
- Re-render grid after marking as Bad
- Call `updateImageGridStatus()` after every grid render
- Use `className` instead of `classList.add()` for proper class setting

### Selection Logic
**Issue**: Can only select Image 2
**Fix**: Explicitly check for "both empty" case first, then handle individual empty slots

### Database Sync
**Issue**: Data lost between SQLite and PostgreSQL
**Fix**: Implemented Google Sheets as single source of truth. All saves/deletes sync to Sheets automatically.

### Google Sheets Alignment
**Issue**: Rows not aligned in Google Sheets
**Fix**: Ensure all rows have exactly 9 columns, padded with empty strings if needed

### Browser Caching
**Issue**: Frontend changes not visible
**Fix**: Cache-busting version parameter in HTML: `app.js?v=19`

### Event Listeners
**Issue**: Duplicate event listeners causing freezes
**Fix**: Store handler reference and remove before adding new one

### Notification Not Appearing
**Issue**: Notification not showing next to checkbox
**Fix**: Insert notification after text span in label, use inline notification styling

---

## Important Implementation Details

### Path Encoding
Image paths from Google Drive contain special characters. Always encode path segments:
```javascript
const encodedPath = img.path.split('/').map(segment => encodeURIComponent(segment)).join('/');
```

### Reconstruction Format
Reconstruction can be:
- JSON array: `["HTR", "Bad"]`
- JSON string: `"HTR"`
- Plain string: `"HTR"`

Always handle all formats when checking for "Bad":
```javascript
let reconstruction = label.reconstruction;
if (typeof reconstruction === 'string') {
    try {
        reconstruction = JSON.parse(reconstruction);
    } catch (e) {
        reconstruction = [reconstruction];
    }
}
if (Array.isArray(reconstruction)) {
    return reconstruction.includes('Bad');
}
return reconstruction === 'Bad';
```

### Database Connection
The app uses `get_db_connection()` which:
- Checks for `DATABASE_URL` environment variable
- If present, connects to PostgreSQL
- Otherwise, uses SQLite (`labels.db`)

### Google Sheets Sync
- Automatic on save/delete
- Uses service account credentials
- Spreadsheet ID and credentials from environment variables
- Two tabs: "Absolute Labels" and "Pairwise Comparisons"

### Image Filtering in Pairwise Mode
- Bad images are **shown** in the grid (so badge is visible)
- Bad images are **prevented** from selection (alert shown)
- **All previously labeled images** (including Bad) are **filtered out** from random pair selection
- This ensures that random pairs only come from unlabeled images

---

## Adding New Features

When adding new features, consider:

1. **Database Changes**: Update schema in `init_db()`, handle migrations
2. **API Endpoints**: Add routes in `app.py`, handle errors gracefully
3. **Frontend State**: Add global variables if needed
4. **UI Components**: Update `index.html` and `style.css`
5. **Event Listeners**: Add in appropriate initialization function
6. **Google Sheets Sync**: Update `google_sheets_sync.py` if data structure changes
7. **Cache Busting**: Increment version in `index.html` for JS/CSS changes
8. **Error Handling**: Always provide user-friendly error messages
9. **Data Validation**: Validate inputs on both frontend and backend
10. **Testing**: Test in both absolute and pairwise modes

---

## Environment Variables

### Required for Google Sheets Sync
- `GOOGLE_SHEETS_CREDENTIALS`: JSON string of service account credentials
- `GOOGLE_SHEETS_SPREADSHEET_ID`: ID of the Google Sheet

### Required for Render Deployment
- `DATABASE_URL`: PostgreSQL connection string

---

## Dependencies

### Python (requirements.txt)
- Flask
- gspread
- google-auth
- psycopg2-binary (for PostgreSQL)
- google-api-python-client
- google-auth-httplib2
- google-auth-oauthlib

### Frontend
- No external dependencies (vanilla JavaScript)
- Uses Intersection Observer API (modern browsers)

---

## Deployment Notes

1. **Render Setup**:
   - Set `DATABASE_URL` environment variable
   - Set Google Sheets credentials if using sync
   - Deploy from GitHub repository

2. **Database Migration**:
   - Tables are created automatically on first run
   - Use import scripts for historical data

3. **Google Drive Setup**:
   - Requires `credentials.json` and `token.json`
   - Run `google_drive_setup.py` to generate mapping

4. **Google Sheets Setup**:
   - Create service account
   - Share spreadsheet with service account email
   - Set environment variables
   - Run `sync_existing_to_sheets.py` for one-time migration

---

## Future Enhancements

Potential features to add:
- Batch labeling
- Label editing (currently requires delete + re-label)
- Image search/filter by name
- Statistics dashboard
- Labeler performance metrics
- Export with date ranges
- Image annotation tools
- Keyboard shortcuts
- Undo/redo functionality

---

*Last Updated: Based on conversation history up to Bad badge and selection logic fixes*



