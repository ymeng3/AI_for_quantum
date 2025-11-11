# Google Drive Integration Setup

Your images are in a public Google Drive folder. Here's how to set it up:

## Quick Setup (Recommended)

### Step 1: Generate Image Mapping File

Run the setup script to create a mapping of all images:

```bash
cd Labeling_software

# Option A: With API Key (faster, more reliable)
export GOOGLE_DRIVE_API_KEY='your-api-key-here'
python google_drive_setup.py

# Option B: Without API Key (will create empty file, you'll need to manually add images)
python google_drive_setup.py
```

**Get a Google Drive API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable "Google Drive API"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the API key
6. (Optional) Restrict the key to Google Drive API only

### Step 2: Commit the Generated File

The script creates `google_drive_images.json` with all your image mappings:

```bash
git add google_drive_images.json
git commit -m "Add Google Drive image mapping"
git push origin main
```

### Step 3: Configure Render

In your Render dashboard, add environment variable:

- **Key**: `USE_GOOGLE_DRIVE`
- **Value**: `true`

(Optional) If you want to regenerate the mapping on the fly:
- **Key**: `GOOGLE_DRIVE_API_KEY`
- **Value**: Your API key

### Step 4: Redeploy

Render will automatically redeploy when you push to GitHub, or you can manually trigger a deploy.

---

## How It Works

1. **Image Mapping**: The `google_drive_setup.py` script scans your Google Drive folder and creates a JSON file mapping image paths to Google Drive file IDs.

2. **Serving Images**: When the app needs an image, it:
   - Looks up the file ID from the mapping
   - Fetches the image from Google Drive using a direct download link
   - Serves it to the user

3. **Caching**: Images are cached by the browser (1 hour) for faster loading.

---

## Troubleshooting

**Images not loading:**
- Make sure `google_drive_images.json` exists in the repo
- Verify `USE_GOOGLE_DRIVE=true` is set in Render
- Check that your Google Drive folder is public
- Look at Render logs for errors

**Mapping file is empty:**
- You need a Google Drive API key to generate the mapping
- Or manually create the JSON file with your image file IDs

**Slow image loading:**
- This is normal - images are fetched from Google Drive on-demand
- Consider uploading to cloud storage (S3, etc.) for better performance

---

## Alternative: Manual File ID Mapping

If you can't use the API, you can manually create `google_drive_images.json`:

```json
{
  "folder_id": "1lRe2jJhgC8Pd6RsCKEirNK4NlWmC1QJU",
  "images": {
    "STO_ideal_HTR/HTR_1.png": {
      "file_id": "YOUR_FILE_ID_HERE",
      "name": "HTR_1.png",
      "mime_type": "image/png"
    }
  },
  "total_count": 1
}
```

To get file IDs:
1. Open image in Google Drive
2. URL format: `https://drive.google.com/file/d/FILE_ID/view`
3. Copy the FILE_ID part

---

## Next Steps

Once set up, your app will:
- ✅ List all images from Google Drive
- ✅ Serve images on-demand
- ✅ Store labels in database (SQLite or PostgreSQL)
- ✅ Work for your entire team!

