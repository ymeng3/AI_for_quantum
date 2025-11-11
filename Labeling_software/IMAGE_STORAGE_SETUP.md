# Image Storage Setup Guide

Your images are currently in Google Drive. Here are the best options for making them accessible in your cloud deployment:

## Option 1: Download and Include in Git (Simplest for Small Datasets)

**Best for**: < 1000 images

1. **Download images from Google Drive**:
   - Go to your [Google Drive folder](https://drive.google.com/drive/u/0/folders/1lRe2jJhgC8Pd6RsCKEirNK4NlWmC1QJU)
   - Download all images
   - Extract to a folder

2. **Add to your repo**:
   ```bash
   cd "/Users/justinmeng/Desktop/Project Quantum"
   # Copy images to data folder (if not already there)
   # Then commit
   git add data/
   git commit -m "Add RHEED images"
   git push origin main
   ```

3. **Update Render**:
   - The images will be included in deployment
   - App will serve them directly

**Pros**: Simple, no extra setup  
**Cons**: Makes repo large, slow Git operations

---

## Option 2: Google Drive Public Links (Easiest)

**Best for**: Quick setup, small to medium datasets

1. **Make Google Drive folder public**:
   - Right-click folder → Share → "Anyone with the link"
   - Copy the folder ID: `1lRe2jJhgC8Pd6RsCKEirNK4NlWmC1QJU`

2. **Use direct Google Drive links**:
   - Each image can be accessed via: `https://drive.google.com/uc?export=view&id=FILE_ID`
   - Update the app to use these links

**Pros**: No upload needed, free  
**Cons**: Requires making folder public, slower loading

---

## Option 3: Upload to Cloud Storage (Best for Production)

**Best for**: Large datasets, production use

### Option 3A: AWS S3 (Free Tier Available)
1. Create AWS account (free tier: 5GB storage, 12 months)
2. Create S3 bucket
3. Upload images
4. Make bucket public or use signed URLs
5. Update app to fetch from S3

### Option 3B: Google Cloud Storage
1. Create GCP account ($300 free credit)
2. Create storage bucket
3. Upload images
4. Update app to fetch from GCS

### Option 3C: Cloudinary (Image CDN)
1. Sign up at [cloudinary.com](https://cloudinary.com) (free tier)
2. Upload images
3. Get URLs
4. Update app to use Cloudinary URLs

**Pros**: Fast, scalable, professional  
**Cons**: Requires setup, may have costs

---

## Option 4: Keep Images Local, Deploy Separately

**Best for**: Testing, small teams

1. Keep images on a shared network drive
2. Team members access locally
3. Only deploy the app code to cloud
4. App serves images from local `data/` folder

**Pros**: No cloud storage costs  
**Cons**: Only works if team has access to shared drive

---

## My Recommendation for Your Case

**For 3000+ images, I recommend:**

1. **Short term**: Use **Google Drive public links** (Option 2)
   - Quick to set up
   - No upload needed
   - Free

2. **Long term**: Upload to **AWS S3** or **Google Cloud Storage** (Option 3)
   - Better performance
   - More reliable
   - Professional solution

---

## Quick Setup: Google Drive Public Links

If you want to use Google Drive directly, I can update the code to:
1. Fetch image list from your Google Drive folder
2. Serve images via Google Drive direct links
3. No need to download/upload images

Would you like me to implement this? It requires:
- Making your Google Drive folder public (or using service account)
- Updating the code to use Google Drive API

Let me know which option you prefer!

