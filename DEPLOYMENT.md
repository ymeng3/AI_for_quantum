# Cloud Deployment Guide

This guide will help you deploy the RHEED labeling tool to the cloud so your team can access it from anywhere.

## Option 1: Render (Recommended - Free Tier Available)

### Step 1: Prepare Your Code
1. Make sure all files are committed to a Git repository (GitHub, GitLab, or Bitbucket)

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `rheed-labeling` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or paid for better performance)

### Step 3: Add PostgreSQL Database
1. In Render dashboard, click "New +" → "PostgreSQL"
2. Name it (e.g., `rheed-db`)
3. Copy the **Internal Database URL**
4. Go back to your Web Service → Environment
5. Add environment variable:
   - **Key**: `DATABASE_URL`
   - **Value**: Paste the database URL you copied

### Step 4: Upload Images
Since cloud services don't have your local `data/` folder, you have two options:

**Option A: Use Cloud Storage (Recommended)**
- Upload images to AWS S3, Google Cloud Storage, or similar
- Update the code to fetch images from cloud storage

**Option B: Include Images in Git (Not recommended for large datasets)**
- Add images to the repository
- This works for small datasets but not ideal for 3000+ images

### Step 5: Access Your App
- Render will give you a URL like: `https://rheed-labeling.onrender.com`
- Share this URL with your team!

---

## Option 2: Railway (Alternative)

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your repository
4. Add PostgreSQL database:
   - Click "New" → "Database" → "Add PostgreSQL"
   - Railway automatically sets `DATABASE_URL` environment variable
5. Deploy!

---

## Option 3: Heroku (Legacy, but still works)

1. Install Heroku CLI: `brew install heroku/brew/heroku`
2. Login: `heroku login`
3. Create app: `heroku create rheed-labeling`
4. Add PostgreSQL: `heroku addons:create heroku-postgresql:mini`
5. Deploy: `git push heroku main`
6. Your app will be at: `https://rheed-labeling.herokuapp.com`

---

## Important Notes

### Image Storage
For cloud deployment, you'll need to handle images differently:

1. **Upload to cloud storage** (AWS S3, Google Cloud Storage, etc.)
2. **Update `get_image_files()` function** to list images from cloud storage
3. **Update `serve_image()` function** to fetch images from cloud storage

### Environment Variables
- `DATABASE_URL`: Automatically set by cloud providers when you add PostgreSQL
- The app automatically detects this and uses PostgreSQL instead of SQLite

### Free Tier Limitations
- Render free tier: Service spins down after 15 minutes of inactivity
- Railway free tier: Limited hours per month
- Consider paid plans for production use

---

## Quick Start Script

For Render, you can use the included `render.yaml` file:
1. Push your code to GitHub
2. In Render, select "New +" → "Blueprint"
3. Connect your repository
4. Render will automatically detect `render.yaml` and configure everything

---

## Troubleshooting

**Database connection errors:**
- Make sure `DATABASE_URL` environment variable is set
- Check that PostgreSQL database is running
- Verify SSL mode is enabled (required for cloud databases)

**Images not loading:**
- Images need to be in cloud storage or included in deployment
- Check file paths are correct
- Verify cloud storage permissions

**App crashes:**
- Check logs in your cloud provider's dashboard
- Verify all dependencies are in `requirements.txt`
- Ensure `gunicorn` is installed

