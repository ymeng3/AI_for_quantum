# Render Deployment - Step by Step Guide

Follow these steps to deploy your RHEED labeling tool to Render.

## Prerequisites

1. **GitHub Account** (free) - [github.com](https://github.com)
2. **Render Account** (free) - [render.com](https://render.com)

## Step 1: Prepare Your Code for Git

1. **Initialize Git** (if not already done):
   ```bash
   cd "/Users/justinmeng/Desktop/Project Quantum/Labeling_software"
   git init
   ```

2. **Create a `.gitignore` file** (already exists, but verify it includes):
   - `*.db` (database files)
   - `__pycache__/`
   - `venv/` or `env/`
   - `.DS_Store`

3. **Commit your code**:
   ```bash
   git add .
   git commit -m "Initial commit - RHEED labeling tool"
   ```

## Step 2: Push to GitHub

1. **Create a new repository on GitHub**:
   - Go to [github.com/new](https://github.com/new)
   - Name it: `rheed-labeling` (or any name you prefer)
   - Make it **Private** (recommended) or Public
   - **Don't** initialize with README (you already have code)

2. **Push your code**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rheed-labeling.git
   git branch -M main
   git push -u origin main
   ```
   (Replace `YOUR_USERNAME` with your GitHub username)

## Step 3: Create PostgreSQL Database on Render

1. **Go to Render Dashboard**: [dashboard.render.com](https://dashboard.render.com)

2. **Create PostgreSQL Database**:
   - Click **"New +"** → **"PostgreSQL"**
   - **Name**: `rheed-db` (or any name)
   - **Database**: `rheed_labels` (or leave default)
   - **User**: `rheed_user` (or leave default)
   - **Region**: Choose closest to you
   - **Plan**: **Free** (or paid if you prefer)
   - Click **"Create Database"**

3. **Copy the Database URL**:
   - Once created, click on your database
   - Find **"Internal Database URL"** or **"Connection String"**
   - Copy it (looks like: `postgresql://user:pass@host:port/dbname`)
   - **Save this somewhere** - you'll need it in Step 4

## Step 4: Deploy Web Service

1. **Create Web Service**:
   - In Render dashboard, click **"New +"** → **"Web Service"**
   - Click **"Connect GitHub"** (or GitLab/Bitbucket)
   - Authorize Render to access your repositories
   - Select your `rheed-labeling` repository

2. **Configure the Service**:
   - **Name**: `rheed-labeling` (or any name)
   - **Region**: Same as your database
   - **Branch**: `main` (or `master`)
   - **Root Directory**: Leave empty (or `Labeling_software` if you put it in a subfolder)
   - **Environment**: **Python 3**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: **Free** (or paid for better performance)

3. **Add Environment Variables**:
   - Scroll down to **"Environment Variables"**
   - Click **"Add Environment Variable"**
   - **Key**: `DATABASE_URL`
   - **Value**: Paste the database URL you copied in Step 3
   - Click **"Add"**

4. **Deploy**:
   - Scroll down and click **"Create Web Service"**
   - Render will start building and deploying your app
   - This takes 5-10 minutes the first time

## Step 5: Handle Images

**Important**: Your images are in a local `data/` folder. For cloud deployment, you have options:

### Option A: Upload to Cloud Storage (Recommended for 3000+ images)
- Upload images to AWS S3, Google Cloud Storage, or similar
- Update the code to fetch from cloud storage
- (I can help with this if needed)

### Option B: Include in Git (For smaller datasets)
- Add images to your Git repository
- Push to GitHub
- Images will be included in deployment
- **Warning**: This makes your repo very large (not ideal for 3000+ images)

### Option C: Use a Shared Network Drive
- Keep images on a shared drive
- Update code to point to that location
- (Only works if all team members have access)

**For now**: The app will deploy but images won't load until you set up image storage.

## Step 6: Access Your App

1. **Wait for deployment** (5-10 minutes)
2. **Get your URL**: Render will give you a URL like:
   - `https://rheed-labeling.onrender.com`
3. **Share with your team**: Everyone can access it from anywhere!

## Troubleshooting

**Build fails?**
- Check the build logs in Render dashboard
- Make sure `requirements.txt` has all dependencies
- Verify Python version matches (3.11.0)

**Database connection error?**
- Verify `DATABASE_URL` environment variable is set correctly
- Check that PostgreSQL database is running
- Make sure you copied the **Internal Database URL**

**App crashes?**
- Check logs in Render dashboard
- Verify `gunicorn` is in `requirements.txt`
- Check that all imports work

**Images not loading?**
- Images need to be in cloud storage or included in deployment
- See Step 5 above

## Next Steps After Deployment

1. **Test the app**: Open your Render URL and test labeling
2. **Share with team**: Send the URL to your team members
3. **Set up image storage**: Choose one of the options in Step 5
4. **Monitor usage**: Check Render dashboard for usage stats

## Cost - Free Options

### Render Free Tier (What we're using)
- **Web Service**: ✅ **FREE** (spins down after 15 min inactivity - first request may be slow)
- **PostgreSQL**: ✅ **FREE for 90 days**, then $7/month
- **Total**: $0 for first 90 days, then $7/month for database only

### Completely Free Alternatives

**Option 1: Railway (Recommended for 100% Free)**
- ✅ $5/month free credit (usually enough for small apps)
- ✅ PostgreSQL included
- ✅ Always-on service
- Sign up: [railway.app](https://railway.app)

**Option 2: Fly.io**
- ✅ Free tier with resource limits
- ✅ PostgreSQL available
- ✅ Good for small apps
- Sign up: [fly.io](https://fly.io)

**Option 3: PythonAnywhere**
- ✅ Free tier available
- ✅ SQLite included (no PostgreSQL needed)
- ✅ Simple deployment
- Sign up: [pythonanywhere.com](https://www.pythonanywhere.com)

**Option 4: Use SQLite Instead of PostgreSQL**
- ✅ 100% FREE forever
- ✅ No database costs
- ✅ Works for small teams (SQLite handles concurrent access)
- ⚠️ Less ideal for many simultaneous users

---

**Need help?** Check the logs in Render dashboard or let me know what error you see!

