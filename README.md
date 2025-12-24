# GitHub Sync Service v2.0

A FastAPI microservice that syncs user's GitHub projects using OAuth and webhooks, and generates AI-powered resume-ready summaries using Google Gemini.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GitHub App + Webhooks Flow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User clicks "Connect GitHub" button in frontend                 │
│  2. User is redirected to GitHub OAuth authorization                │
│  3. After authorization, callback stores access token               │
│  4. User can manually sync repos or receive webhook events          │
│  5. AI descriptions generated on-demand via Gemini API              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

- 🔗 **GitHub OAuth**: Secure authorization flow for repository access
- 🔔 **Webhooks**: Real-time updates when repositories change
- 🤖 **AI Summarization**: Professional project descriptions via Gemini
- 🎥 **Video Upload**: Support for project intro videos
- 🗄️ **Supabase Storage**: Secure video and data storage

## 📋 Prerequisites

### 1. Create a GitHub OAuth App

1. Go to: https://github.com/settings/developers
2. Click **"New OAuth App"** (or create a GitHub App for webhooks)
3. Fill in the following:

| Field | Value |
|-------|-------|
| **Application name** | CareerAutomate Projects |
| **Homepage URL** | http://localhost:3000 |
| **Authorization callback URL** | http://localhost:8005/v1/github/callback |

4. After creation, note down:
   - **Client ID**
   - **Client Secret** (generate one)

### 2. (Optional) Create a GitHub App for Webhooks

For production webhook support:

1. Go to: https://github.com/settings/apps/new
2. Fill in:

| Field | Value |
|-------|-------|
| **GitHub App name** | CareerAutomate Sync |
| **Homepage URL** | http://localhost:3000 |
| **Callback URL** | http://localhost:8005/v1/github/callback |
| **Webhook URL** | https://your-public-url/v1/github/webhook |
| **Webhook secret** | (generate a random string) |

3. Permissions (Repository):
   - **Contents**: Read-only
   - **Metadata**: Read-only

4. Subscribe to events:
   - Push
   - Repository

## 🛠️ Setup

### 1. Install Dependencies

```bash
cd GitHub-Sync-Service
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Update `.env` with your GitHub App credentials:

```env
GITHUB_CLIENT_ID=your_client_id_here
GITHUB_CLIENT_SECRET=your_client_secret_here
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here
GITHUB_APP_ID=your_app_id_here
```

### 3. Run Database Migration

Run the SQL migration in Supabase SQL Editor:
```
sql/migration_v2_github_webhooks.sql
```

### 4. Start Server

```bash
python main.py
```

Server starts at http://127.0.0.1:8005

## 📚 API Endpoints

### Health Check
```
GET /health
```

### GitHub OAuth
```
GET /v1/github/authorize?user_id=<uuid>
→ Redirects to GitHub OAuth

GET /v1/github/callback?code=<code>&state=<state>
→ Handles OAuth callback

POST /v1/github/webhook
→ Receives GitHub webhook events
```

### Projects
```
POST /v1/projects/sync
Authorization: Bearer <token>
→ Syncs repositories from GitHub

GET /v1/projects?page=1&page_size=20&q=search
Authorization: Bearer <token>
→ Lists user's repositories

POST /v1/projects/:id/describe
Authorization: Bearer <token>
Body: { "regenerate": true }
→ Generates AI description

POST /v1/projects/:id/video/signed-url
Authorization: Bearer <token>
Body: { "content_type": "video/webm", "size": 10000000 }
→ Gets signed URL for video upload

GET /v1/projects/:id/video
Authorization: Bearer <token>
→ Gets video status and playback URL
```

## 🧪 Testing with Postman

### Step 1: Get Access Token
1. Login to frontend at http://localhost:3000
2. Open DevTools → Application → Local Storage
3. Find Supabase session and copy `access_token`

### Step 2: Connect GitHub (via Browser)
1. Go to: http://localhost:8005/v1/github/authorize?user_id=YOUR_USER_ID
2. Authorize on GitHub
3. You'll be redirected back to frontend

### Step 3: Sync Repositories
```
POST http://localhost:8005/v1/projects/sync
Headers:
  Authorization: Bearer YOUR_ACCESS_TOKEN
```

### Step 4: Generate AI Description
```
POST http://localhost:8005/v1/projects/REPO_ID/describe
Headers:
  Authorization: Bearer YOUR_ACCESS_TOKEN
Body:
  { "regenerate": true }
```

## 📁 Project Structure

```
GitHub-Sync-Service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables
├── README.md            # This file
└── sql/
    └── migration_v2_github_webhooks.sql  # Database schema
```

## 🔄 Webhook Events Handled

| Event | Action | Description |
|-------|--------|-------------|
| `push` | - | Updates repo sync status |
| `repository` | `created` | Adds new repository |
| `repository` | `deleted` | Removes repository |

## 🐛 Troubleshooting

### "GitHub not configured"
- Ensure GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET are set in .env

### "Token expired"
- User needs to reconnect GitHub via OAuth

### "Gemini quota exceeded"
- User should provide their own API key in profile settings

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| SUPABASE_URL | Supabase project URL | ✅ |
| SUPABASE_KEY | Supabase service role key | ✅ |
| JWT_SECRET | JWT signing secret | ✅ |
| GITHUB_CLIENT_ID | GitHub OAuth Client ID | ✅ |
| GITHUB_CLIENT_SECRET | GitHub OAuth Client Secret | ✅ |
| GITHUB_WEBHOOK_SECRET | Webhook signature secret | Optional |
| DEV_GEMINI_API_KEY | Fallback Gemini API key | Optional |
| FRONTEND_URL | Frontend URL for redirects | ✅ |
| SERVICE_URL | This service's public URL | ✅ |
