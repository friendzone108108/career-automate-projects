# main.py - GitHub Sync Service with Webhooks
# Syncs user's GitHub projects using OAuth and receives webhook events

from fastapi import FastAPI, HTTPException, Depends, Header, Request, status, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import httpx
import os
import hmac
import hashlib
from google import genai
from supabase import create_client, Client
from jose import jwt, JWTError
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "your_super_secret_key_here_123")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
DEV_GEMINI_API_KEY = os.getenv("DEV_GEMINI_API_KEY")

# Groq API for free genre detection
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Genre categories
GENRE_CATEGORIES = [
    "web-development",
    "mobile-app", 
    "machine-learning",
    "data-science",
    "devops",
    "game-development",
    "backend",
    "frontend",
    "api",
    "automation",
    "cli-tool",
    "library",
    "blockchain",
    "iot",
    "other"
]

# GitHub App Configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")

# URLs
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8005")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# State storage for OAuth (in production, use Redis)
oauth_states: dict = {}

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="GitHub Sync Service",
    description="Syncs GitHub projects via OAuth and webhooks, generates AI-powered resume summaries",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================
class RepositoryResponse(BaseModel):
    id: str
    name: str
    full_name: str
    html_url: str
    description: Optional[str]
    description_ai: Optional[str]
    language: Optional[str]
    stars_count: int
    has_intro_video: bool
    last_synced_at: str
    sync_status: str

class SyncResponse(BaseModel):
    success: bool
    message: str
    synced_count: int

class DescribeRequest(BaseModel):
    regenerate: bool = False

class DescribeResponse(BaseModel):
    description_ai: str

class VideoSignedUrlRequest(BaseModel):
    content_type: str
    size: int

class VideoSignedUrlResponse(BaseModel):
    upload_url: str
    storage_path: str

class GenreResponse(BaseModel):
    genres: List[str]
    detected_at: str

# ============================================================================
# Authentication Dependency
# ============================================================================
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Verify JWT token and extract user_id."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )
    
    token = parts[1]
    
    try:
        # Try Supabase token validation first
        user_response = supabase.auth.get_user(token)
        if user_response and user_response.user:
            return {
                "user_id": user_response.user.id,
                "email": user_response.user.email
            }
    except Exception:
        pass
    
    try:
        # Fallback to JWT decode
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_aud": False})
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"user_id": user_id, "email": payload.get("email")}
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}")

# ============================================================================
# Helper Functions
# ============================================================================
def get_gemini_api_key(user_id: str) -> str:
    """Get Gemini API key for the user."""
    try:
        profile = supabase.table("profiles").select("api_keys").eq("id", user_id).single().execute()
        if profile.data and profile.data.get("api_keys"):
            api_keys = profile.data["api_keys"]
            if isinstance(api_keys, dict) and api_keys.get("gemini_ai_key"):
                return api_keys["gemini_ai_key"]
    except Exception:
        pass
    
    if DEV_GEMINI_API_KEY:
        return DEV_GEMINI_API_KEY
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Gemini API key required. Configure in your profile settings."
    )

async def fetch_readme_content(owner: str, repo: str, access_token: str) -> Optional[str]:
    """Fetch README content from GitHub."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "CareerAutomate-GitHubSync"
    }
    
    async with httpx.AsyncClient() as client:
        for branch in ["main", "master"]:
            for readme in ["README.md", "readme.md", "Readme.md"]:
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{readme}"
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return response.text
    return None

def generate_ai_summary(readme_content: str, api_key: str) -> str:
    """Generate AI summary using Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""Read this project documentation. Summarize the project into a professional, first-person description ('I built...', 'Implemented...'). The summary must be exactly 4-5 lines long. Focus on the problem solved and the tech stack used. Do NOT use markdown. Do NOT use bullet points.

Project README:
{readme_content[:8000]}"""
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        if response and response.text:
            return response.text.strip()
        return "Project summary could not be generated."
        
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "rate" in error_msg or "resource_exhausted" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Gemini API quota exceeded. Please provide a different API key."
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI summarization failed: {str(e)}"
        )

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature."""
    if not GITHUB_WEBHOOK_SECRET:
        return True  # Skip verification in development
    
    expected_sig = "sha256=" + hmac.new(
        GITHUB_WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_sig, signature)

async def detect_project_genres(readme_content: str) -> List[str]:
    """Detect project genres using Groq API (free tier)."""
    if not GROQ_API_KEY:
        # Fallback to simple keyword-based detection
        return detect_genres_simple(readme_content)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are a project classifier. Analyze the README and return ONLY 2 genres from this list: {', '.join(GENRE_CATEGORIES)}. Return just the genre names separated by comma, nothing else."
                        },
                        {
                            "role": "user",
                            "content": f"Classify this project:\n\n{readme_content[:4000]}"
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                genres_text = data["choices"][0]["message"]["content"].strip().lower()
                # Parse genres
                genres = [g.strip() for g in genres_text.split(",")]
                # Filter to only valid genres
                valid_genres = [g for g in genres if g in GENRE_CATEGORIES][:2]
                return valid_genres if valid_genres else ["other"]
            else:
                return detect_genres_simple(readme_content)
                
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        return detect_genres_simple(readme_content)

def detect_genres_simple(readme_content: str) -> List[str]:
    """Simple keyword-based genre detection as fallback."""
    content_lower = readme_content.lower()
    detected = []
    
    # Web development
    web_keywords = ["react", "vue", "angular", "next.js", "html", "css", "frontend", "web app", "tailwind", "bootstrap"]
    if any(kw in content_lower for kw in web_keywords):
        detected.append("web-development")
    
    # Mobile app
    mobile_keywords = ["react native", "flutter", "swift", "kotlin", "android", "ios", "mobile app"]
    if any(kw in content_lower for kw in mobile_keywords):
        detected.append("mobile-app")
    
    # Machine Learning
    ml_keywords = ["tensorflow", "pytorch", "keras", "machine learning", "neural network", "deep learning", "model training"]
    if any(kw in content_lower for kw in ml_keywords):
        detected.append("machine-learning")
    
    # Data Science
    ds_keywords = ["pandas", "numpy", "data analysis", "jupyter", "visualization", "matplotlib", "data science"]
    if any(kw in content_lower for kw in ds_keywords):
        detected.append("data-science")
    
    # DevOps
    devops_keywords = ["docker", "kubernetes", "ci/cd", "terraform", "ansible", "devops", "aws", "azure", "gcp"]
    if any(kw in content_lower for kw in devops_keywords):
        detected.append("devops")
    
    # Backend
    backend_keywords = ["fastapi", "express", "django", "flask", "node.js", "api", "rest", "graphql", "server"]
    if any(kw in content_lower for kw in backend_keywords):
        detected.append("backend")
    
    # API
    api_keywords = ["api", "rest", "graphql", "endpoint", "microservice"]
    if any(kw in content_lower for kw in api_keywords):
        detected.append("api")
    
    # Automation
    auto_keywords = ["automation", "bot", "scraper", "script", "cronjob"]
    if any(kw in content_lower for kw in auto_keywords):
        detected.append("automation")
    
    # Game development
    game_keywords = ["unity", "unreal", "godot", "game", "pygame", "phaser"]
    if any(kw in content_lower for kw in game_keywords):
        detected.append("game-development")
    
    # Return top 2 or 'other' if none found
    return detected[:2] if detected else ["other"]

# ============================================================================
# GitHub OAuth Endpoints
# ============================================================================
@app.get("/v1/github/authorize")
async def github_authorize(user_id: str = Query(...)):
    """Initiate GitHub OAuth flow."""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GitHub App not configured. Set GITHUB_CLIENT_ID."
        )
    
    # Generate state token
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"user_id": user_id, "created_at": datetime.now(timezone.utc)}
    
    # Build GitHub OAuth URL
    scopes = "read:user,repo"
    github_auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri={SERVICE_URL}/v1/github/callback"
        f"&scope={scopes}"
        f"&state={state}"
    )
    
    return RedirectResponse(url=github_auth_url)

@app.get("/v1/github/callback")
async def github_callback(code: str = Query(None), state: str = Query(None), error: str = Query(None)):
    """Handle GitHub OAuth callback."""
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/projects?error={error}")
    
    if not code or not state:
        return RedirectResponse(f"{FRONTEND_URL}/projects?error=Missing+code+or+state")
    
    # Validate state
    state_data = oauth_states.pop(state, None)
    if not state_data:
        return RedirectResponse(f"{FRONTEND_URL}/projects?error=Invalid+state")
    
    user_id = state_data["user_id"]
    
    try:
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                    "redirect_uri": f"{SERVICE_URL}/v1/github/callback"
                },
                headers={"Accept": "application/json"}
            )
            
            token_data = token_response.json()
            
            if "error" in token_data:
                return RedirectResponse(f"{FRONTEND_URL}/projects?error={token_data.get('error_description', 'OAuth failed')}")
            
            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")
            
            # Get GitHub user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            
            github_user = user_response.json()
        
        # Store GitHub integration
        integration_data = {
            "user_id": user_id,
            "github_user_id": github_user["id"],
            "github_username": github_user["login"],
            "access_token": access_token,
            "refresh_token": refresh_token,
            "scopes": token_data.get("scope", "").split(","),
            "is_active": True
        }
        
        supabase.table("github_integrations").upsert(
            integration_data,
            on_conflict="user_id"
        ).execute()
        
        return RedirectResponse(f"{FRONTEND_URL}/projects?github_connected=true")
        
    except Exception as e:
        print(f"GitHub OAuth error: {str(e)}")
        return RedirectResponse(f"{FRONTEND_URL}/projects?error=OAuth+failed")

# ============================================================================
# Projects Endpoints
# ============================================================================
@app.post("/v1/projects/sync", response_model=SyncResponse)
async def sync_projects(
    current_user: dict = Depends(get_current_user),
    background_tasks: BackgroundTasks = None
):
    """Sync repositories from GitHub."""
    user_id = current_user["user_id"]
    
    # Get GitHub integration
    integration = supabase.table("github_integrations").select("*").eq("user_id", user_id).single().execute()
    
    if not integration.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="GitHub not connected. Please connect your GitHub account first."
        )
    
    access_token = integration.data["access_token"]
    github_username = integration.data["github_username"]
    
    # Fetch repositories from GitHub
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.github.com/user/repos",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github.v3+json"
            },
            params={
                "type": "owner",
                "sort": "updated",
                "per_page": 100
            }
        )
        
        if response.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="GitHub token expired. Please reconnect your GitHub account."
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to fetch repositories from GitHub"
            )
        
        repos = response.json()
    
    # Sync each repository with README content
    synced_count = 0
    readme_fetched_count = 0
    
    for repo in repos:
        try:
            repo_name = repo["name"]
            owner = repo["full_name"].split("/")[0]
            
            # Check if repo already exists with README content
            existing = supabase.table("repositories").select("id, readme_content, readme_last_fetched_at").eq("user_id", user_id).eq("provider_repo_id", repo["id"]).execute()
            
            # Prepare base repo data
            repo_data = {
                "user_id": user_id,
                "provider_repo_id": repo["id"],
                "name": repo_name,
                "full_name": repo["full_name"],
                "html_url": repo["html_url"],
                "description": repo["description"],
                "default_branch": repo.get("default_branch", "main"),
                "language": repo.get("language"),
                "topics": repo.get("topics", []),
                "stars_count": repo.get("stargazers_count", 0),
                "forks_count": repo.get("forks_count", 0),
                "is_private": repo.get("private", False),
                "is_fork": repo.get("fork", False),
                "last_synced_at": datetime.now(timezone.utc).isoformat(),
                "sync_status": "synced"
            }
            
            # If repo doesn't exist or doesn't have README content, fetch it
            should_fetch_readme = True
            if existing.data and len(existing.data) > 0:
                existing_repo = existing.data[0]
                # Only fetch README if it doesn't exist yet
                if existing_repo.get("readme_content"):
                    should_fetch_readme = False
            
            if should_fetch_readme:
                readme_content = await fetch_readme_content(owner, repo_name, access_token)
                if readme_content:
                    repo_data["readme_content"] = readme_content[:15000]  # Store up to 15KB
                    repo_data["readme_last_fetched_at"] = datetime.now(timezone.utc).isoformat()
                    readme_fetched_count += 1
            
            # Upsert repository (won't create duplicates due to UNIQUE constraint)
            supabase.table("repositories").upsert(
                repo_data,
                on_conflict="user_id,provider_repo_id"
            ).execute()
            
            synced_count += 1
            
        except Exception as e:
            print(f"Error syncing repo {repo['name']}: {str(e)}")
            continue
    
    return SyncResponse(
        success=True,
        message=f"Synced {synced_count} repositories. Fetched README for {readme_fetched_count} new repos.",
        synced_count=synced_count
    )

@app.get("/v1/projects")
async def get_projects(
    current_user: dict = Depends(get_current_user),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    q: Optional[str] = Query(None)
):
    """Get user's synced repositories."""
    user_id = current_user["user_id"]
    
    query = supabase.table("repositories").select("*").eq("user_id", user_id)
    
    if q:
        query = query.ilike("name", f"%{q}%")
    
    # Pagination
    offset = (page - 1) * page_size
    query = query.range(offset, offset + page_size - 1).order("last_synced_at", desc=True)
    
    result = query.execute()
    
    return {
        "success": True,
        "page": page,
        "page_size": page_size,
        "count": len(result.data) if result.data else 0,
        "projects": result.data or []
    }

@app.post("/v1/projects/{repo_id}/describe", response_model=DescribeResponse)
async def describe_project(
    repo_id: str,
    request: DescribeRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate AI description for a project.
    
    Uses stored README content from database first.
    Only fetches from GitHub if README is not in database.
    Stores both README and AI description for future use.
    """
    user_id = current_user["user_id"]
    
    # Get repository with all fields
    repo = supabase.table("repositories").select("*").eq("id", repo_id).eq("user_id", user_id).single().execute()
    
    if not repo.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    
    # Check if we already have a description and regenerate is false
    if repo.data.get("description_ai") and not request.regenerate:
        return DescribeResponse(description_ai=repo.data["description_ai"])
    
    # Try to use stored README content first
    readme_content = repo.data.get("readme_content")
    
    # If no stored README, fetch from GitHub
    if not readme_content:
        # Get GitHub integration for access token
        integration = supabase.table("github_integrations").select("access_token").eq("user_id", user_id).single().execute()
        
        if not integration.data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="GitHub not connected")
        
        owner = repo.data["full_name"].split("/")[0]
        readme_content = await fetch_readme_content(owner, repo.data["name"], integration.data["access_token"])
        
        if not readme_content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No README found in this repository"
            )
        
        # Store the fetched README for future use
        supabase.table("repositories").update({
            "readme_content": readme_content[:15000],
            "readme_last_fetched_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", repo_id).execute()
    
    # Get Gemini API key and generate summary
    gemini_key = get_gemini_api_key(user_id)
    ai_summary = generate_ai_summary(readme_content, gemini_key)
    
    # Update repository with AI description
    supabase.table("repositories").update({
        "description_ai": ai_summary
    }).eq("id", repo_id).execute()
    
    return DescribeResponse(description_ai=ai_summary)

@app.post("/v1/projects/generate-all-descriptions")
async def generate_all_descriptions(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(10, ge=1, le=50, description="Max repos to process")
):
    """
    Generate AI descriptions for all repos that have README but no AI description.
    
    This is a batch operation - use with caution as it consumes Gemini API quota.
    Processes repos without description_ai but with readme_content.
    """
    user_id = current_user["user_id"]
    
    # Get repos with README but without AI description
    repos = supabase.table("repositories").select(
        "id, name, readme_content"
    ).eq("user_id", user_id).is_("description_ai", "null").not_.is_("readme_content", "null").limit(limit).execute()
    
    if not repos.data or len(repos.data) == 0:
        return {
            "success": True,
            "message": "No repositories need AI descriptions",
            "processed_count": 0,
            "results": []
        }
    
    # Get Gemini API key once
    gemini_key = get_gemini_api_key(user_id)
    
    results = []
    processed_count = 0
    
    for repo in repos.data:
        try:
            readme_content = repo.get("readme_content")
            if not readme_content:
                continue
            
            # Generate AI summary
            ai_summary = generate_ai_summary(readme_content, gemini_key)
            
            # Update repository
            supabase.table("repositories").update({
                "description_ai": ai_summary
            }).eq("id", repo["id"]).execute()
            
            results.append({
                "id": repo["id"],
                "name": repo["name"],
                "status": "success",
                "description_ai": ai_summary[:200] + "..." if len(ai_summary) > 200 else ai_summary
            })
            processed_count += 1
            
        except HTTPException as e:
            # Stop on quota exceeded
            if e.status_code == 429:
                results.append({
                    "id": repo["id"],
                    "name": repo["name"],
                    "status": "quota_exceeded",
                    "error": str(e.detail)
                })
                break
        except Exception as e:
            results.append({
                "id": repo["id"],
                "name": repo["name"],
                "status": "error",
                "error": str(e)
            })
    
    return {
        "success": True,
        "message": f"Generated AI descriptions for {processed_count} repositories",
        "processed_count": processed_count,
        "results": results
    }

@app.get("/v1/projects/for-resume")
async def get_projects_for_resume(
    current_user: dict = Depends(get_current_user)
):
    """
    Get all projects ready for resume generation.
    
    Returns repos that have AI descriptions generated.
    This endpoint is designed for the Resume Generation Stack to consume.
    """
    user_id = current_user["user_id"]
    
    # Get repos with AI descriptions
    repos = supabase.table("repositories").select(
        "id, name, full_name, html_url, description, description_ai, language, topics, stars_count"
    ).eq("user_id", user_id).not_.is_("description_ai", "null").order("stars_count", desc=True).execute()
    
    return {
        "success": True,
        "count": len(repos.data) if repos.data else 0,
        "projects": repos.data or []
    }

@app.get("/v1/projects/{repo_id}/readme")
async def get_project_readme(
    repo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the stored README content for a project.
    
    Returns the README content stored in database.
    Useful for resume generation or other processing.
    """
    user_id = current_user["user_id"]
    
    repo = supabase.table("repositories").select(
        "id, name, readme_content, readme_last_fetched_at"
    ).eq("id", repo_id).eq("user_id", user_id).single().execute()
    
    if not repo.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    
    if not repo.data.get("readme_content"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="README content not available. Please sync repositories first."
        )
    
    return {
        "success": True,
        "repo_id": repo_id,
        "name": repo.data["name"],
        "readme_content": repo.data["readme_content"],
        "last_fetched_at": repo.data.get("readme_last_fetched_at")
    }

@app.post("/v1/projects/{repo_id}/detect-genre", response_model=GenreResponse)
async def detect_project_genre(
    repo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect project genres using AI (Groq) or keyword fallback.
    
    Returns up to 2 genres for the project.
    Uses free Groq API if available, otherwise falls back to keyword detection.
    """
    user_id = current_user["user_id"]
    
    # Get repository
    repo = supabase.table("repositories").select(
        "id, name, readme_content, genres"
    ).eq("id", repo_id).eq("user_id", user_id).single().execute()
    
    if not repo.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    
    # Check if README exists
    readme_content = repo.data.get("readme_content")
    if not readme_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="README content not available. Please sync repositories first."
        )
    
    # Detect genres
    genres = await detect_project_genres(readme_content)
    detected_at = datetime.now(timezone.utc).isoformat()
    
    # Store in database
    supabase.table("repositories").update({
        "genres": genres,
        "genre_detected_at": detected_at
    }).eq("id", repo_id).execute()
    
    return GenreResponse(genres=genres, detected_at=detected_at)

@app.post("/v1/projects/{repo_id}/video/signed-url", response_model=VideoSignedUrlResponse)
async def get_video_signed_url(
    repo_id: str,
    request: VideoSignedUrlRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get a signed URL for video upload."""
    user_id = current_user["user_id"]
    
    # Verify repository belongs to user
    repo = supabase.table("repositories").select("id").eq("id", repo_id).eq("user_id", user_id).single().execute()
    
    if not repo.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    
    # Generate storage path
    storage_path = f"{user_id}/{repo_id}/{secrets.token_urlsafe(8)}.webm"
    
    # Create signed URL (Supabase storage)
    # Note: This is a placeholder - actual implementation depends on Supabase storage API
    upload_url = f"{SUPABASE_URL}/storage/v1/object/project-videos/{storage_path}"
    
    return VideoSignedUrlResponse(
        upload_url=upload_url,
        storage_path=storage_path
    )

@app.get("/v1/projects/{repo_id}/video")
async def get_video_status(
    repo_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get video status for a project."""
    user_id = current_user["user_id"]
    
    # Get video record
    video = supabase.table("project_videos").select("*").eq("repo_id", repo_id).eq("user_id", user_id).single().execute()
    
    if not video.data:
        return {"status": "no_video", "playback_url": None}
    
    return {
        "status": video.data["status"],
        "playback_url": video.data.get("playback_url"),
        "duration_sec": video.data.get("duration_sec")
    }

# ============================================================================
# Webhook Endpoint
# ============================================================================
@app.post("/v1/github/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: Optional[str] = Header(None),
    x_github_event: Optional[str] = Header(None),
    x_github_delivery: Optional[str] = Header(None)
):
    """Handle GitHub webhook events."""
    payload = await request.body()
    
    # Verify signature
    if x_hub_signature_256 and not verify_webhook_signature(payload, x_hub_signature_256):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")
    
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")
    
    # Log the event
    event_log = {
        "event_type": x_github_event,
        "action": data.get("action"),
        "delivery_id": x_github_delivery,
        "payload": data,
        "installation_id": data.get("installation", {}).get("id"),
        "processed": False
    }
    
    try:
        supabase.table("github_webhook_events").insert(event_log).execute()
    except Exception as e:
        print(f"Failed to log webhook event: {str(e)}")
    
    # Process event based on type
    if x_github_event == "push":
        # Repository was pushed to - might want to refresh README
        repo_data = data.get("repository", {})
        sender = data.get("sender", {})
        
        # Find the user with this GitHub account
        integration = supabase.table("github_integrations").select("user_id").eq("github_user_id", sender.get("id")).single().execute()
        
        if integration.data:
            # Update the repository's sync status
            supabase.table("repositories").update({
                "sync_status": "pending",
                "last_synced_at": datetime.now(timezone.utc).isoformat()
            }).eq("provider_repo_id", repo_data.get("id")).eq("user_id", integration.data["user_id"]).execute()
    
    elif x_github_event == "repository":
        action = data.get("action")
        repo_data = data.get("repository", {})
        sender = data.get("sender", {})
        
        integration = supabase.table("github_integrations").select("user_id").eq("github_user_id", sender.get("id")).single().execute()
        
        if integration.data:
            if action == "created":
                # New repository - add it
                new_repo = {
                    "user_id": integration.data["user_id"],
                    "provider_repo_id": repo_data["id"],
                    "name": repo_data["name"],
                    "full_name": repo_data["full_name"],
                    "html_url": repo_data["html_url"],
                    "description": repo_data.get("description"),
                    "default_branch": repo_data.get("default_branch", "main"),
                    "language": repo_data.get("language"),
                    "is_private": repo_data.get("private", False),
                    "last_synced_at": datetime.now(timezone.utc).isoformat(),
                    "sync_status": "synced"
                }
                supabase.table("repositories").upsert(new_repo, on_conflict="user_id,provider_repo_id").execute()
                
            elif action == "deleted":
                # Repository deleted - remove it
                supabase.table("repositories").delete().eq("provider_repo_id", repo_data["id"]).eq("user_id", integration.data["user_id"]).execute()
    
    return {"received": True}

# ============================================================================
# Health Check
# ============================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "GitHub Sync Service",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# ============================================================================
# Run Server
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8005))
    
    print(f"🚀 Starting GitHub Sync Service v2.0 on http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"🔗 GitHub OAuth Callback: {SERVICE_URL}/v1/github/callback")
    print(f"🔔 GitHub Webhook URL: {SERVICE_URL}/v1/github/webhook")
    
    uvicorn.run(app, host=host, port=port)
