-- ============================================================================
-- Migration V2: GitHub Webhooks Architecture
-- Run this in Supabase SQL Editor
-- ============================================================================

-- ============================================================================
-- STEP 1: ROLLBACK PREVIOUS CHANGES TO projects TABLE
-- ============================================================================

-- Drop the unique constraint we added previously
ALTER TABLE public.projects 
DROP CONSTRAINT IF EXISTS unique_user_repo;

-- Drop the indexes we added previously
DROP INDEX IF EXISTS idx_projects_github_repo_url;
DROP INDEX IF EXISTS idx_projects_synced_at;

-- Remove the columns we added previously (if they exist)
ALTER TABLE public.projects 
DROP COLUMN IF EXISTS github_repo_url,
DROP COLUMN IF EXISTS readme_content,
DROP COLUMN IF EXISTS synced_at;

-- ============================================================================
-- STEP 2: CREATE NEW GITHUB INTEGRATIONS TABLE
-- Stores user's GitHub App installation and access tokens
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.github_integrations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- GitHub App Installation
    installation_id BIGINT,  -- GitHub App installation ID
    github_user_id BIGINT NOT NULL,  -- GitHub user ID
    github_username TEXT NOT NULL,  -- GitHub username
    
    -- OAuth Tokens
    access_token TEXT NOT NULL,  -- GitHub access token (encrypted in production)
    refresh_token TEXT,  -- For token refresh
    token_expires_at TIMESTAMPTZ,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    scopes TEXT[],  -- Granted scopes
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id),  -- One GitHub integration per user
    UNIQUE(github_user_id)  -- One user per GitHub account
);

-- ============================================================================
-- STEP 3: CREATE REPOSITORIES TABLE
-- Synced repositories from GitHub
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.repositories (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    
    -- GitHub Repository Info
    provider_repo_id BIGINT NOT NULL,  -- GitHub repo ID
    name TEXT NOT NULL,  -- Repo name
    full_name TEXT NOT NULL,  -- owner/repo format
    html_url TEXT NOT NULL,  -- GitHub URL
    description TEXT,  -- Original GitHub description
    description_ai TEXT,  -- AI-generated resume description
    
    -- Repo Details
    default_branch TEXT DEFAULT 'main',
    language TEXT,  -- Primary programming language
    topics TEXT[],  -- Repository topics/tags
    stars_count INTEGER DEFAULT 0,
    forks_count INTEGER DEFAULT 0,
    is_private BOOLEAN DEFAULT FALSE,
    is_fork BOOLEAN DEFAULT FALSE,
    
    -- README Content (for AI processing)
    readme_content TEXT,
    readme_last_fetched_at TIMESTAMPTZ,
    
    -- Video Feature
    has_intro_video BOOLEAN DEFAULT FALSE,
    
    -- Sync Status
    last_synced_at TIMESTAMPTZ DEFAULT NOW(),
    sync_status TEXT DEFAULT 'synced',  -- synced, pending, error
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, provider_repo_id)  -- One entry per repo per user
);

-- ============================================================================
-- STEP 4: CREATE PROJECT VIDEOS TABLE
-- Video recordings/uploads for project introductions
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.project_videos (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    repo_id UUID NOT NULL REFERENCES public.repositories(id) ON DELETE CASCADE,
    
    -- Storage Info
    storage_path TEXT NOT NULL,  -- Path in Supabase Storage
    storage_bucket TEXT DEFAULT 'project-videos',
    
    -- Video Metadata
    file_name TEXT,
    file_size BIGINT,  -- Size in bytes
    content_type TEXT,  -- MIME type
    duration_sec INTEGER,  -- Duration in seconds
    
    -- Processing Status
    status TEXT DEFAULT 'uploaded',  -- uploaded, transcoding, transcoded, failed
    transcode_error TEXT,
    
    -- URLs
    playback_url TEXT,
    thumbnail_url TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(repo_id)  -- One video per repository
);

-- ============================================================================
-- STEP 5: CREATE WEBHOOK EVENTS LOG TABLE
-- For debugging and audit trail
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.github_webhook_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- Event Info
    event_type TEXT NOT NULL,  -- push, repository, create, etc.
    action TEXT,  -- created, deleted, etc.
    delivery_id TEXT,  -- GitHub delivery ID
    
    -- Payload
    payload JSONB NOT NULL,
    
    -- Related Entities
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    repo_id UUID REFERENCES public.repositories(id) ON DELETE SET NULL,
    installation_id BIGINT,
    
    -- Processing
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,
    error TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- STEP 6: CREATE INDEXES
-- ============================================================================

-- GitHub Integrations
CREATE INDEX IF NOT EXISTS idx_github_integrations_user_id 
ON public.github_integrations(user_id);

CREATE INDEX IF NOT EXISTS idx_github_integrations_github_user_id 
ON public.github_integrations(github_user_id);

CREATE INDEX IF NOT EXISTS idx_github_integrations_installation_id 
ON public.github_integrations(installation_id);

-- Repositories
CREATE INDEX IF NOT EXISTS idx_repositories_user_id 
ON public.repositories(user_id);

CREATE INDEX IF NOT EXISTS idx_repositories_provider_repo_id 
ON public.repositories(provider_repo_id);

CREATE INDEX IF NOT EXISTS idx_repositories_last_synced_at 
ON public.repositories(last_synced_at);

-- Project Videos
CREATE INDEX IF NOT EXISTS idx_project_videos_user_id 
ON public.project_videos(user_id);

CREATE INDEX IF NOT EXISTS idx_project_videos_repo_id 
ON public.project_videos(repo_id);

-- Webhook Events
CREATE INDEX IF NOT EXISTS idx_github_webhook_events_event_type 
ON public.github_webhook_events(event_type);

CREATE INDEX IF NOT EXISTS idx_github_webhook_events_processed 
ON public.github_webhook_events(processed);

-- ============================================================================
-- STEP 7: CREATE TRIGGERS FOR updated_at
-- ============================================================================

CREATE TRIGGER github_integrations_updated
    BEFORE UPDATE ON public.github_integrations
    FOR EACH ROW
    EXECUTE PROCEDURE public.handle_updated_at();

CREATE TRIGGER repositories_updated
    BEFORE UPDATE ON public.repositories
    FOR EACH ROW
    EXECUTE PROCEDURE public.handle_updated_at();

CREATE TRIGGER project_videos_updated
    BEFORE UPDATE ON public.project_videos
    FOR EACH ROW
    EXECUTE PROCEDURE public.handle_updated_at();

-- ============================================================================
-- STEP 8: ENABLE ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- GitHub Integrations
ALTER TABLE public.github_integrations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own github integration" 
ON public.github_integrations FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own github integration" 
ON public.github_integrations FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own github integration" 
ON public.github_integrations FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own github integration" 
ON public.github_integrations FOR DELETE USING (auth.uid() = user_id);

-- Repositories
ALTER TABLE public.repositories ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own repositories" 
ON public.repositories FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own repositories" 
ON public.repositories FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own repositories" 
ON public.repositories FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own repositories" 
ON public.repositories FOR DELETE USING (auth.uid() = user_id);

-- Project Videos
ALTER TABLE public.project_videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own project videos" 
ON public.project_videos FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own project videos" 
ON public.project_videos FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own project videos" 
ON public.project_videos FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own project videos" 
ON public.project_videos FOR DELETE USING (auth.uid() = user_id);

-- Webhook Events (service role only, no user access needed)
ALTER TABLE public.github_webhook_events ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (for the microservice)
CREATE POLICY "Service role full access to webhook events" 
ON public.github_webhook_events FOR ALL 
USING (auth.role() = 'service_role')
WITH CHECK (auth.role() = 'service_role');

-- ============================================================================
-- STEP 9: CREATE STORAGE BUCKET FOR PROJECT VIDEOS
-- ============================================================================

INSERT INTO storage.buckets (id, name, public)
VALUES ('project-videos', 'project-videos', false)
ON CONFLICT (id) DO NOTHING;

-- Storage Policies
CREATE POLICY "Users can upload own project videos"
ON storage.objects FOR INSERT
WITH CHECK (
    bucket_id = 'project-videos' 
    AND auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can read own project videos"
ON storage.objects FOR SELECT
USING (
    bucket_id = 'project-videos' 
    AND auth.uid()::text = (storage.foldername(name))[1]
);

CREATE POLICY "Users can delete own project videos"
ON storage.objects FOR DELETE
USING (
    bucket_id = 'project-videos' 
    AND auth.uid()::text = (storage.foldername(name))[1]
);

-- ============================================================================
-- STEP 10: VERIFY SCHEMA
-- ============================================================================

SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public' 
    AND table_name IN ('github_integrations', 'repositories', 'project_videos', 'github_webhook_events')
ORDER BY table_name, ordinal_position;
