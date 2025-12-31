-- LivingWorld Initial Schema
-- Version: 001
-- Description: Create all tables for interactive story generation with pgvector support

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Stories table - main story sessions
CREATE TABLE IF NOT EXISTS stories (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    system_prompt TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    meta JSONB DEFAULT '{}'::jsonb
);

-- Scenes table - individual story scenes with embeddings
CREATE TABLE IF NOT EXISTS scenes (
    id SERIAL PRIMARY KEY,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    parent_scene_id INTEGER REFERENCES scenes(id) ON DELETE SET NULL,
    scene_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    raw_response TEXT,
    choices_generated JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding VECTOR(384),
    meta JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_scene UNIQUE(story_id, scene_number)
);

-- Choices table - track user choices
CREATE TABLE IF NOT EXISTS choices (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    choice_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    selected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Characters table with embeddings and agent fields
CREATE TABLE IF NOT EXISTS characters (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    first_appeared_in_scene INTEGER REFERENCES scenes(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding VECTOR(384),
    meta JSONB DEFAULT '{}'::jsonb,
    -- Agent-specific fields
    personality TEXT,
    goals TEXT,
    background TEXT,
    agent_config JSONB DEFAULT '{}'::jsonb
);

-- Scene-Characters junction table
CREATE TABLE IF NOT EXISTS scene_characters (
    scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    role VARCHAR(100),
    importance INTEGER DEFAULT 1,
    PRIMARY KEY (scene_id, character_id)
);

-- User instructions table with embeddings
CREATE TABLE IF NOT EXISTS user_instructions (
    id SERIAL PRIMARY KEY,
    scene_id INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    instruction TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    embedding VECTOR(384)
);

-- Memories table with embeddings
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    scene_id INTEGER REFERENCES scenes(id) ON DELETE SET NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50),
    importance FLOAT DEFAULT 0.5,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Character agent memories/conversations table
CREATE TABLE IF NOT EXISTS character_memories (
    id SERIAL PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    story_id INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    memory_type VARCHAR(50),
    content TEXT NOT NULL,
    emotional_valence FLOAT DEFAULT 0.0,
    importance FLOAT DEFAULT 0.5,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity indexes using IVFFlat
CREATE INDEX IF NOT EXISTS idx_scenes_embedding
ON scenes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_characters_embedding
ON characters USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_instructions_embedding
ON user_instructions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_memories_embedding
ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_character_memories_embedding
ON character_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_scenes_story_number ON scenes(story_id, scene_number);
CREATE INDEX IF NOT EXISTS idx_memories_story_type ON memories(story_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_character_memories_story ON character_memories(character_id, story_id);
CREATE INDEX IF NOT EXISTS idx_scene_characters_scene ON scene_characters(scene_id);
CREATE INDEX IF NOT EXISTS idx_scene_characters_character ON scene_characters(character_id);
CREATE INDEX IF NOT EXISTS idx_choices_scene ON choices(scene_id);
CREATE INDEX IF NOT EXISTS idx_stories_active ON stories(is_active) WHERE is_active = TRUE;
