# LivingWorld

Interactive story generator powered by Ollama and PostgreSQL with pgvector semantic search.

## Features

- **AI-Powered Story Generation**: Uses Ollama's LLM models to generate interactive branching narratives
- **Semantic Search**: PostgreSQL with pgvector for intelligent content retrieval
- **Embedding-Based Context**: SentenceTransformer embeddings for story consistency
- **Rich CLI Interface**: Beautiful terminal interface powered by Rich
- **Character Agents**: LangChain-based agents for autonomous NPC personalities

## Prerequisites

- Python 3.11 or higher
- PostgreSQL with pgvector extension
- Ollama running locally

## Installation

### 1. Clone the repository

```bash
cd /home/michael/Projects/LivingWorld
```

### 2. Install Python dependencies

```bash
uv sync
```

Or install development dependencies:

```bash
uv sync --dev
```

### 3. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/)

Pull the required model:

```bash
ollama pull hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M
```

Or use any compatible model of your choice.

### 4. Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Database
DB_HOST=sql.micnor.dk
DB_PORT=5432
DB_NAME=livingworld
DB_USER=your_username
DB_PASSWORD=your_password

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M
OLLAMA_TIMEOUT=120

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### 5. Set up the database

Create the PostgreSQL database:

```bash
createdb -h sql.micnor.dk -U your_username livingworld
```

Enable pgvector extension (if not already enabled):

```bash
psql -h sql.micnor.dk -U your_username -d livingworld -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Run database migrations:

```bash
python -m src.database.migrate migrate
```

## Usage

### Start the application

```bash
python -m livingworld
```

Or install the package and use:

```bash
livingworld
```

### Interactive Story Flow

1. **Start a new story**: Enter a title and describe the setting
2. **Read the scene**: The AI generates an immersive scene with vivid descriptions
3. **Make a choice**: Select one of 3 numbered options to progress
4. **Add instructions** (optional): Add parenthetical instructions like `(ask her name)`
5. **Continue**: The story branches based on your choices

### Example session

```
Welcome to Living World!
Interactive Story Generator

Enter a title for your story: Paradise Beach
Describe the story setting: A remote Cambodian fishing village with a beautiful beach

[AI generates opening scene...]

What do you do?
1. Approach the village
2. Walk along the beach first
3. Set up camp nearby

→ 1 (Also look for friendly villagers)

[AI continues story based on your choice...]
```

## Database Management

### Run migrations

```bash
python -m src.database.migrate migrate
```

### Reset database (WARNING: Deletes all data)

```bash
python -m src.database.migrate reset
```

## Project Structure

```
LivingWorld/
├── main.py                      # Entry point
├── pyproject.toml               # Dependencies
├── .env.example                 # Environment template
├── README.md
│
├── src/
│   ├── core/                    # Configuration & exceptions
│   ├── llm/                     # Ollama client & story generation
│   ├── database/                # Database models & migrations
│   ├── embeddings/              # Semantic search & encoding
│   ├── story/                   # Story state management
│   ├── agents/                  # Character agents
│   └── cli/                     # Interactive CLI
│
└── prompts/
    └── system_prompt.txt        # Default system prompt
```

## Architecture

### Data Flow

1. User enters choice via CLI
2. Story state is retrieved from database
3. Context is built using semantic search:
   - Recent scenes
   - Relevant memories
   - Similar past situations
4. Ollama generates next scene
5. Embeddings created for semantic indexing
6. Scene saved to database with embeddings
7. Displayed to user with new choices

### Key Components

- **OllamaClient**: Async client for LLM communication with retry logic
- **EmbeddingEncoder**: SentenceTransformer wrapper for vector embeddings
- **SemanticSearch**: pgvector-based similarity search
- **StoryGenerator**: Orchestrates scene generation and storage
- **StoryStateManager**: Manages story state persistence

## Customization

### System Prompt

Edit `prompts/system_prompt.txt` to customize how the AI generates stories.

### Model

Change the model in `.env`:

```bash
OLLAMA_MODEL=llama3  # Or any other Ollama model
```

### Embedding Model

Change the embedding model in `.env`:

```bash
EMBEDDING_MODEL=all-mpnet-base-v2  # Higher quality, slower
```

## Troubleshooting

### Ollama connection issues

Make sure Ollama is running:

```bash
ollama list
```

Check Ollama is accessible:

```bash
curl http://localhost:11434/api/tags
```

### Database connection issues

Test database connection:

```bash
psql -h sql.micnor.dk -U your_username -d livingworld
```

Verify pgvector is installed:

```bash
psql -h sql.micnor.dk -U your_username -d livingworld -c "SELECT extversion FROM pg_extension WHERE extname='vector';"
```

### Model not found

The application will attempt to pull the model automatically. If that fails, pull manually:

```bash
ollama pull hf.co/TheDrummer/Cydonia-24B-v4.3-GGUF:Q4_K_M
```

## Development

### Running tests

```bash
pytest
```

### Code style

The project uses standard Python conventions. Consider using:

- `black` for code formatting
- `ruff` for linting
- `mypy` for type checking

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runtime
- [LangChain](https://langchain.com/) - LLM application framework
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search for PostgreSQL
- [SentenceTransformers](https://www.sbert.net/) - Embedding generation
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
