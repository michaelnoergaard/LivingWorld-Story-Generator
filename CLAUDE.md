# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LivingWorld is a Python project in early development. The codebase is currently minimal with only a "Hello World" entry point implemented.

## Development Commands

**Run the application:**
```bash
python main.py
```

**Install dependencies:**
```bash
# Always use uv sync to install dependencies
uv sync
```

For development dependencies:
```bash
uv sync --dev
```

## Project Structure

- `main.py` - Entry point with a simple `main()` function
- `pyproject.toml` - Modern Python project configuration (PEP 517/518)
- `ai_docs/` - Contains documentation/example content for AI-related features
- `.python-version` - Specifies Python 3.11+ requirement

## Python Environment

- Minimum Python version: 3.11
- Uses `pyproject.toml` for package configuration (no setup.py)
- Currently has no external dependencies

## Current State

This is a pre-alpha project with infrastructure in place but no core functionality developed yet.
