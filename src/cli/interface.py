"""Interactive CLI interface for LivingWorld."""

import asyncio
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich import box
from rich.markdown import Markdown

from src.core.config import AppConfig
from src.core.exceptions import LivingWorldError
from src.core.logging_config import get_logger
from src.core.validation import (
    validate_id,
    validate_string,
    validate_choice,
    validate_file_path,
    validate_directory_path,
    validate_story_title,
    validate_story_setting,
    validate_content,
    ValidationError,
)
from src.database.connection import init_database, close_database, get_database
from src.embeddings.encoder import get_encoder
from src.llm.ollama_client import get_ollama_client
from src.llm.prompt_builder import PromptBuilder
from src.llm.story_generator import StoryGenerator
from src.story.state import StoryStateManager
from src.story.io import StoryExporter, StoryImporter

logger = get_logger(__name__)


class StoryCLI:
    """Interactive CLI for story generation."""

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize CLI.

        Args:
            config: Application configuration
        """
        self.config = config
        self.console = Console()
        self.prompt_ui = Prompt()

        # Components will be initialized later
        self.generator: Optional[StoryGenerator] = None
        self.current_story_id: Optional[int] = None
        
        # Internal thoughts display setting
        self.show_internal_thoughts = False

    async def initialize(self) -> None:
        """Initialize database and components."""
        logger.info("Initializing LivingWorld CLI application")
        try:
            # Initialize database (will create if doesn't exist)
            logger.debug("Initializing database connection")
            db_created = await init_database(self.config.database)
            if db_created:
                logger.info("Database created successfully and migrations applied")
                self.console.print("[green]Database created and migrations applied![/green]")
            else:
                logger.debug("Database already exists, skipping creation")

            db = get_database()
            logger.debug("Database connection established")

            # Initialize components
            logger.debug("Initializing encoder")
            encoder = get_encoder(self.config.embeddings)
            logger.info("Embedding encoder initialized")

            logger.debug("Initializing Ollama client")
            ollama_client = get_ollama_client(self.config.ollama)
            logger.info("Ollama client initialized with model: %s", self.config.ollama.model)

            # Check if Ollama model is available
            logger.debug("Checking if Ollama model is available")
            if not ollama_client.check_model_available():
                logger.warning("Model %s not found in Ollama, initiating download", self.config.ollama.model)
                self.console.print(
                    f"[yellow]Model {self.config.ollama.model} not found in Ollama.[/yellow]"
                )
                self.console.print("[cyan]Pulling model... this may take a while.[/cyan]")
                await ollama_client.pull_model()
                logger.info("Model downloaded successfully")
                self.console.print("[green]Model downloaded successfully![/green]")
            else:
                logger.debug("Model %s is available", self.config.ollama.model)

            logger.debug("Loading system prompt from: %s", self.config.story.default_system_prompt_path)
            prompt_builder = PromptBuilder(self.config.story.default_system_prompt_path)
            logger.info("System prompt loaded successfully")

            logger.debug("Initializing StoryGenerator")
            self.generator = StoryGenerator(
                ollama_client=ollama_client,
                prompt_builder=prompt_builder,
                encoder=encoder,
                session_factory=db.session_factory,
            )
            logger.info("StoryGenerator initialized")

            # Update agent factory with internal thoughts setting
            if self.generator.agent_factory:
                self.generator.agent_factory.set_show_internal_thoughts(self.show_internal_thoughts)
                logger.debug("Agent factory configured with show_internal_thoughts=%s", self.show_internal_thoughts)

            logger.info("Application initialization completed successfully")

        except LivingWorldError as e:
            logger.error("Application initialization failed: %s", e, exc_info=True)
            self.console.print(f"[red]Failed to initialize: {e}[/red]")
            raise
        except Exception as e:
            logger.exception("Unexpected error during initialization")
            self.console.print(f"[red]Failed to initialize: {e}[/red]")
            raise

    def display_welcome(self) -> None:
        """Display welcome message."""
        welcome_text = Text()
        welcome_text.append("Living World", style="bold magenta")
        welcome_text.append("\nInteractive Story Generator", style="cyan")
        welcome_text.append(
            "\n\nPowered by Ollama & PostgreSQL with pgvector",
            style="dim",
        )

        panel = Panel(
            welcome_text,
            border_style="magenta",
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def display_scene(self, content: str, choices: list[str], characters: Optional[list] = None) -> None:
        """
        Display scene content and choices.

        Args:
            content: Scene content
            choices: List of 3 choices
            characters: Optional list of character dicts present in scene
        """
        logger.debug("Displaying scene with %d choices", len(choices))
        logger.debug("Scene content length: %d characters", len(content))
        if characters:
            logger.debug("Scene has %d characters present", len(characters))

        # Display scene content
        scene_panel = Panel(
            content,
            title="Scene",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(scene_panel)

        # Display characters present if any
        if characters and len(characters) > 0:
            logger.debug("Displaying character table with %d characters", len(characters))
            from rich.table import Table

            char_table = Table(
                title="Characters Present",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
                title_style="bold magenta"
            )
            char_table.add_column("Name", style="cyan", width=20)
            char_table.add_column("Role", style="white", width=30)
            char_table.add_column("Mood", style="yellow", width=12)
            char_table.add_column("Thoughts", style="dim italic", width=25)

            for char in characters:
                # Get character info
                name = char.get('name', 'Unknown')
                description = char.get('description', '')
                mood = char.get('current_mood', 'unknown')
                internal_thought = char.get('internal_thought', '')

                # Truncate description if too long
                if description and len(description) > 30:
                    role = description[:30] + "..."
                else:
                    role = description or "A mysterious figure"

                # Style mood based on value
                mood_style = "green" if mood == "positive" else ("red" if mood == "negative" else "white")
                mood_display = f"[{mood_style}]{mood or 'neutral'}[/{mood_style}]"

                # Show internal thought if enabled and available
                if self.show_internal_thoughts and internal_thought:
                    # Truncate if too long
                    if len(internal_thought) > 25:
                        thought_display = f"\"{internal_thought[:25]}...\""
                    else:
                        thought_display = f"\"{internal_thought}\""
                else:
                    thought_display = "[dim](hidden)[/dim]"

                char_table.add_row(name, role, mood_display, thought_display)

            self.console.print()
            self.console.print(char_table)

        self.console.print()

        # Display choices
        self.console.print("[bold cyan]What do you do?[/bold cyan]")
        self.console.print()

        for i, choice in enumerate(choices, start=1):
            self.console.print(f"[dim]{i}.[/dim] {choice}")

        self.console.print()

    def parse_input(self, user_input: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Parse user input to extract choice and instruction.

        Args:
            user_input: Raw user input

        Returns:
            Tuple of (choice_number, instruction)
        """
        # Extract instruction in parentheses
        instruction_match = re.search(r"\(([^)]+)\)", user_input)
        instruction = instruction_match.group(1) if instruction_match else None

        # Extract choice number
        choice_match = re.search(r"^\s*(\d)", user_input)
        choice = int(choice_match.group(1)) if choice_match else None

        return choice, instruction

    async def start_new_story(self) -> None:
        """Start a new interactive story session."""
        logger.info("Starting new story session")
        self.console.print("[bold cyan]Starting a new story[/bold cyan]")
        self.console.print()

        # Get story title with validation
        while True:
            title = self.prompt_ui.ask("Enter a title for your story", default="My Adventure")
            try:
                validated_title = validate_story_title(title)
                logger.debug("Story title validated: %s", validated_title)
                break
            except ValueError as e:
                logger.debug("Story title validation failed: %s", e)
                self.console.print(f"[red]{e}[/red]")

        # Get story setting with validation
        self.console.print()
        while True:
            setting = self.prompt_ui.ask(
                "Describe the story setting",
                default="A remote Southeast Asian village with a beautiful beach",
            )
            try:
                validated_setting = validate_story_setting(setting)
                logger.debug("Story setting validated (length: %d)", len(validated_setting))
                break
            except ValueError as e:
                logger.debug("Story setting validation failed: %s", e)
                self.console.print(f"[red]{e}[/red]")

        # Get optional instructions
        self.console.print()
        self.console.print(
            "[dim]You can add optional instructions for the story (or press Enter to skip)[/dim]"
        )
        instructions = self.prompt_ui.ask("Instructions", default="")
        if instructions:
            logger.debug("User provided instructions (length: %d)", len(instructions))
        else:
            logger.debug("No user instructions provided")

        # Ask about internal thoughts
        self.console.print()
        thoughts_response = self.prompt_ui.ask(
            "Show NPC internal thoughts? (adds narrative depth)",
            choices=["yes", "no"],
            default="no",
        )
        self.show_internal_thoughts = thoughts_response.lower() == "yes"
        logger.info("NPC internal thoughts setting: %s", self.show_internal_thoughts)

        # Update agent factory
        if self.generator.agent_factory:
            self.generator.agent_factory.set_show_internal_thoughts(self.show_internal_thoughts)
            logger.debug("Agent factory updated with show_internal_thoughts=%s", self.show_internal_thoughts)

        status_msg = "[green]enabled[/green]" if self.show_internal_thoughts else "[dim]disabled[/dim]"
        self.console.print(f"NPC internal thoughts: {status_msg}")
        self.console.print()

        # Create story
        from src.story.state import StoryStateManager

        logger.debug("Creating story in database")
        state_manager = StoryStateManager(self.generator.session_factory)
        state = await state_manager.create_story(title=title)

        self.current_story_id = state.story_id
        logger.info("Story created with ID: %d, title: %s", state.story_id, title)

        self.console.print()
        self.console.print(f"[green]Story '{title}' created![/green]")
        self.console.print("[dim]Generating initial scene...[/dim]")
        self.console.print()

        # Generate initial scene
        try:
            logger.info("Generating initial scene for story %d with setting: %s", state.story_id, validated_setting[:100])
            scene = await self.generator.generate_initial_scene(
                story_id=state.story_id,
                story_setting=setting,
                user_instructions=instructions if instructions else None,
            )
            logger.info("Initial scene generated successfully, scene ID: %d", scene.id)

            # Get characters in this scene
            logger.debug("Fetching characters for scene %d", scene.id)
            characters = await self.get_scene_characters(scene.id)
            logger.debug("Found %d characters in scene", len(characters))

            # Display scene
            self.display_scene(scene.content, scene.choices, characters)

            # Interactive loop
            logger.info("Starting interactive story loop for story %d", state.story_id)
            await self.story_loop()

        except LivingWorldError as e:
            logger.error("Error during story generation for story %d: %s", state.story_id, e, exc_info=True)
            self.console.print(f"[red]Error: {e}[/red]")

    async def list_stories(self) -> None:
        """List all available stories."""
        logger.info("Listing all stories")
        try:
            state_manager = StoryStateManager(self.generator.session_factory)
            stories = await state_manager.list_stories(active_only=False)
            logger.debug("Found %d stories", len(stories))

            if not stories:
                logger.debug("No stories found in database")
                self.console.print("[yellow]No stories found.[/yellow]")
                return

            # Create table
            table = Table(title="Your Stories", box=box.ROUNDED)
            table.add_column("ID", style="cyan", width=6)
            table.add_column("Title", style="magenta")
            table.add_column("Scenes", justify="right", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Created", style="dim")
            table.add_column("Updated", style="dim")

            for story in stories:
                # Count scenes
                scene_count = len(story.scenes)
                logger.debug("Story %d: %s (%d scenes, status: %s)",
                           story.id, story.title, scene_count,
                           "Active" if story.is_active else "Archived")

                table.add_row(
                    str(story.id),
                    story.title,
                    str(scene_count),
                    "Active" if story.is_active else "Archived",
                    story.created_at.strftime("%Y-%m-%d"),
                    story.updated_at.strftime("%Y-%m-%d"),
                )

            self.console.print(table)
            self.console.print()

        except Exception as e:
            logger.error("Failed to list stories: %s", e, exc_info=True)
            self.console.print(f"[red]Error listing stories: {e}[/red]")

    async def load_story(self, story_id: int) -> None:
        """Load an existing story and continue playing."""
        logger.info("Loading story ID: %d", story_id)
        try:
            # Validate story ID
            validated_story_id = validate_id(story_id, field_name="story_id")
            logger.debug("Validated story ID: %d", validated_story_id)

            state_manager = StoryStateManager(self.generator.session_factory)
            state = await state_manager.load_story(validated_story_id)
            logger.info("Loaded story: %s (ID: %d, scene_number: %d)",
                       state.title, state.story_id, state.scene_number)

            self.current_story_id = state.story_id

            self.console.print(f"[green]Loaded story: {state.title}[/green]")
            self.console.print(f"[dim]Scene {state.scene_number}[/dim]")
            self.console.print()

            # Get last scene
            from src.database.models import Scene
            from sqlalchemy import select

            logger.debug("Fetching last scene (scene_id: %d)", state.current_scene_id)
            async with self.generator.session_factory() as session:
                result = await session.execute(
                    select(Scene).where(Scene.id == state.current_scene_id)
                )
                last_scene = result.scalar_one_or_none()

                if last_scene:
                    logger.debug("Last scene found: %d", last_scene.id)
                    # Display last scene
                    from src.database.models import Choice

                    result = await session.execute(
                        select(Choice)
                        .where(Choice.scene_id == last_scene.id)
                        .order_by(Choice.choice_number)
                    )
                    choices = result.scalars().all()
                    logger.debug("Found %d choices for scene", len(choices))

                    # Get characters in this scene
                    characters = await self.get_scene_characters(last_scene.id)
                    logger.debug("Found %d characters in scene", len(characters))

                    self.display_scene(last_scene.content, [c.content for c in choices], characters)

                    # Ask if user wants to continue
                    response = self.prompt_ui.ask(
                        "\nContinue from this scene? (y/n)",
                        default="y",
                    )

                    if response.lower() != "y":
                        logger.info("User chose not to continue story %d", validated_story_id)
                        return

                    logger.info("User chose to continue story %d", validated_story_id)

            # Enter story loop
            logger.info("Entering story loop for story %d", validated_story_id)
            await self.story_loop()

        except LivingWorldError as e:
            logger.error("Failed to load story %d: %s", story_id, e, exc_info=True)
            self.console.print(f"[red]Error loading story: {e}[/red]")
        except Exception as e:
            logger.exception("Unexpected error loading story %d", story_id)
            self.console.print(f"[red]Error loading story: {e}[/red]")

    async def export_story(self, story_id: Optional[int] = None) -> None:
        """Export a story to file."""
        logger.info("Exporting story (story_id: %s)", story_id)
        try:
            if story_id is None:
                # List stories and ask which to export
                await self.list_stories()
                story_id_str = self.prompt_ui.ask("Enter story ID to export", default="")

                if not story_id_str:
                    logger.debug("No story ID provided for export, cancelling")
                    return

                # Validate story ID
                validated_story_id = validate_id(story_id_str, field_name="story_id")
            else:
                validated_story_id = validate_id(story_id, field_name="story_id")

            logger.debug("Exporting story ID: %d", validated_story_id)

            # Ask for format and filename
            format_choice = self.prompt_ui.ask(
                "Export format",
                choices=["json", "markdown"],
                default="json",
            )
            logger.debug("Export format: %s", format_choice)

            default_filename = f"story_{validated_story_id}.{format_choice if format_choice == 'markdown' else 'md'}"
            output_path_str = self.prompt_ui.ask(
                "Output filename",
                default=default_filename,
            )

            # Validate output path
            output_path = validate_file_path(
                output_path_str,
                field_name="output_path",
                allowed_extensions=[".json", ".md", ".txt"]
            )
            logger.debug("Output path validated: %s", output_path)

            # Export
            exporter = StoryExporter()
            logger.info("Starting export of story %d to %s format", validated_story_id, format_choice)
            async with self.generator.session_factory() as session:
                if format_choice == "json":
                    path = await exporter.export_to_json(
                        session,
                        validated_story_id,
                        Path(output_path),
                    )
                else:
                    path = await exporter.export_to_markdown(
                        session,
                        validated_story_id,
                        Path(output_path),
                    )

            logger.info("Story %d exported successfully to: %s", validated_story_id, path)
            self.console.print(f"[green]Story exported to: {path}[/green]")

        except LivingWorldError as e:
            logger.error("Failed to export story %d: %s", story_id, e, exc_info=True)
            self.console.print(f"[red]Error exporting story: {e}[/red]")
        except (OSError, IOError) as e:
            logger.error("File system error exporting story %d to %s: %s", story_id, output_path, e, exc_info=True)
            self.console.print(f"[red]Error exporting story: {e}[/red]")
        except Exception as e:
            logger.exception("Unexpected error exporting story %d", story_id)
            self.console.print(f"[red]Error exporting story: {e}[/red]")

    async def import_story(self) -> None:
        """Import a story from file."""
        logger.info("Starting story import")
        try:
            # Ask for file path
            file_path_str = self.prompt_ui.ask(
                "Path to file to import",
                default="",
            )

            if not file_path_str:
                logger.debug("No file path provided for import, cancelling")
                return

            # Validate file path
            try:
                file_path = validate_file_path(
                    file_path_str,
                    field_name="import_file_path",
                    must_exist=True,
                    allowed_extensions=[".json", ".txt", ".md"]
                )
                logger.debug("Import file path validated: %s", file_path)
            except ValidationError as e:
                logger.debug("Import file path validation failed: %s", e)
                self.console.print(f"[red]{e}[/red]")
                return

            # Import based on file type
            logger.info("Importing story from %s (format: %s)", file_path, file_path.suffix)
            importer = StoryImporter()
            async with self.generator.session_factory() as session:
                if file_path.suffix == ".json":
                    logger.debug("Importing from JSON format")
                    story_id = await importer.import_from_json(
                        session,
                        file_path,
                        create_new_story=True,
                    )
                elif file_path.suffix in [".txt", ".md"]:
                    logger.debug("Importing from text/markdown format")
                    title = self.prompt_ui.ask("Story title", default="Imported Story")
                    content = file_path.read_text(encoding="utf-8")
                    logger.debug("Read %d characters from file", len(content))
                    story_id = await importer.import_from_text(
                        session,
                        title,
                        content,
                    )
                else:
                    logger.warning("Unsupported file format for import: %s", file_path.suffix)
                    self.console.print(
                        f"[red]Unsupported file format: {file_path.suffix}[/red]"
                    )
                    return

            logger.info("Story imported successfully with ID: %d", story_id)
            self.console.print(f"[green]Story imported with ID: {story_id}[/green]")

        except LivingWorldError as e:
            logger.error("Failed to import story from %s: %s", file_path, e, exc_info=True)
            self.console.print(f"[red]Error importing story: {e}[/red]")
        except (OSError, IOError, UnicodeDecodeError) as e:
            logger.error("File system error reading %s: %s", file_path, e, exc_info=True)
            self.console.print(f"[red]Error reading file: {e}[/red]")
        except Exception as e:
            logger.exception("Unexpected error importing story from %s", file_path)
            self.console.print(f"[red]Error importing story: {e}[/red]")

    async def show_settings(self) -> None:
        """Show and modify story settings."""
        logger.debug("Opening settings menu")
        self.console.print("\n[bold cyan]Settings[/bold cyan]\n")

        # Show current setting
        status = "[green]ON[/green]" if self.show_internal_thoughts else "[dim]OFF[/dim]"
        logger.debug("Current internal_thoughts setting: %s", self.show_internal_thoughts)
        self.console.print(f"1. NPC Internal Thoughts: {status}")
        self.console.print("2. Back to main menu\n")

        choice = self.prompt_ui.ask(
            "Choose an option",
            choices=["1", "2"],
            default="2",
        )

        if choice == "1":
            # Toggle internal thoughts
            new_value = not self.show_internal_thoughts
            logger.info("Toggling internal_thoughts from %s to %s", self.show_internal_thoughts, new_value)
            self.show_internal_thoughts = new_value

            # Update agent factory
            if self.generator and self.generator.agent_factory:
                self.generator.agent_factory.set_show_internal_thoughts(new_value)
                logger.debug("Agent factory updated with new internal_thoughts setting")

            status = "[green]ON[/green]" if new_value else "[dim]OFF[/dim]"
            self.console.print(f"\nNPC Internal Thoughts: {status}")

            if new_value:
                self.console.print("[dim italic]NPCs will now share their private thoughts with you.[/dim italic]")
            else:
                self.console.print("[dim italic]NPCs will keep their thoughts to themselves.[/dim italic]")

    async def show_main_menu(self) -> str:
        """Display main menu and get user choice."""
        self.console.print("\n[bold cyan]Main Menu[/bold cyan]\n")
        self.console.print("1. Start a new story")
        self.console.print("2. List stories")
        self.console.print("3. Load story")
        self.console.print("4. Settings")
        self.console.print("5. Export story")
        self.console.print("6. Import story")
        self.console.print("7. Quit")
        self.console.print()

        choice = self.prompt_ui.ask(
            "Choose an option",
            choices=["1", "2", "3", "4", "5", "6", "7"],
            default="1",
        )

        logger.debug("Main menu choice: %s", choice)
        return choice

    async def get_scene_characters(self, scene_id: int) -> list[dict]:
        """
        Get characters present in a scene.

        Args:
            scene_id: Scene ID

        Returns:
            List of character dictionaries
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        db = get_database()

        async with db.session_factory() as session:
            try:
                # Get scene_characters junction records
                from src.database.models import SceneCharacter, Character

                result = await session.execute(
                    select(SceneCharacter)
                    .options(selectinload(SceneCharacter.character))
                    .where(SceneCharacter.scene_id == scene_id)
                    .order_by(SceneCharacter.importance.desc())
                )

                scene_characters = result.scalars().all()

                # Build character info list
                characters = []
                for sc in scene_characters:
                    char = sc.character
                    char_info = {
                        'id': char.id,
                        'name': char.name,
                        'description': char.description,
                        'personality': char.personality,
                        'current_mood': char.current_mood,
                        'emotional_state': char.emotional_state,
                        'role': sc.role,
                        'importance': sc.importance
                    }
                    
                    # Get recent internal thought if enabled
                    if self.show_internal_thoughts:
                        from src.database.models import CharacterMemory
                        # Try to get the most recent internal thought
                        thought_result = await session.execute(
                            select(CharacterMemory)
                            .where(CharacterMemory.character_id == char.id)
                            .where(CharacterMemory.memory_type == "internal_thought")
                            .order_by(CharacterMemory.created_at.desc())
                            .limit(1)
                        )
                        recent_thought = thought_result.scalar_one_or_none()
                        if recent_thought:
                            # Clean up the thought content
                            thought_content = recent_thought.content
                            if thought_content.startswith("Internal thought: "):
                                thought_content = thought_content[18:]  # Remove prefix
                            char_info['internal_thought'] = thought_content
                    
                    characters.append(char_info)

                return characters

            except Exception as e:
                logger.warning("Could not load characters for scene %d: %s", current_scene_id, e, exc_info=True)
                self.console.print(f"[yellow]Warning: Could not load characters: {e}[/yellow]")
                return []

    async def story_loop(self) -> None:
        """Main interactive story loop."""
        logger.info("Entering story loop for story %d", self.current_story_id)
        while True:
            try:
                # Get user input
                user_input = self.prompt_ui.ask(
                    "\n[yellow]→[/yellow] ",
                    console=self.console,
                )

                # Check for special commands
                if user_input.lower() in ("quit", "exit", "q"):
                    logger.info("User chose to quit story %d", self.current_story_id)
                    self.console.print("[dim]Thanks for playing![/dim]")
                    break

                if user_input.lower() in ("thoughts on", "thoughtson"):
                    logger.info("User enabled internal thoughts display")
                    self.show_internal_thoughts = True
                    if self.generator and self.generator.agent_factory:
                        self.generator.agent_factory.set_show_internal_thoughts(True)
                    self.console.print("[green italic]NPC internal thoughts now visible[/green italic]")
                    continue

                if user_input.lower() in ("thoughts off", "thoughtsoff"):
                    logger.info("User disabled internal thoughts display")
                    self.show_internal_thoughts = False
                    if self.generator and self.generator.agent_factory:
                        self.generator.agent_factory.set_show_internal_thoughts(False)
                    self.console.print("[dim italic]NPC internal thoughts now hidden[/dim italic]")
                    continue

                # Parse input
                choice, instruction = self.parse_input(user_input)
                logger.debug("User input parsed - choice: %s, instruction: %s", choice, instruction)

                try:
                    validated_choice = validate_choice(choice)
                except ValueError as e:
                    logger.debug("Choice validation failed: %s", e)
                    self.console.print(f"[red]{e}[/red]")
                    continue

                if instruction:
                    # Validate instruction length and content
                    instruction = validate_string(
                        instruction,
                        field_name="instruction",
                        min_length=1,
                        max_length=200,
                        strip_whitespace=True,
                        allowed_chars=(
                            "abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "0123456789"
                            " ,.!?;:'\"-()"
                        )
                    )
                    logger.debug("User provided instruction: %s", instruction[:50])

                # Generate next scene
                self.console.print()
                self.console.print("[dim]Generating next scene...[/dim]")
                self.console.print()

                logger.info("Generating next scene for story %d, choice: %d", self.current_story_id, validated_choice)
                scene = await self.generator.generate_next_scene(
                    story_id=self.current_story_id,
                    choice=validated_choice,
                    user_instruction=instruction if instruction else None,
                )
                logger.info("Scene %d generated successfully", scene.id)

                # Get characters in this scene
                logger.debug("Fetching characters for scene %d", scene.id)
                characters = await self.get_scene_characters(scene.id)
                logger.debug("Found %d characters in new scene", len(characters))

                # Display scene
                self.display_scene(scene.content, scene.choices, characters)

            except KeyboardInterrupt:
                logger.info("Story %d interrupted by user", self.current_story_id)
                self.console.print("\n[dim]Story interrupted.[/dim]")
                break
            except LivingWorldError as e:
                logger.error("Error in story loop for story %d: %s", self.current_story_id, e, exc_info=True)
                self.console.print(f"[red]Error: {e}[/red]")
                continue

    async def run(self) -> None:
        """Run the CLI application with main menu."""
        logger.info("Starting LivingWorld CLI application")
        try:
            self.display_welcome()
            await self.initialize()

            while True:
                # Show main menu
                choice = await self.show_main_menu()

                if choice == "1":
                    await self.start_new_story()
                elif choice == "2":
                    await self.list_stories()
                elif choice == "3":
                    story_id_str = self.prompt_ui.ask("Enter story ID", default="")
                    if story_id_str:
                        try:
                            validated_story_id = validate_id(story_id_str, field_name="story_id")
                            await self.load_story(validated_story_id)
                        except ValueError as e:
                            logger.debug("Story ID validation failed: %s", e)
                            self.console.print(f"[red]{e}[/red]")
                elif choice == "4":
                    await self.show_settings()
                elif choice == "5":
                    await self.export_story()
                elif choice == "6":
                    await self.import_story()
                elif choice == "7":
                    logger.info("User chose to quit application")
                    self.console.print("[dim]Goodbye![/dim]")
                    break

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            self.console.print("\n[dim]Goodbye![/dim]")
        except Exception as e:
            logger.exception("Unexpected error in main menu loop")
            self.console.print(f"[red]Unexpected error: {e}[/red]")
            sys.exit(1)
        finally:
            # Cleanup
            logger.info("Closing database connection")
            await close_database()
            logger.info("LivingWorld CLI application terminated")


async def main_async(config: AppConfig) -> None:
    """
    Main async entry point.

    Args:
        config: Application configuration
    """
    cli = StoryCLI(config)
    await cli.run()


def main() -> None:
    """Main entry point."""
    from src.core.config import get_config

    # Load configuration
    config = get_config()

    # Run async main
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
