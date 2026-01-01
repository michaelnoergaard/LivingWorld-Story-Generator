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
from src.database.connection import init_database, close_database, get_database
from src.embeddings.encoder import get_encoder
from src.llm.ollama_client import get_ollama_client
from src.llm.prompt_builder import PromptBuilder
from src.llm.story_generator import StoryGenerator
from src.story.state import StoryStateManager
from src.story.io import StoryExporter, StoryImporter


class StoryCLI:
    """Interactive CLI for story generation."""

    def __init__(self, config: AppConfig):
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

    async def initialize(self):
        """Initialize database and components."""
        try:
            # Initialize database (will create if doesn't exist)
            db_created = await init_database(self.config.database)
            if db_created:
                self.console.print("[green]Database created and migrations applied![/green]")

            db = get_database()

            # Initialize components
            encoder = get_encoder(self.config.embeddings)
            ollama_client = get_ollama_client(self.config.ollama)

            # Check if Ollama model is available
            if not ollama_client.check_model_available():
                self.console.print(
                    f"[yellow]Model {self.config.ollama.model} not found in Ollama.[/yellow]"
                )
                self.console.print("[cyan]Pulling model... this may take a while.[/cyan]")
                await ollama_client.pull_model()
                self.console.print("[green]Model downloaded successfully![/green]")

            prompt_builder = PromptBuilder(self.config.story.default_system_prompt_path)

            self.generator = StoryGenerator(
                ollama_client=ollama_client,
                prompt_builder=prompt_builder,
                encoder=encoder,
                session_factory=db.session_factory,
            )
            
            # Update agent factory with internal thoughts setting
            if self.generator.agent_factory:
                self.generator.agent_factory.set_show_internal_thoughts(self.show_internal_thoughts)

        except Exception as e:
            self.console.print(f"[red]Failed to initialize: {e}[/red]")
            raise

    def display_welcome(self):
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

    def display_scene(self, content: str, choices: list[str], characters: Optional[list] = None):
        """
        Display scene content and choices.

        Args:
            content: Scene content
            choices: List of 3 choices
            characters: Optional list of character dicts present in scene
        """
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

    async def start_new_story(self):
        """Start a new interactive story session."""
        self.console.print("[bold cyan]Starting a new story[/bold cyan]")
        self.console.print()

        # Get story title
        title = self.prompt_ui.ask("Enter a title for your story", default="My Adventure")

        # Get story setting
        self.console.print()
        setting = self.prompt_ui.ask(
            "Describe the story setting",
            default="A remote Southeast Asian village with a beautiful beach",
        )

        # Get optional instructions
        self.console.print()
        self.console.print(
            "[dim]You can add optional instructions for the story (or press Enter to skip)[/dim]"
        )
        instructions = self.prompt_ui.ask("Instructions", default="")

        # Ask about internal thoughts
        self.console.print()
        thoughts_response = self.prompt_ui.ask(
            "Show NPC internal thoughts? (adds narrative depth)",
            choices=["yes", "no"],
            default="no",
        )
        self.show_internal_thoughts = thoughts_response.lower() == "yes"
        
        # Update agent factory
        if self.generator.agent_factory:
            self.generator.agent_factory.set_show_internal_thoughts(self.show_internal_thoughts)
        
        status_msg = "[green]enabled[/green]" if self.show_internal_thoughts else "[dim]disabled[/dim]"
        self.console.print(f"NPC internal thoughts: {status_msg}")
        self.console.print()

        # Create story
        from src.story.state import StoryStateManager

        state_manager = StoryStateManager(self.generator.session_factory)
        state = await state_manager.create_story(title=title)

        self.current_story_id = state.story_id

        self.console.print()
        self.console.print(f"[green]Story '{title}' created![/green]")
        self.console.print("[dim]Generating initial scene...[/dim]")
        self.console.print()

        # Generate initial scene
        try:
            scene = await self.generator.generate_initial_scene(
                story_id=state.story_id,
                story_setting=setting,
                user_instructions=instructions if instructions else None,
            )

            # Get characters in this scene
            characters = await self.get_scene_characters(scene.id)

            # Display scene
            self.display_scene(scene.content, scene.choices, characters)

            # Interactive loop
            await self.story_loop()

        except LivingWorldError as e:
            self.console.print(f"[red]Error: {e}[/red]")

    async def list_stories(self):
        """List all available stories."""
        try:
            state_manager = StoryStateManager(self.generator.session_factory)
            stories = await state_manager.list_stories(active_only=False)

            if not stories:
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
            self.console.print(f"[red]Error listing stories: {e}[/red]")

    async def load_story(self, story_id: int):
        """Load an existing story and continue playing."""
        try:
            state_manager = StoryStateManager(self.generator.session_factory)
            state = await state_manager.load_story(story_id)

            self.current_story_id = state.story_id

            self.console.print(f"[green]Loaded story: {state.title}[/green]")
            self.console.print(f"[dim]Scene {state.scene_number}[/dim]")
            self.console.print()

            # Get last scene
            from src.database.models import Scene
            from sqlalchemy import select

            async with self.generator.session_factory() as session:
                result = await session.execute(
                    select(Scene).where(Scene.id == state.current_scene_id)
                )
                last_scene = result.scalar_one_or_none()

                if last_scene:
                    # Display last scene
                    from src.database.models import Choice

                    result = await session.execute(
                        select(Choice)
                        .where(Choice.scene_id == last_scene.id)
                        .order_by(Choice.choice_number)
                    )
                    choices = result.scalars().all()

                    # Get characters in this scene
                    characters = await self.get_scene_characters(last_scene.id)

                    self.display_scene(last_scene.content, [c.content for c in choices], characters)

                    # Ask if user wants to continue
                    response = self.prompt_ui.ask(
                        "\nContinue from this scene? (y/n)",
                        default="y",
                    )

                    if response.lower() != "y":
                        return

            # Enter story loop
            await self.story_loop()

        except Exception as e:
            self.console.print(f"[red]Error loading story: {e}[/red]")

    async def export_story(self, story_id: Optional[int] = None):
        """Export a story to file."""
        try:
            if story_id is None:
                # List stories and ask which to export
                await self.list_stories()
                story_id_str = self.prompt_ui.ask("Enter story ID to export", default="")

                if not story_id_str:
                    return

                story_id = int(story_id_str)

            # Ask for format and filename
            format_choice = self.prompt_ui.ask(
                "Export format",
                choices=["json", "markdown"],
                default="json",
            )

            default_filename = f"story_{story_id}.{format_choice if format_choice == 'markdown' else 'md'}"
            output_path = self.prompt_ui.ask(
                "Output filename",
                default=default_filename,
            )

            # Export
            exporter = StoryExporter()
            async with self.generator.session_factory() as session:
                if format_choice == "json":
                    path = await exporter.export_to_json(
                        session,
                        story_id,
                        Path(output_path),
                    )
                else:
                    path = await exporter.export_to_markdown(
                        session,
                        story_id,
                        Path(output_path),
                    )

            self.console.print(f"[green]Story exported to: {path}[/green]")

        except Exception as e:
            self.console.print(f"[red]Error exporting story: {e}[/red]")

    async def import_story(self):
        """Import a story from file."""
        try:
            # Ask for file path
            file_path = self.prompt_ui.ask(
                "Path to file to import",
                default="",
            )

            if not file_path:
                return

            file_path = Path(file_path)
            if not file_path.exists():
                self.console.print(f"[red]File not found: {file_path}[/red]")
                return

            # Import based on file type
            importer = StoryImporter()
            async with self.generator.session_factory() as session:
                if file_path.suffix == ".json":
                    story_id = await importer.import_from_json(
                        session,
                        file_path,
                        create_new_story=True,
                    )
                elif file_path.suffix in [".txt", ".md"]:
                    title = self.prompt_ui.ask("Story title", default="Imported Story")
                    content = file_path.read_text(encoding="utf-8")
                    story_id = await importer.import_from_text(
                        session,
                        title,
                        content,
                    )
                else:
                    self.console.print(
                        f"[red]Unsupported file format: {file_path.suffix}[/red]"
                    )
                    return

            self.console.print(f"[green]Story imported with ID: {story_id}[/green]")

        except Exception as e:
            self.console.print(f"[red]Error importing story: {e}[/red]")

    async def show_settings(self):
        """Show and modify story settings."""
        self.console.print("\n[bold cyan]Settings[/bold cyan]\n")
        
        # Show current setting
        status = "[green]ON[/green]" if self.show_internal_thoughts else "[dim]OFF[/dim]"
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
            self.show_internal_thoughts = new_value
            
            # Update agent factory
            if self.generator and self.generator.agent_factory:
                self.generator.agent_factory.set_show_internal_thoughts(new_value)
            
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
                self.console.print(f"[yellow]Warning: Could not load characters: {e}[/yellow]")
                return []

    async def story_loop(self):
        """Main interactive story loop."""
        while True:
            try:
                # Get user input
                user_input = self.prompt_ui.ask(
                    "\n[yellow]→[/yellow] ",
                    console=self.console,
                )

                # Check for special commands
                if user_input.lower() in ("quit", "exit", "q"):
                    self.console.print("[dim]Thanks for playing![/dim]")
                    break
                
                if user_input.lower() in ("thoughts on", "thoughtson"):
                    self.show_internal_thoughts = True
                    if self.generator and self.generator.agent_factory:
                        self.generator.agent_factory.set_show_internal_thoughts(True)
                    self.console.print("[green italic]NPC internal thoughts now visible[/green italic]")
                    continue
                
                if user_input.lower() in ("thoughts off", "thoughtsoff"):
                    self.show_internal_thoughts = False
                    if self.generator and self.generator.agent_factory:
                        self.generator.agent_factory.set_show_internal_thoughts(False)
                    self.console.print("[dim italic]NPC internal thoughts now hidden[/dim italic]")
                    continue

                # Parse input
                choice, instruction = self.parse_input(user_input)

                if choice is None or choice < 1 or choice > 3:
                    self.console.print(
                        "[red]Please enter a choice between 1 and 3[/red]"
                    )
                    continue

                # Generate next scene
                self.console.print()
                self.console.print("[dim]Generating next scene...[/dim]")
                self.console.print()

                scene = await self.generator.generate_next_scene(
                    story_id=self.current_story_id,
                    choice=choice,
                    user_instruction=instruction,
                )

                # Get characters in this scene
                characters = await self.get_scene_characters(scene.id)

                # Display scene
                self.display_scene(scene.content, scene.choices, characters)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Story interrupted.[/dim]")
                break
            except LivingWorldError as e:
                self.console.print(f"[red]Error: {e}[/red]")
                continue

    async def run(self):
        """Run the CLI application with main menu."""
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
                        await self.load_story(int(story_id_str))
                elif choice == "4":
                    await self.show_settings()
                elif choice == "5":
                    await self.export_story()
                elif choice == "6":
                    await self.import_story()
                elif choice == "7":
                    self.console.print("[dim]Goodbye![/dim]")
                    break

        except KeyboardInterrupt:
            self.console.print("\n[dim]Goodbye![/dim]")
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {e}[/red]")
            sys.exit(1)
        finally:
            # Cleanup
            await close_database()


async def main_async(config: AppConfig):
    """
    Main async entry point.

    Args:
        config: Application configuration
    """
    cli = StoryCLI(config)
    await cli.run()


def main():
    """Main entry point."""
    from src.core.config import get_config

    # Load configuration
    config = get_config()

    # Run async main
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
