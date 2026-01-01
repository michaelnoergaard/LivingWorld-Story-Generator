"""Character agent implementation using LangChain."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

# Try to import LangChain agent components (may not be available in all versions)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    LANGCHAIN_AGENTS_AVAILABLE = True
except ImportError:
    # Fallback for newer LangChain versions
    try:
        from langchain_core.agents import AgentExecutor, create_react_agent
        LANGCHAIN_AGENTS_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AGENTS_AVAILABLE = False
        AgentExecutor = None
        create_react_agent = None

from src.core.exceptions import AgentError
from src.core.validation import (
    validate_id,
    validate_string,
    validate_choice,
    validate_content,
    ValidationError,
)
from src.database.models import Character, CharacterRelationship
from src.llm.prompt_builder import PromptBuilder
from src.llm.ollama_client import OllamaClient
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.search import SemanticSearch
from src.agents.agent_tools import CharacterAgentTools
from src.core.constants import (
    LLMConstants,
    AgentConstants,
    SentimentWords,
    StoryConstants,
)

logger = logging.getLogger(__name__)


class CharacterAgent:
    """
    LangChain-based agent for autonomous NPC characters.

    Each character has their own agent with unique personality,
    goals, memories, and ability to generate in-character responses.
    """

    def __init__(
        self,
        character: Character,
        ollama_client: OllamaClient,
        encoder: EmbeddingEncoder,
        semantic_search: SemanticSearch,
        prompt_builder: PromptBuilder,
        show_internal_thoughts: bool = False,
    ) -> None:
        """
        Initialize character agent.

        Args:
            character: Character database model
            ollama_client: Ollama API client
            encoder: Embedding encoder
            semantic_search: Semantic search instance
            prompt_builder: Prompt builder
            show_internal_thoughts: Whether to include internal thoughts in responses
        """
        self.character = character
        self.ollama_client = ollama_client
        self.encoder = encoder
        self.semantic_search = semantic_search
        self.prompt_builder = prompt_builder
        self.show_internal_thoughts = show_internal_thoughts

        # Build system prompt for this character
        self.system_prompt = self._build_character_prompt()

        # Tools will be initialized per-session
        self.tools: Optional[CharacterAgentTools] = None

        # LangChain agent components (initialized when needed)
        self._langchain_agent = None
        self._agent_executor = None

    def _build_character_prompt(self) -> str:
        """
        Build system prompt for this character.

        Returns:
            Character-specific system prompt
        """
        personality = self.character.personality or "A friendly person"
        goals = self.character.goals or "To interact with others"
        background = self.character.background or "Living in this world"

        # Get current emotional state
        emotional_context = ""
        if self.character.emotional_state:
            emotional_context = f"\n\n## Current Emotional State\n{json.dumps(self.character.emotional_state, indent=2)}"

        if self.character.current_mood:
            emotional_context += f"\n\n## Current Mood\n{self.character.current_mood}"

        base_prompt = self.prompt_builder.build_character_system_prompt(
            character_name=self.character.name,
            personality=personality,
            goals=goals,
            background=background,
        )

        return base_prompt + emotional_context

    async def initialize_session(
        self,
        session: AsyncSession,
        story_id: int,
    ) -> None:
        """
        Initialize agent session with tools.

        Args:
            session: Database session
            story_id: Story ID
        """
        # Validate parameters
        validated_story_id = validate_id(story_id, field_name="story_id")
        validated_character_id = validate_id(self.character.id, field_name="character_id")

        self.tools = CharacterAgentTools(
            session=session,
            character_id=validated_character_id,
            story_id=validated_story_id,
            semantic_search=self.semantic_search,
        )

    async def respond_to(
        self,
        context: str,
        scene_content: str,
        other_characters: Optional[List[str]] = None,
        use_agent: bool = False,
    ) -> str:
        """
        Generate character's response based on context.

        Args:
            context: Current situation/context
            scene_content: Current scene content
            other_characters: Optional list of other character names present
            use_agent: Whether to use LangChain Agent (with tools) or direct generation

        Returns:
            Character's response/dialogue

        Raises:
            AgentError: If response generation fails
        """
        # Validate parameters
        validated_context = validate_string(
            context,
            field_name="context",
            min_length=1,
            max_length=2000,
            strip_whitespace=True,
        )
        validated_scene_content = validate_content(
            scene_content,
            field_name="scene_content",
        )

        if other_characters:
            other_characters = validate_list(
                other_characters,
                field_name="other_characters",
                min_length=1,
                max_length=10,
                item_validator=lambda x: validate_string(
                    x, field_name="character_name", min_length=1, max_length=100
                )
            )

        if self.tools is None:
            raise AgentError(
                "session initialization",
                error_details="Agent session not initialized. Call initialize_session() first."
            )

        try:
            if use_agent:
                # Use LangChain Agent with tool capabilities
                return await self._agent_respond(context, scene_content, other_characters)
            else:
                # Use direct generation (faster, simpler)
                return await self._direct_respond(context, scene_content, other_characters)

        except Exception as e:
            raise AgentError("generating response", error_details=str(e)) from e

    async def _direct_respond(
        self,
        context: str,
        scene_content: str,
        other_characters: Optional[List[str]] = None,
    ) -> str:
        """
        Direct response generation without LangChain Agent.

        Args:
            context: Current situation/context
            scene_content: Current scene content
            other_characters: Optional list of other character names present

        Returns:
            Character's response/dialogue (possibly with internal thought)
        """
        # Use validated parameters from respond_to
        memory_query = f"{context} {scene_content}"

        # Prioritize emotionally charged memories
        memories_result = await self.tools.query_memories(
            query=memory_query,
            limit=AgentConstants.DEFAULT_MEMORY_QUERY_LIMIT,
        )

        # Check relationships with other characters
        relationship_info = ""
        if other_characters:
            for char_name in other_characters:
                rel_info = await self.tools.get_relationships(char_name)
                if "not found" not in rel_info.lower() and "not established" not in rel_info.lower():
                    relationship_info += f"\n{rel_info}\n"

        # Build prompt
        prompt_parts = [
            f"## Current Situation\n{context}\n",
            f"## Scene\n{scene_content[:StoryConstants.SCENE_CONTENT_PREVIEW_LENGTH]}\n",
        ]

        if other_characters:
            prompt_parts.append(f"## Others Present\n{', '.join(other_characters)}\n")

        if relationship_info:
            prompt_parts.append(f"## Your Relationships\n{relationship_info}\n")

        if memories_result and "No relevant memories" not in memories_result:
            prompt_parts.append(f"## Your Memories\n{memories_result}\n")

        # Add emotional state context
        if self.character.emotional_state:
            prompt_parts.append(f"## Your Emotional State\n{json.dumps(self.character.emotional_state)}\n")

        prompt_parts.append(
            "\n## Your Response\n"
            "Respond in character. Show your personality through your words. "
            "Consider your goals and how you would react to this situation. "
            "Your response should be what you say or do next."
        )

        prompt = "".join(prompt_parts)

        # Generate response
        response = await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

        # Detect and store emotional valence
        emotional_valence = await self._detect_emotional_valence(response)

        # Generate internal thought
        internal_thought = None
        if self.show_internal_thoughts:
            internal_thought = await self.generate_internal_thought(
                what_was_said=response,
                situation=context,
                other_characters_present=other_characters,
            )
            # Store the thought as a memory
            if internal_thought:
                await self.tools.store_memory(
                    content=f"Internal thought: {internal_thought}",
                    memory_type="internal_thought",
                    emotional_valence=emotional_valence,
                    importance=AgentConstants.INTERNAL_THOUGHT_IMPORTANCE,
                )

        # Store this interaction as a memory
        await self.tools.store_memory(
            content=f"Responded to: {context[:StoryConstants.MEMORY_TRUNCATION_LENGTH]}... | Said: {response[:StoryConstants.MEMORY_TRUNCATION_LENGTH]}...",
            memory_type="conversation",
            emotional_valence=emotional_valence,
            importance=AgentConstants.DEFAULT_MEMORY_IMPORTANCE,
        )

        # Update emotional state
        await self._update_emotional_state(emotional_valence)

        # Format response with internal thought if present
        if internal_thought:
            return f'{response}\n\n*Internal thought: {internal_thought}*'

        return response

    async def _agent_respond(
        self,
        context: str,
        scene_content: str,
        other_characters: Optional[List[str]] = None,
    ) -> str:
        """
        Response generation using LangChain Agent with tools.

        Args:
            context: Current situation/context
            scene_content: Current scene content
            other_characters: Optional list of other character names present

        Returns:
            Character's response/dialogue
        """
        # Check if LangChain agents are available
        if not LANGCHAIN_AGENTS_AVAILABLE:
            # Fall back to direct mode
            return await self._direct_respond(context, scene_content, other_characters)

        # Initialize LangChain Agent if needed
        if self._agent_executor is None:
            await self._initialize_langchain_agent()

        # Build agent input
        others_section = "## Others Present\n" + ", ".join(other_characters) + "\n" if other_characters else ""

        agent_input = f"""
## Current Situation
{context}

## Scene
{scene_content[:500]}

{others_section}
## Your Task
Respond in character to the situation. You may use your available tools to:
- Query your memories for relevant past experiences
- Store new memories about this interaction
- Check your relationships with others
- Recall past conversations

Your response should be what you say or do next, staying true to your personality and goals.
"""

        # Run agent
        try:
            result = await self._run_agent_async(agent_input)
            
            # Generate internal thought if enabled
            internal_thought = None
            if self.show_internal_thoughts:
                internal_thought = await self.generate_internal_thought(
                    what_was_said=result,
                    situation=context,
                    other_characters_present=other_characters,
                )
                if internal_thought:
                    await self.tools.store_memory(
                        content=f"Internal thought: {internal_thought}",
                        memory_type="internal_thought",
                        emotional_valence=await self._detect_emotional_valence(result),
                        importance=0.6,
                    )
                    return f'{result}\n\n*Internal thought: {internal_thought}*'
            
            return result
        except Exception as e:
            # Fall back to direct response if agent fails
            return await self._direct_respond(context, scene_content, other_characters)

    async def _initialize_langchain_agent(self) -> None:
        """Initialize the LangChain Agent with tools."""
        try:
            # Get LangChain tools
            tools = self.tools.get_langchain_tools()

            # Create ChatOllama instance
            llm = ChatOllama(
                model=self.ollama_client.config.model,
                temperature=self._get_temperature(),
                base_url=self.ollama_client.config.base_url,
            )

            # Create agent prompt
            agent_prompt = f"""You are {self.character.name}.

{self.system_prompt}

You have access to the following tools:
{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}"""

            # Create ReAct agent
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate.from_template(agent_prompt)

            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt,
            )

            # Create executor
            self._agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=LLMConstants.AGENT_MAX_ITERATIONS,
            )

        except Exception as e:
            raise AgentError("initializing LangChain agent", error_details=str(e)) from e

    async def _run_agent_async(self, input_text: str) -> str:
        """
        Run agent asynchronously.

        Args:
            input_text: Agent input

        Returns:
            Agent output
        """
        loop = asyncio.get_running_loop()

        try:
            # Run executor in thread pool
            result = await loop.run_in_executor(
                None,
                lambda: self._agent_executor.invoke({"input": input_text})
            )

            # Extract output
            output = result.get("output", "")
            return output

        except Exception as e:
            raise AgentError("agent execution", error_details=str(e)) from e

    def _get_temperature(self) -> float:
        """Get temperature from agent config."""
        config = self.character.agent_config or {}
        return config.get("temperature", LLMConstants.CHARACTER_TEMPERATURE)

    async def _detect_emotional_valence(self, text: str) -> float:
        """
        Detect emotional valence of text using simple heuristics.

        Args:
            text: Text to analyze

        Returns:
            Emotional valence score (-1.0 to 1.0)
        """
        text_lower = text.lower()

        positive_count = sum(1 for word in SentimentWords.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in SentimentWords.NEGATIVE_WORDS if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return AgentConstants.DEFAULT_EMOTIONAL_VALENCE

        # Calculate valence (-1.0 to 1.0)
        valence = (positive_count - negative_count) / max(total, 1)
        return max(-1.0, min(1.0, valence))

    async def _update_emotional_state(self, valence: float) -> None:
        """
        Update character's emotional state based on valence.

        Args:
            valence: Emotional valence score
        """
        # Initialize emotional state if not present
        if not self.character.emotional_state:
            self.character.emotional_state = {
                "arousal": AgentConstants.INITIAL_EMOTIONAL_AROUSAL,
                "valence": AgentConstants.INITIAL_EMOTIONAL_VALENCE,
                "dominance": AgentConstants.INITIAL_EMOTIONAL_DOMINANCE
            }

        # Update valence with exponential moving average
        current_valence = self.character.emotional_state.get("valence", AgentConstants.DEFAULT_EMOTIONAL_VALENCE)
        ema_coeff = AgentConstants.EMOTIONAL_STATE_EMA_COEFFICIENT
        new_valence = ema_coeff * current_valence + (1 - ema_coeff) * valence
        self.character.emotional_state["valence"] = new_valence

        # Update mood based on valence
        if new_valence > AgentConstants.POSITIVE_VALENCE_THRESHOLD:
            self.character.current_mood = "positive"
        elif new_valence < AgentConstants.NEGATIVE_VALENCE_THRESHOLD:
            self.character.current_mood = "negative"
        else:
            self.character.current_mood = "neutral"

    async def generate_internal_thought(
        self,
        what_was_said: str,
        situation: str,
        other_characters_present: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Generate character's internal thought about a response or action.

        Args:
            what_was_said: What the character said outwardly
            situation: Current situation/context
            other_characters_present: Optional list of other character names present

        Returns:
            Internal thought string, or None if generation fails
        """
        try:
            personality = self.character.personality or "A friendly person"
            goals = self.character.goals or "To interact with others"
            background = self.character.background or "Living in this world"

            # Build thought generation prompt
            prompt = self.prompt_builder.build_internal_thought_prompt(
                character_name=self.character.name,
                personality=personality,
                goals=goals,
                background=background,
                what_was_said=what_was_said,
                situation=situation,
                emotional_state=self.character.emotional_state,
                other_characters_present=other_characters_present,
            )

            # Generate thought with lower temperature for more focused results
            thought = await self.ollama_client.generate_with_retry(
                prompt=prompt,
                system_prompt="",  # No additional system prompt needed
                temperature=LLMConstants.THOUGHT_TEMPERATURE,
            )

            # Clean up the thought
            thought = thought.strip()
            
            # Remove any quotes if the entire thought is wrapped in them
            if thought.startswith('"') and thought.endswith('"'):
                thought = thought[1:-1]
            elif thought.startswith("'") and thought.endswith("'"):
                thought = thought[1:-1]
            
            return thought

        except Exception as e:
            logger.warning("Failed to generate internal thought: %s", e)
            return None

    async def observe_and_react(
        self,
        event: str,
        emotional_valence: float = 0.0,
    ) -> str:
        """
        Observe an event and react to it.

        Args:
            event: Event description
            emotional_valence: Emotional impact (-1.0 to 1.0)

        Returns:
            Character's reaction (possibly with internal thought)
        """
        # Validate parameters
        validated_event = validate_string(
            event,
            field_name="event",
            min_length=1,
            max_length=2000,
            strip_whitespace=True,
        )

        if not isinstance(emotional_valence, (int, float)) or not -1.0 <= emotional_valence <= 1.0:
            raise ValidationError(
                field="emotional_valence",
                value=emotional_valence,
                message="Emotional valence must be a number between -1.0 and 1.0",
                constraint="float between -1.0 and 1.0"
            )

        if self.tools is None:
            raise AgentError(
                "session initialization",
                error_details="Agent session not initialized. Call initialize_session() first."
            )

        # Store observation as memory
        await self.tools.store_memory(
            content=f"Observed: {validated_event}",
            memory_type="observation",
            emotional_valence=emotional_valence,
            importance=AgentConstants.DEFAULT_MEMORY_IMPORTANCE,
        )

        # Update emotional state
        await self._update_emotional_state(emotional_valence)

        # Generate reaction
        prompt = f"Something just happened: {validated_event}\n\nHow do you react? Express your response naturally."

        response = await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

        # Generate internal thought if enabled
        internal_thought = None
        if self.show_internal_thoughts:
            internal_thought = await self._generate_observation_thought(event)
            if internal_thought:
                await self.tools.store_memory(
                    content=f"Internal thought about observation: {internal_thought}",
                    memory_type="internal_thought",
                    emotional_valence=emotional_valence,
                    importance=AgentConstants.INTERNAL_THOUGHT_IMPORTANCE,
                )
                return f'{response}\n\n*Internal thought: {internal_thought}*'

        return response

    async def _generate_observation_thought(self, event_observed: str) -> Optional[str]:
        """
        Generate internal thought about an observed event.

        Args:
            event_observed: What was observed

        Returns:
            Internal thought string, or None if generation fails
        """
        try:
            personality = self.character.personality or "A friendly person"
            goals = self.character.goals or "To interact with others"

            prompt = self.prompt_builder.build_observation_thought_prompt(
                character_name=self.character.name,
                personality=personality,
                goals=goals,
                event_observed=event_observed,
                emotional_state=self.character.emotional_state,
            )

            thought = await self.ollama_client.generate_with_retry(
                prompt=prompt,
                system_prompt="",
                temperature=LLMConstants.THOUGHT_TEMPERATURE,
            )

            return thought.strip()

        except Exception as e:
            logger.warning("Failed to generate observation thought: %s", e)
            return None

    async def decide_action(
        self,
        situation: str,
        available_actions: List[str],
    ) -> str:
        """
        Decide on an action based on personality and goals.

        Args:
            situation: Current situation
            available_actions: List of possible actions

        Returns:
            Chosen action and reasoning
        """
        if self.tools is None:
            raise AgentError(
                "session initialization",
                error_details="Agent session not initialized. Call initialize_session() first."
            )

        actions_text = "\n".join(f"{i+1}. {action}" for i, action in enumerate(available_actions))

        prompt = (
            f"## Situation\n{situation}\n\n"
            f"## Available Actions\n{actions_text}\n\n"
            "Which action would you take? Consider your personality and goals. "
            "Explain your reasoning and state your choice."
        )

        return await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

    async def autonomous_action(
        self,
        situation: str,
        other_characters_present: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Decide and take an autonomous action based on goals and personality.

        Args:
            situation: Current situation description
            other_characters_present: IDs of other characters present

        Returns:
            Dictionary with action type, content, internal thought, and any targets
        """
        if self.tools is None:
            raise AgentError(
                "session initialization",
                error_details="Agent session not initialized. Call initialize_session() first."
            )

        # Retrieve relevant memories
        memories = await self.tools.query_memories(
            query=f"goals personality {situation}",
            limit=AgentConstants.AUTONOMOUS_MEMORY_LIMIT,
        )

        # Check relationships with present characters
        relationship_context = ""
        if other_characters_present:
            for char_id in other_characters_present:
                if char_id != self.character.id:
                    # Get relationship
                    await self.tools.get_relationships()

        # Build decision prompt
        memories_section = f"## Your Relevant Memories\n{memories}\n" if "No relevant memories" not in memories else ""

        prompt = f"""
## Current Situation
{situation}

{memories_section}
## Your Task
Based on your personality, goals, and current emotional state, decide what you want to do next.

Consider:
1. Your goals and motivations
2. Your personality traits
3. Your relationships with others present
4. Your current mood

Describe your intended action clearly and specifically.
"""

        # Generate action decision
        response = await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

        # Generate internal thought if enabled
        internal_thought = None
        if self.show_internal_thoughts:
            internal_thought = await self._generate_action_thought(
                situation=situation,
                intended_action=response,
            )
            if internal_thought:
                await self.tools.store_memory(
                    content=f"Internal thought about action: {internal_thought}",
                    memory_type="internal_thought",
                    emotional_valence=self.character.emotional_state.get("valence", 0.0),
                    importance=0.6,
                )

        # Store the action as a memory
        await self.tools.store_memory(
            content=f"Decided to: {response[:StoryConstants.MEMORY_TRUNCATION_LENGTH]}...",
            memory_type="action_taken",
            emotional_valence=self.character.emotional_state.get("valence", AgentConstants.DEFAULT_EMOTIONAL_VALENCE),
            importance=AgentConstants.ACTION_MEMORY_IMPORTANCE,
        )

        result = {
            "character_id": self.character.id,
            "character_name": self.character.name,
            "action": response,
            "emotional_state": self.character.emotional_state,
        }

        if internal_thought:
            result["internal_thought"] = internal_thought

        return result

    async def _generate_action_thought(
        self,
        situation: str,
        intended_action: str,
    ) -> Optional[str]:
        """
        Generate internal thought about an intended action.

        Args:
            situation: Current situation
            intended_action: What the character intends to do

        Returns:
            Internal thought string, or None if generation fails
        """
        try:
            personality = self.character.personality or "A friendly person"
            goals = self.character.goals or "To interact with others"

            prompt = self.prompt_builder.build_autonomous_action_thought_prompt(
                character_name=self.character.name,
                personality=personality,
                goals=goals,
                situation=situation,
                intended_action=intended_action,
                emotional_state=self.character.emotional_state,
            )

            thought = await self.ollama_client.generate_with_retry(
                prompt=prompt,
                system_prompt="",
                temperature=LLMConstants.THOUGHT_TEMPERATURE,
            )

            return thought.strip()

        except Exception as e:
            logger.warning("Failed to generate action thought: %s", e)
            return None

    async def interact_with_character(
        self,
        other_character_id: int,
        interaction_content: str,
        interaction_type: str = "conversation",
    ) -> str:
        """
        Interact with another character and update relationship.

        Args:
            other_character_id: ID of character to interact with
            interaction_content: What was said/done
            interaction_type: Type of interaction

        Returns:
            Response/Reaction
        """
        # Validate parameters
        validated_other_character_id = validate_id(
            other_character_id,
            field_name="other_character_id",
            min_value=1
        )
        validated_interaction_content = validate_string(
            interaction_content,
            field_name="interaction_content",
            min_length=1,
            max_length=2000,
            strip_whitespace=True,
        )
        validated_interaction_type = validate_string(
            interaction_type,
            field_name="interaction_type",
            min_length=1,
            max_length=50,
            allowed_chars=(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                "_-"
            )
        )

        if self.tools is None:
            raise AgentError(
                "session initialization",
                error_details="Agent session not initialized. Call initialize_session() first."
            )

        # Calculate sentiment delta based on interaction content
        sentiment_delta = await self._detect_emotional_valence(interaction_content)

        # Update relationship
        await self.tools.update_relationship(
            other_character_id=validated_other_character_id,
            sentiment_delta=sentiment_delta * AgentConstants.RELATIONMENT_SENTIMENT_MULTIPLIER,
            trust_delta=AgentConstants.TRUST_DELTA_POSITIVE if sentiment_delta > 0 else AgentConstants.TRUST_DELTA_NEGATIVE,
            interaction_type=validated_interaction_type,
        )

        # Store interaction memory
        await self.tools.store_memory(
            content=f"Interacted with character {validated_other_character_id}: {validated_interaction_content[:StoryConstants.MEMORY_TRUNCATION_LENGTH]}...",
            memory_type="conversation",
            emotional_valence=sentiment_delta,
            importance=AgentConstants.INTERNAL_THOUGHT_IMPORTANCE,
        )

        # Generate response
        prompt = f"""You just interacted with someone: {validated_interaction_content}

How do you respond? Consider the relationship and your personality."""

        response = await self.ollama_client.generate_with_retry(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self._get_temperature(),
        )

        # Generate internal thought if enabled
        internal_thought = None
        if self.show_internal_thoughts:
            internal_thought = await self.generate_internal_thought(
                what_was_said=response,
                situation=f"Interaction with character {validated_other_character_id}: {validated_interaction_content[:100]}",
            )
            if internal_thought:
                await self.tools.store_memory(
                    content=f"Internal thought about interaction: {internal_thought}",
                    memory_type="internal_thought",
                    emotional_valence=sentiment_delta,
                    importance=AgentConstants.INTERNAL_THOUGHT_IMPORTANCE,
                )
                return f'{response}\n\n*Internal thought: {internal_thought}*'

        return response

    def set_show_internal_thoughts(self, show: bool) -> None:
        """
        Set whether to show internal thoughts in responses.

        Args:
            show: Whether to show internal thoughts
        """
        self.show_internal_thoughts = show

    def get_summary(self) -> Dict[str, Any]:
        """
        Get character summary.

        Returns:
            Dictionary with character info
        """
        return {
            "id": self.character.id,
            "name": self.character.name,
            "description": self.character.description,
            "personality": self.character.personality,
            "goals": self.character.goals,
            "background": self.character.background,
            "agent_config": self.character.agent_config,
            "emotional_state": self.character.emotional_state,
            "current_mood": self.character.current_mood,
            "show_internal_thoughts": self.show_internal_thoughts,
        }
