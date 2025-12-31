"""Tests for character agent and agent tools modules."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.character_agent import CharacterAgent
from src.agents.agent_tools import CharacterAgentTools
from src.core.exceptions import AgentError
from src.database.models import Character, CharacterMemory
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.search import SemanticSearch
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_builder import PromptBuilder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_character():
    """Create mock character."""
    char = MagicMock()
    char.id = 1
    char.name = "Sreykeo"
    char.description = "A young Cambodian village girl"
    char.personality = "Curious, friendly, and brave"
    char.goals = "Learn about the world"
    char.background = "Born in a small village"
    char.agent_config = {"temperature": 0.8}
    return char


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client."""
    client = MagicMock()
    client.generate_with_retry = AsyncMock(return_value="I am ready to help!")
    return client


@pytest.fixture
def mock_encoder():
    """Mock embedding encoder."""
    encoder = MagicMock()
    encoder.encode_async = AsyncMock(return_value=[0.1] * 384)
    return encoder


@pytest.fixture
def mock_semantic_search():
    """Mock semantic search."""
    search = MagicMock()
    search.retrieve_character_memories = AsyncMock(return_value=[])
    return search


@pytest.fixture
def mock_prompt_builder():
    """Mock prompt builder."""
    builder = MagicMock()
    builder.build_character_system_prompt = MagicMock(
        return_value="You are Sreykeo, a curious village girl."
    )
    return builder


@pytest.fixture
def character_agent(
    mock_character,
    mock_ollama_client,
    mock_encoder,
    mock_semantic_search,
    mock_prompt_builder,
):
    """Create character agent instance."""
    return CharacterAgent(
        character=mock_character,
        ollama_client=mock_ollama_client,
        encoder=mock_encoder,
        semantic_search=mock_semantic_search,
        prompt_builder=mock_prompt_builder,
    )


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = MagicMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def agent_tools(mock_session, mock_semantic_search):
    """Create agent tools instance."""
    return CharacterAgentTools(
        session=mock_session,
        character_id=1,
        story_id=1,
        semantic_search=mock_semantic_search,
    )


# =============================================================================
# CharacterAgent Tests
# =============================================================================

class TestCharacterAgentInit:
    """Tests for CharacterAgent initialization."""

    def test_init(self, character_agent, mock_character):
        """Test agent initialization."""
        assert character_agent.character is mock_character
        assert character_agent.tools is None
        assert character_agent.system_prompt == "You are Sreykeo, a curious village girl."

    def test_build_character_prompt(self, character_agent, mock_prompt_builder):
        """Test building character-specific prompt."""
        mock_prompt_builder.build_character_system_prompt.assert_called_once_with(
            character_name="Sreykeo",
            personality="Curious, friendly, and brave",
            goals="Learn about the world",
            background="Born in a small village",
        )

    def test_init_with_none_personality(self, mock_character, mock_ollama_client, mock_encoder, mock_semantic_search, mock_prompt_builder):
        """Test agent initialization with None personality."""
        mock_character.personality = None
        mock_character.goals = None
        mock_character.background = None

        agent = CharacterAgent(
            character=mock_character,
            ollama_client=mock_ollama_client,
            encoder=mock_encoder,
            semantic_search=mock_semantic_search,
            prompt_builder=mock_prompt_builder,
        )

        # Should use default values
        assert agent.system_prompt is not None

    def test_get_temperature(self, character_agent):
        """Test getting temperature from config."""
        temp = character_agent._get_temperature()
        assert temp == 0.8

    def test_get_temperature_default(self, mock_character, mock_ollama_client, mock_encoder, mock_semantic_search, mock_prompt_builder):
        """Test getting default temperature when not in config."""
        mock_character.agent_config = {}

        agent = CharacterAgent(
            character=mock_character,
            ollama_client=mock_ollama_client,
            encoder=mock_encoder,
            semantic_search=mock_semantic_search,
            prompt_builder=mock_prompt_builder,
        )

        temp = agent._get_temperature()
        assert temp == 0.8  # Default value


@pytest.mark.asyncio
class TestCharacterAgentInitializeSession:
    """Tests for initialize_session method."""

    async def test_initialize_session(self, character_agent, mock_session):
        """Test initializing agent session."""
        await character_agent.initialize_session(mock_session, story_id=1)

        assert character_agent.tools is not None
        assert isinstance(character_agent.tools, CharacterAgentTools)
        assert character_agent.tools.session is mock_session
        assert character_agent.tools.story_id == 1


@pytest.mark.asyncio
class TestCharacterAgentRespondTo:
    """Tests for respond_to method."""

    async def test_respond_to_success(self, character_agent, mock_session):
        """Test successful response generation."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.query_memories = AsyncMock(
            return_value="No relevant memories found."
        )
        character_agent.tools.store_memory = AsyncMock(return_value="Memory stored.")

        response = await character_agent.respond_to(
            context="The party approaches the village.",
            scene_content="A group of travelers arrives at the gate.",
        )

        assert isinstance(response, str)
        assert len(response) > 0

    async def test_respond_to_with_other_characters(self, character_agent, mock_session):
        """Test response with other characters present."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.query_memories = AsyncMock(return_value="")
        character_agent.tools.store_memory = AsyncMock(return_value="Stored.")

        response = await character_agent.respond_to(
            context="At the market",
            scene_content="People shopping for goods.",
            other_characters=["Alice", "Bob"],
        )

        assert isinstance(response, str)

    async def test_respond_to_with_memories(self, character_agent, mock_session):
        """Test response using retrieved memories."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.query_memories = AsyncMock(
            return_value="Relevant memories:\n- Met a traveler yesterday (relevance: 0.85)"
        )
        character_agent.tools.store_memory = AsyncMock(return_value="Stored.")

        response = await character_agent.respond_to(
            context="A traveler approaches",
            scene_content="Someone comes down the path.",
        )

        assert isinstance(response, str)

    async def test_respond_to_without_initialization(self, character_agent):
        """Test that response fails without initialization."""
        character_agent.tools = None

        with pytest.raises(AgentError, match="Agent session not initialized"):
            await character_agent.respond_to(
                context="Test",
                scene_content="Test",
            )

    async def test_respond_to_stores_memory(self, character_agent, mock_session):
        """Test that response is stored as memory."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.query_memories = AsyncMock(return_value="")
        character_agent.tools.store_memory = AsyncMock(return_value="Stored.")

        await character_agent.respond_to(
            context="Testing memory storage",
            scene_content="Scene content here",
        )

        # Verify memory was stored
        character_agent.tools.store_memory.assert_called_once()
        call_args = character_agent.tools.store_memory.call_args
        assert "conversation" in call_args[1]["memory_type"]


@pytest.mark.asyncio
class TestCharacterAgentObserveAndReact:
    """Tests for observe_and_react method."""

    async def test_observe_and_react_success(self, character_agent, mock_session):
        """Test successful observation and reaction."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.store_memory = AsyncMock(return_value="Stored.")

        reaction = await character_agent.observe_and_react(
            event="A dragon flies overhead",
            emotional_valence=-0.8,
        )

        assert isinstance(reaction, str)

    async def test_observe_and_react_without_initialization(self, character_agent):
        """Test that observation fails without initialization."""
        character_agent.tools = None

        with pytest.raises(AgentError, match="Agent session not initialized"):
            await character_agent.observe_and_react(
                event="Something happens",
            )

    async def test_observe_and_react_stores_memory(self, character_agent, mock_session):
        """Test that observation is stored as memory."""
        await character_agent.initialize_session(mock_session, story_id=1)

        character_agent.tools.store_memory = AsyncMock(return_value="Stored.")

        await character_agent.observe_and_react(
            event="A storm approaches",
            emotional_valence=0.5,
        )

        # Verify memory was stored with correct type and valence
        character_agent.tools.store_memory.assert_called_once()
        call_args = character_agent.tools.store_memory.call_args
        assert call_args[1]["memory_type"] == "observation"
        assert call_args[1]["emotional_valence"] == 0.5


@pytest.mark.asyncio
class TestCharacterAgentDecideAction:
    """Tests for decide_action method."""

    async def test_decide_action_success(self, character_agent, mock_session):
        """Test successful action decision."""
        await character_agent.initialize_session(mock_session, story_id=1)

        decision = await character_agent.decide_action(
            situation="You see a locked door",
            available_actions=["Pick the lock", "Break it down", "Look for a key"],
        )

        assert isinstance(decision, str)
        assert len(decision) > 0

    async def test_decide_action_without_initialization(self, character_agent):
        """Test that decision fails without initialization."""
        character_agent.tools = None

        with pytest.raises(AgentError, match="Agent session not initialized"):
            await character_agent.decide_action(
                situation="Test",
                available_actions=["Action 1"],
            )

    async def test_decide_action_empty_actions(self, character_agent, mock_session):
        """Test decision with no available actions."""
        await character_agent.initialize_session(mock_session, story_id=1)

        decision = await character_agent.decide_action(
            situation="Test",
            available_actions=[],
        )

        # Should still return a response (LLM might comment on lack of options)
        assert isinstance(decision, str)


class TestCharacterAgentGetSummary:
    """Tests for get_summary method."""

    def test_get_summary(self, character_agent, mock_character):
        """Test getting character summary."""
        summary = character_agent.get_summary()

        assert isinstance(summary, dict)
        assert summary["id"] == mock_character.id
        assert summary["name"] == mock_character.name
        assert summary["description"] == mock_character.description
        assert summary["personality"] == mock_character.personality
        assert summary["goals"] == mock_character.goals
        assert summary["background"] == mock_character.background


# =============================================================================
# CharacterAgentTools Tests
# =============================================================================

@pytest.mark.asyncio
class TestAgentToolsQueryMemories:
    """Tests for query_memories method."""

    async def test_query_memories_success(self, agent_tools, mock_semantic_search):
        """Test successful memory query."""
        mock_memory = MagicMock()
        mock_memory.content = "Remembered something important"
        mock_semantic_search.retrieve_character_memories = AsyncMock(
            return_value=[(mock_memory, 0.9)]
        )

        result = await agent_tools.query_memories(
            query="What do I remember",
            limit=5,
        )

        assert "Relevant memories:" in result
        assert "Remembered something important" in result
        assert "0.90" in result

    async def test_query_memories_no_results(self, agent_tools, mock_semantic_search):
        """Test memory query with no results."""
        mock_semantic_search.retrieve_character_memories = AsyncMock(return_value=[])

        result = await agent_tools.query_memories(query="Query")

        assert result == "No relevant memories found."

    async def test_query_memories_with_type_filter(self, agent_tools, mock_semantic_search):
        """Test memory query with type filter."""
        mock_semantic_search.retrieve_character_memories = AsyncMock(return_value=[])

        await agent_tools.query_memories(
            query="Conversations",
            memory_types=["conversation"],
        )

        mock_semantic_search.retrieve_character_memories.assert_called_once()
        call_args = mock_semantic_search.retrieve_character_memories.call_args
        assert call_args[1]["memory_types"] == ["conversation"]

    async def test_query_memories_error(self, agent_tools, mock_semantic_search):
        """Test handling of error in memory query."""
        mock_semantic_search.retrieve_character_memories = AsyncMock(
            side_effect=Exception("Database error")
        )

        result = await agent_tools.query_memories(query="Query")

        assert "Error querying memories" in result


@pytest.mark.asyncio
class TestAgentToolsStoreMemory:
    """Tests for store_memory method."""

    async def test_store_memory_success(self, agent_tools, mock_encoder):
        """Test successful memory storage."""
        result = await agent_tools.store_memory(
            content="I just saw something amazing",
            memory_type="observation",
            emotional_valence=0.8,
            importance=0.9,
        )

        assert "Memory stored" in result
        assert "I just saw something amazing" in result

        # Verify encoder was called
        mock_encoder.encode_async.assert_called_once_with("I just saw something amazing")

        # Verify session add was called
        agent_tools.session.add.assert_called_once()
        added_memory = agent_tools.session.add.call_args[0][0]
        assert isinstance(added_memory, CharacterMemory)

    async def test_store_memory_with_defaults(self, agent_tools):
        """Test storing memory with default values."""
        result = await agent_tools.store_memory(
            content="Test memory",
            memory_type="test",
        )

        assert "Memory stored" in result

    async def test_store_memory_error(self, agent_tools):
        """Test handling of error in memory storage."""
        agent_tools.session.flush = AsyncMock(side_effect=Exception("Database error"))

        result = await agent_tools.store_memory(
            content="Test",
            memory_type="test",
        )

        assert "Error storing memory" in result


@pytest.mark.asyncio
class TestAgentToolsObserveScene:
    """Tests for observe_scene method."""

    async def test_observe_scene(self, agent_tools):
        """Test observing a scene."""
        scene_content = "This is a very long scene description that goes on and on. " * 20

        result = await agent_tools.observe_scene(scene_content)

        assert "Current scene:" in result
        assert len(result) < len(scene_content)  # Should be truncated

    async def test_observe_scene_short_content(self, agent_tools):
        """Test observing a short scene."""
        scene_content = "Short scene."

        result = await agent_tools.observe_scene(scene_content)

        assert "Current scene:" in result
        assert "Short scene" in result


@pytest.mark.asyncio
class TestAgentToolsGetRelationships:
    """Tests for get_relationships method."""

    async def test_get_relationships_general(self, agent_tools):
        """Test getting general relationship info."""
        result = await agent_tools.get_relationships()

        assert "No specific relationships tracked" in result

    async def test_get_relationships_specific_character(self, agent_tools):
        """Test getting relationship with specific character."""
        result = await agent_tools.get_relationships("Alice")

        assert "Alice" in result


@pytest.mark.asyncio
class TestAgentToolsRecallConversation:
    """Tests for recall_conversation method."""

    async def test_recall_conversation(self, agent_tools):
        """Test recalling conversations."""
        agent_tools.query_memories = AsyncMock(
            return_value="Relevant memories:\n- Spoke to hero (relevance: 0.85)"
        )

        result = await agent_tools.recall_conversation("hero", limit=3)

        assert "Spoke to hero" in result

        # Verify query_memories was called with correct parameters
        agent_tools.query_memories.assert_called_once_with(
            query="hero",
            memory_types=["conversation"],
            limit=3,
        )


class TestAgentToolsLangChainTools:
    """Tests for get_langchain_tools method."""

    def test_get_langchain_tools(self, agent_tools):
        """Test getting LangChain tools."""
        tools = agent_tools.get_langchain_tools()

        assert len(tools) == 5
        tool_names = [tool.name for tool in tools]
        assert "query_memories" in tool_names
        assert "store_memory" in tool_names
        assert "observe_scene" in tool_names
        assert "get_relationships" in tool_names
        assert "recall_conversation" in tool_names

    def test_langchain_tools_have_descriptions(self, agent_tools):
        """Test that LangChain tools have descriptions."""
        tools = agent_tools.get_langchain_tools()

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 0
