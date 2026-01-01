"""Tests for LangChain agent capabilities."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.character_agent import CharacterAgent
from src.agents.agent_tools import CharacterAgentTools
from src.database.models import Character, CharacterRelationship, CharacterMemory
from src.core.exceptions import AgentError


@pytest.fixture
def mock_character():
    """Create a mock character."""
    character = Mock(spec=Character)
    character.id = 1
    character.name = "Test Character"
    character.description = "A test character"
    character.personality = "Brave, curious, and friendly"
    character.goals = "To explore the world and help others"
    character.background = "Grew up in a small village"
    character.agent_config = {"temperature": 0.8}
    character.emotional_state = {"valence": 0.0, "arousal": 0.5}
    character.current_mood = "neutral"
    return character


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = Mock(spec=AsyncSession)
    session.add = Mock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_tools(mock_session):
    """Create mock agent tools."""
    tools = Mock(spec=CharacterAgentTools)
    tools.query_memories = AsyncMock(return_value="No relevant memories found.")
    tools.store_memory = AsyncMock(return_value="Memory stored")
    tools.get_relationships = AsyncMock(return_value="No relationships yet")
    tools.observe_scene = AsyncMock(return_value="Scene observed")
    return tools


@pytest.mark.asyncio
async def test_character_agent_initialization(mock_character):
    """Test that character agent initializes correctly."""
    with patch('src.agents.character_agent.CharacterAgentTools'):
        ollama_client = Mock()
        encoder = Mock()
        semantic_search = Mock()
        prompt_builder = Mock()

        agent = CharacterAgent(
            character=mock_character,
            ollama_client=ollama_client,
            encoder=encoder,
            semantic_search=semantic_search,
            prompt_builder=prompt_builder,
        )

        assert agent.character == mock_character
        assert agent.system_prompt is not None
        assert mock_character.name in agent.system_prompt
        assert agent.tools is None


@pytest.mark.asyncio
async def test_character_agent_direct_respond(mock_character, mock_session, mock_tools):
    """Test character agent direct response generation."""
    ollama_client = Mock()
    ollama_client.generate_with_retry = AsyncMock(return_value="Hello! I'm glad to meet you.")
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()
    prompt_builder.build_character_system_prompt = Mock(return_value="You are Test Character")

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    agent.tools = mock_tools

    response = await agent._direct_respond(
        context="Someone approaches you",
        scene_content="A stranger walks into the room",
        other_characters=["Stranger"],
    )

    assert response is not None
    assert len(response) > 0
    assert mock_tools.query_memories.called
    assert mock_tools.get_relationships.called
    assert mock_tools.store_memory.called


@pytest.mark.asyncio
async def test_emotional_valence_detection(mock_character, mock_session, mock_tools):
    """Test emotional valence detection."""
    ollama_client = Mock()
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    # Test positive text
    positive_valence = await agent._detect_emotional_valence("I am so happy and excited!")
    assert positive_valence > 0

    # Test negative text
    negative_valence = await agent._detect_emotional_valence("This is terrible and makes me angry")
    assert negative_valence < 0

    # Test neutral text
    neutral_valence = await agent._detect_emotional_valence("The weather is cloudy")
    assert neutral_valence == 0


@pytest.mark.asyncio
async def test_emotional_state_update(mock_character, mock_session, mock_tools):
    """Test emotional state updates."""
    ollama_client = Mock()
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    # Test positive emotion update
    await agent._update_emotional_state(0.5)
    assert agent.character.emotional_state["valence"] > 0
    assert agent.character.current_mood == "positive"

    # Test negative emotion update
    await agent._update_emotional_state(-0.7)
    assert agent.character.emotional_state["valence"] < 0
    assert agent.character.current_mood == "negative"


@pytest.mark.asyncio
async def test_autonomous_action(mock_character, mock_session, mock_tools):
    """Test autonomous action generation."""
    ollama_client = Mock()
    ollama_client.generate_with_retry = AsyncMock(
        return_value="I decide to approach the stranger cautiously and ask who they are."
    )
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    agent.tools = mock_tools

    action = await agent.autonomous_action(
        situation="A stranger arrives at the village",
        other_characters_present=[2, 3],
    )

    assert action is not None
    assert "character_id" in action
    assert action["character_id"] == 1
    assert "action" in action
    assert "emotional_state" in action
    assert mock_tools.store_memory.called


@pytest.mark.asyncio
async def test_character_interaction(mock_character, mock_session, mock_tools):
    """Test character-to-character interaction."""
    ollama_client = Mock()
    ollama_client.generate_with_retry = AsyncMock(
        return_value="I smile and greet them warmly."
    )
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    agent.tools = mock_tools

    response = await agent.interact_with_character(
        other_character_id=2,
        interaction_content="The other character waves and says hello",
        interaction_type="conversation",
    )

    assert response is not None
    assert mock_tools.update_relationship.called
    assert mock_tools.store_memory.called


@pytest.mark.asyncio
async def test_observe_and_react(mock_character, mock_session, mock_tools):
    """Test character observation and reaction."""
    ollama_client = Mock()
    ollama_client.generate_with_retry = AsyncMock(
        return_value="I look around curiously, trying to understand what happened."
    )
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    agent.tools = mock_tools

    reaction = await agent.observe_and_react(
        event="A loud thunderclap echoes through the valley",
        emotional_valence=-0.3,
    )

    assert reaction is not None
    assert mock_tools.store_memory.called


@pytest.mark.asyncio
async def test_relationship_tracking(mock_session):
    """Test relationship tracking between characters."""
    # Create a relationship
    relationship = CharacterRelationship(
        story_id=1,
        character_a_id=1,
        character_b_id=2,
        sentiment_score=0.7,
        trust_level=0.8,
        familiarity=0.5,
        relationship_type="friend",
        interaction_count=5,
    )

    mock_session.add(relationship)
    await mock_session.flush()

    assert relationship.id is not None
    assert relationship.sentiment_score == 0.7
    assert relationship.relationship_type == "friend"


@pytest.mark.asyncio
async def test_langchain_tool_wrappers():
    """Test that LangChain tool wrappers work correctly."""
    with patch('asyncio.get_event_loop') as mock_loop:
        # Setup mock event loop
        loop = Mock()
        loop.run_until_complete = Mock(return_value="test result")
        mock_loop.return_value = loop

        session = Mock(spec=AsyncSession)
        semantic_search = Mock()

        tools = CharacterAgentTools(
            session=session,
            character_id=1,
            story_id=1,
            semantic_search=semantic_search,
        )

        # Get LangChain tools
        langchain_tools = tools.get_langchain_tools()

        assert len(langchain_tools) == 5

        # Test each tool
        tool_names = [tool.name for tool in langchain_tools]
        assert "query_memories" in tool_names
        assert "store_memory" in tool_names
        assert "observe_scene" in tool_names
        assert "get_relationships" in tool_names
        assert "recall_conversation" in tool_names

        # Verify tools are callable
        for tool in langchain_tools:
            assert callable(tool.func)


@pytest.mark.asyncio
async def test_agent_without_session_raises_error(mock_character):
    """Test that agent raises error when session not initialized."""
    ollama_client = Mock()
    encoder = Mock()
    semantic_search = Mock()
    prompt_builder = Mock()

    agent = CharacterAgent(
        character=mock_character,
        ollama_client=ollama_client,
        encoder=encoder,
        semantic_search=semantic_search,
        prompt_builder=prompt_builder,
    )

    # Don't initialize session
    agent.tools = None

    with pytest.raises(AgentError, match="session not initialized"):
        await agent._direct_respond(
            context="Test",
            scene_content="Test scene",
        )


@pytest.mark.asyncio
async def test_character_memory_with_emotional_valence(mock_session):
    """Test that character memories store emotional valence."""
    memory = CharacterMemory(
        character_id=1,
        story_id=1,
        memory_type="conversation",
        content="Had a wonderful conversation with a friend",
        emotional_valence=0.8,
        importance=0.7,
        embedding=[0.1] * 384,
    )

    mock_session.add(memory)
    await mock_session.flush()

    assert memory.id is not None
    assert memory.emotional_valence == 0.8
    assert memory.memory_type == "conversation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
