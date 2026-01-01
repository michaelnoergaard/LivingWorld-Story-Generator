# LangChain Agent Implementation Guide

This document describes the full LangChain Agent implementation for autonomous character NPCs in LivingWorld.

## Overview

The LivingWorld system now implements true LangChain Agents with:

1. **Autonomous Tool-Using Agents**: Characters can use LangChain's ReAct agents to autonomously decide which tools to use
2. **Relationship Tracking**: Characters track and update relationships with other characters
3. **Emotional State System**: Characters maintain emotional state (valence, arousal) that affects behavior
4. **Autonomous Actions**: Characters can initiate actions based on their goals and personality
5. **Emotional Valence Detection**: System detects and stores emotional content of memories

## Architecture

### Database Schema

#### Character Table (Enhanced)
```python
- current_mood: VARCHAR(100)  # Current mood state
- emotional_state: JSON       # {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}
```

#### CharacterRelationship Table (New)
```python
- story_id: INTEGER
- character_a_id: INTEGER     # Source character
- character_b_id: INTEGER     # Target character
- sentiment_score: FLOAT      # -1.0 (hate) to 1.0 (love)
- trust_level: FLOAT          # 0.0 to 1.0
- familiarity: FLOAT          # 0.0 (stranger) to 1.0 (know well)
- relationship_type: VARCHAR  # friend, enemy, rival, mentor, etc.
- interaction_count: INTEGER  # Number of interactions
- last_interaction: TIMESTAMP
- notes: TEXT
```

### Character Agent System

#### CharacterAgent Class

The `CharacterAgent` class provides:

1. **Direct Response Mode** (`_direct_respond`)
   - Fast, simple response generation
   - Uses Ollama directly without agent overhead
   - Retrieves memories, relationships, and emotional context

2. **LangChain Agent Mode** (`_agent_respond`)
   - Full ReAct agent with tool access
   - Agent autonomously decides which tools to use
   - Can query memories, store memories, check relationships
   - Falls back to direct mode if agent fails

3. **Autonomous Action** (`autonomous_action`)
   - Character decides what to do based on goals
   - Considers personality, memories, relationships
   - Stores action as memory

4. **Character Interaction** (`interact_with_character`)
   - Handles character-to-character interactions
   - Updates relationship scores
   - Stores interaction memory

5. **Emotional System**
   - `_detect_emotional_valence()`: Detects emotional content
   - `_update_emotional_state()`: Updates emotional state
   - Uses exponential moving average for smooth changes

### CharacterAgentTools Class

Provides LangChain-compatible tools:

1. **query_memories**: Search memories by semantic similarity
2. **store_memory**: Store new memories with emotional valence
3. **observe_scene**: Get current scene context
4. **get_relationships**: Query relationship information
5. **recall_conversation**: Recall past conversations

#### Async/Sync Wrappers

LangChain's Tool wrapper requires synchronous functions. The system uses `asyncio.get_event_loop().run_until_complete()` to create sync wrappers for async methods:

```python
def sync_query_memories(query: str) -> str:
    """Synchronous wrapper for query_memories."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(self.query_memories(query))
```

## Usage Examples

### Basic Character Response

```python
# Create agent
agent = await agent_factory.create_agent(character, session, story_id)

# Generate response
response = await agent.respond_to(
    context="Someone approaches you",
    scene_content="A stranger walks into the tavern",
    other_characters=["Stranger"],
)
```

### Using LangChain Agent Mode

```python
# Use full agent with tools
response = await agent.respond_to(
    context="Someone approaches you",
    scene_content="A stranger walks into the tavern",
    other_characters=["Stranger"],
    use_agent=True,  # Enable LangChain Agent
)
```

### Autonomous Action

```python
# Character decides what to do
action = await agent.autonomous_action(
    situation="A stranger arrives at the village",
    other_characters_present=[2, 3],
)

# Returns:
# {
#     "character_id": 1,
#     "character_name": "Guard Captain",
#     "action": "I approach cautiously and ask their business",
#     "emotional_state": {"valence": 0.1, "arousal": 0.6}
# }
```

### Character Interaction

```python
# Two characters interact
response = await agent.interact_with_character(
    other_character_id=2,
    interaction_content="The merchant waves and says hello",
    interaction_type="conversation",
)

# Automatically:
# - Updates relationship sentiment
# - Stores interaction memory
# - Generates character response
```

### Observation and Reaction

```python
# Character observes event and reacts
reaction = await agent.observe_and_react(
    event="A loud thunderclap echoes through the valley",
    emotional_valence=-0.3,  # Slightly negative
)
```

## Story Generation Integration

### Autonomous Character Actions in Scenes

The `StoryGenerator` now includes autonomous character actions:

```python
# In generate_next_scene()
character_actions = await self._generate_character_autonomous_actions(
    session=session,
    story_id=story_id,
    scene_id=scene_id,
    situation=f"Player chose: {choice_text}",
)

# Actions are included in scene content
if character_actions:
    current_scene_content += "\n\nCharacter Actions:\n"
    for action in character_actions:
        current_scene_content += f"- {action['character_name']}: {action['action'][:100]}...\n"
```

Each character has a 40% chance to act autonomously in each scene (configurable).

## Relationship Tracking

### Creating Relationships

Relationships are automatically created when characters interact:

```python
# In interact_with_character()
await self.tools.update_relationship(
    other_character_id=other_character_id,
    sentiment_delta=sentiment_delta * 0.2,  # Small incremental change
    trust_delta=0.05 if sentiment_delta > 0 else -0.05,
    interaction_type=interaction_type,
)
```

### Relationship Queries

```python
# Get all relationships
rel_info = await agent.tools.get_relationships()

# Get specific relationship
rel_info = await agent.tools.get_relationships("Merchant Guildmaster")

# Returns:
# "Your relationships:
# - Merchant Guildmaster: friend (positive, trust: 75%, familiarity: 40%)"
```

## Emotional State System

### Emotional Valence Detection

Simple keyword-based approach:

```python
positive_words = ["happy", "joy", "love", "good", "great", ...]
negative_words = ["sad", "angry", "hate", "bad", "terrible", ...]

valence = (positive_count - negative_count) / max(total, 1)
```

### Emotional State Update

Uses exponential moving average:

```python
new_valence = 0.7 * current_valence + 0.3 * detected_valence

# Mood updates based on valence
if valence > 0.3: mood = "positive"
elif valence < -0.3: mood = "negative"
else: mood = "neutral"
```

### Emotional Context in Prompts

Character prompts include emotional state:

```python
if self.character.emotional_state:
    prompt += f"\n## Your Emotional State\n{json.dumps(self.character.emotional_state)}"

if self.character.current_mood:
    prompt += f"\n## Your Current Mood\n{self.character.current_mood}"
```

## LangChain Agent Implementation

### ReAct Agent Setup

```python
# Create ChatOllama instance
llm = ChatOllama(
    model=self.ollama_client.config.model,
    temperature=self._get_temperature(),
)

# Create ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create executor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=5,
)
```

### Agent Prompt Template

```
You are {character_name}.

{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

## Database Migration

### Applying the Migration

```bash
python src/database/migrations/add_character_relationships.py
```

### Rolling Back

```bash
python src/database/migrations/add_character_relationships.py down
```

The migration:
1. Adds `current_mood` and `emotional_state` columns to `characters` table
2. Creates `character_relationships` table
3. Creates indexes for performance

## Testing

### Running Tests

```bash
pytest tests/test_langchain_agents.py -v
```

### Test Coverage

Tests cover:
- Character agent initialization
- Direct response generation
- Emotional valence detection
- Emotional state updates
- Autonomous action generation
- Character-to-character interaction
- Observation and reaction
- Relationship tracking
- LangChain tool wrappers

## Configuration

### Character Agent Config

```python
character.agent_config = {
    "temperature": 0.8,  # Response randomness
    "use_autonomous_actions": True,
    "autonomous_action_probability": 0.4,
}
```

### Character Emotional State

```python
character.emotional_state = {
    "valence": 0.0,      # -1.0 (negative) to 1.0 (positive)
    "arousal": 0.5,      # 0.0 (calm) to 1.0 (excited)
    "dominance": 0.5,    # 0.0 (submissive) to 1.0 (dominant)
}
character.current_mood = "neutral"  # or "positive", "negative"
```

## Performance Considerations

### Direct Mode vs Agent Mode

- **Direct Mode**: Faster, lower latency, suitable for simple responses
- **Agent Mode**: More sophisticated, slower, better for complex decisions

Use `use_agent=False` for:
- Simple dialogue responses
- High-frequency interactions
- Scenes with many characters

Use `use_agent=True` for:
- Complex decisions
- When tool use is beneficial
- Important story moments

### Async Tool Wrappers

The async/sync wrapper approach has limitations:
- Assumes running in async context
- May block event loop during tool execution
- Consider using `asyncio.create_task()` for parallel tool execution

## Future Enhancements

1. **Better Sentiment Analysis**: Use actual sentiment analysis model instead of keyword matching
2. **Parallel Tool Execution**: Run multiple tools concurrently
3. **Relationship Inference**: Infer relationships from story content
4. **Memory Prioritization**: Use emotional valence to prioritize memory retrieval
5. **Character Development**: Characters change personality over time based on experiences
6. **Social Dynamics**: Group behaviors, reputation, gossip
7. **Advanced Agent Types**: Implement different agent types (e.g., planning agents, reflective agents)

## Troubleshooting

### Agent Fails to Initialize

Check that:
- Ollama is running and model is available
- Database connection is working
- Character has required fields (personality, goals, background)

### Relationships Not Updating

Check that:
- `CharacterRelationship` table exists
- Characters have been created in database
- `interact_with_character()` is being called

### Emotional State Not Changing

Check that:
- Emotional valence detection is working
- `_update_emotional_state()` is being called
- Character model is being saved to database

### LangChain Agent Not Using Tools

Check that:
- Tools are properly initialized
- Agent executor is created
- `use_agent=True` is passed to `respond_to()`
- Agent prompt is correctly formatted

## Files Modified

- `/home/michael/Projects/LivingWorld/src/database/models.py` - Added CharacterRelationship model
- `/home/michael/Projects/LivingWorld/src/agents/agent_tools.py` - Fixed async wrappers, added relationship tracking
- `/home/michael/Projects/LivingWorld/src/agents/character_agent.py` - Full LangChain Agent implementation
- `/home/michael/Projects/LivingWorld/src/llm/story_generator.py` - Integrated autonomous character actions
- `/home/michael/Projects/LivingWorld/src/database/migrations/add_character_relationships.py` - Database migration
- `/home/michael/Projects/LivingWorld/tests/test_langchain_agents.py` - Comprehensive tests

## Summary

The LivingWorld system now has:
- True LangChain Agents with autonomous tool-using capabilities
- Characters that can initiate actions based on their goals
- Relationship tracking between characters
- Emotional state affecting character behavior
- Proper async tool wrappers that work with LangChain
- Comprehensive testing for all new features

This creates a more dynamic and responsive story world where characters are true autonomous agents with their own goals, memories, emotions, and relationships.
