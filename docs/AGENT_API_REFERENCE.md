# Character Agent API Reference

Quick reference for the CharacterAgent API.

## CharacterAgent

### Initialization

```python
from src.agents.agent_factory import get_agent_factory

factory = get_agent_factory()
agent = await factory.create_agent(character, session, story_id)
```

### Methods

#### respond_to()

Generate character response to a situation.

```python
response = await agent.respond_to(
    context: str,                    # Current situation
    scene_content: str,              # Scene description
    other_characters: List[str] = None,  # Other characters present
    use_agent: bool = False,         # Use LangChain Agent?
) -> str
```

**Example:**
```python
response = await agent.respond_to(
    context="The merchant offers you a deal",
    scene_content="You stand in the busy marketplace",
    other_characters=["Merchant"],
    use_agent=True,
)
```

---

#### autonomous_action()

Character decides what to do autonomously.

```python
action = await agent.autonomous_action(
    situation: str,                      # Current situation
    other_characters_present: List[int] = None,  # Character IDs present
) -> Dict[str, Any]
```

**Returns:**
```python
{
    "character_id": int,
    "character_name": str,
    "action": str,
    "emotional_state": dict,
}
```

**Example:**
```python
action = await agent.autonomous_action(
    situation="A goblin approaches the village",
    other_characters_present=[2, 3, 4],
)
print(f"{action['character_name']}: {action['action']}")
```

---

#### interact_with_character()

Interact with another character.

```python
response = await agent.interact_with_character(
    other_character_id: int,         # Target character ID
    interaction_content: str,        # What was said/done
    interaction_type: str = "conversation",  # Type of interaction
) -> str
```

**Example:**
```python
response = await agent.interact_with_character(
    other_character_id=2,
    interaction_content="The merchant smiles and offers a discount",
    interaction_type="trade",
)
```

---

#### observe_and_react()

Observe an event and react to it.

```python
reaction = await agent.observe_and_react(
    event: str,                      # Event description
    emotional_valence: float = 0.0,  # -1.0 to 1.0
) -> str
```

**Example:**
```python
reaction = await agent.observe_and_react(
    event="The castle bell rings loudly",
    emotional_valence=-0.3,  # Slightly negative (alarm)
)
```

---

#### decide_action()

Decide on an action from available options.

```python
decision = await agent.decide_action(
    situation: str,              # Current situation
    available_actions: List[str],  # Action options
) -> str
```

**Example:**
```python
decision = await agent.decide_action(
    situation="The door is locked",
    available_actions=[
        "Pick the lock",
        "Break down the door",
        "Look for another entrance",
    ],
)
```

---

## CharacterAgentTools

Tools are automatically initialized when agent session starts.

### query_memories()

```python
memories = await tools.query_memories(
    query: str,                          # Search query
    memory_types: List[str] = None,      # Filter by type
    limit: int = 5,                      # Max results
) -> str
```

**Example:**
```python
memories = await tools.query_memories(
    query="merchant interactions",
    memory_types=["conversation", "trade"],
    limit=3,
)
```

---

### store_memory()

```python
result = await tools.store_memory(
    content: str,                        # Memory content
    memory_type: str,                    # Type
    emotional_valence: float = 0.0,      # -1.0 to 1.0
    importance: float = 0.5,             # 0.0 to 1.0
) -> str
```

**Example:**
```python
await tools.store_memory(
    content="Traded with the merchant, got a good price",
    memory_type="trade",
    emotional_valence=0.7,  # Positive
    importance=0.8,  # Important
)
```

---

### get_relationships()

```python
relationships = await tools.get_relationships(
    other_character_name: str = None,  # Specific character
) -> str
```

**Example:**
```python
# Get all relationships
all_rels = await tools.get_relationships()

# Get specific relationship
merchant_rel = await tools.get_relationships("Merchant Guildmaster")
```

---

### update_relationship()

```python
result = await tools.update_relationship(
    other_character_id: int,
    sentiment_delta: float = 0.0,       # Change in sentiment
    trust_delta: float = 0.0,           # Change in trust
    interaction_type: str = None,       # Type label
) -> str
```

**Example:**
```python
await tools.update_relationship(
    other_character_id=2,
    sentiment_delta=0.2,   # Improve sentiment
    trust_delta=0.1,       # Increase trust
    interaction_type="friend",
)
```

---

## StoryGenerator Integration

### Autonomous Actions in Scenes

Characters can act autonomously during scene generation:

```python
from src.llm.story_generator import StoryGenerator

generator = StoryGenerator(...)

# During scene generation, characters may act autonomously
scene = await generator.generate_next_scene(
    story_id=1,
    choice=1,
)

# Check if characters acted
if scene.character_actions:
    for action in scene.character_actions:
        print(f"{action['character_name']}: {action['action']}")
```

---

## Relationship System

### Creating Relationships

Relationships are created automatically when characters interact.

```python
# Relationship fields
relationship = CharacterRelationship(
    story_id=1,
    character_a_id=1,       # Source character
    character_b_id=2,       # Target character
    sentiment_score=0.5,    # -1.0 to 1.0
    trust_level=0.6,        # 0.0 to 1.0
    familiarity=0.3,        # 0.0 to 1.0
    relationship_type="friend",  # Optional type label
    interaction_count=3,
)
```

### Querying Relationships

```python
from sqlalchemy import select
from src.database.models import CharacterRelationship

# Get character's relationships
result = await session.execute(
    select(CharacterRelationship).where(
        CharacterRelationship.character_a_id == character_id
    )
)
relationships = result.scalars().all()

for rel in relationships:
    print(f"Sentiment: {rel.sentiment_score}, Trust: {rel.trust_level}")
```

---

## Emotional State

### Accessing Emotional State

```python
# Get current emotional state
emotional_state = agent.character.emotional_state
# {"valence": 0.3, "arousal": 0.6, "dominance": 0.5}

# Get current mood
mood = agent.character.current_mood
# "positive", "negative", or "neutral"
```

### Setting Initial Emotional State

```python
character.emotional_state = {
    "valence": 0.5,      # Positive
    "arousal": 0.7,      # Excited
    "dominance": 0.6,    # Confident
}
character.current_mood = "positive"
```

---

## LangChain Agent Usage

### Enabling LangChain Agent Mode

```python
# Use full agent with tools
response = await agent.respond_to(
    context="Someone asks for directions",
    scene_content="You are at the village square",
    use_agent=True,  # Enable LangChain Agent
)

# Agent will autonomously:
# 1. Query relevant memories
# 2. Check relationships
# 3. Store new memory
# 4. Generate response
```

### When to Use Agent Mode

**Use Agent Mode (use_agent=True) when:**
- Character needs to make complex decisions
- You want agent to use tools autonomously
- Story situation is important/complex
- You have time for additional processing

**Use Direct Mode (use_agent=False) when:**
- Simple dialogue responses
- High-frequency interactions
- Performance is critical
- Tools aren't needed

---

## Error Handling

### Common Errors

```python
from src.core.exceptions import AgentError

try:
    response = await agent.respond_to(...)
except AgentError as e:
    # Session not initialized
    if "session not initialized" in str(e):
        await agent.initialize_session(session, story_id)
        response = await agent.respond_to(...)

    # Agent execution failed
    elif "execution failed" in str(e):
        # Falls back to direct mode automatically
        response = await agent.respond_to(..., use_agent=False)
```

---

## Configuration

### Character Agent Config

```python
character.agent_config = {
    "temperature": 0.8,  # Response randomness (0.0-1.0)
}
```

### Story Generator Config

```python
generator = StoryGenerator(
    ollama_client=ollama,
    prompt_builder=builder,
    encoder=encoder,
    session_factory=session_factory,
    use_agents=True,  # Enable character agents
)
```

---

## Quick Examples

### Complete Character Interaction

```python
# Setup
factory = get_agent_factory()
agent = await factory.create_agent(character, session, story_id)

# Character responds to situation
response = await agent.respond_to(
    context="A traveler asks for shelter",
    scene_content="Evening falls on the village",
    other_characters=["Traveler"],
)

# Character interacts with another
reaction = await agent.interact_with_character(
    other_character_id=2,
    interaction_content="The traveler bows respectfully",
    interaction_type="conversation",
)

# Character acts autonomously
action = await agent.autonomous_action(
    situation="The village is under attack",
    other_characters_present=[2, 3, 4],
)
```

### Story with Autonomous Characters

```python
# Generate scene with autonomous character actions
scene = await generator.generate_next_scene(
    story_id=1,
    choice=1,
    # Characters may act autonomously during generation
)

# Display character actions
if scene.character_actions:
    print("\nAutonomous Actions:")
    for action in scene.character_actions:
        print(f"  {action['character_name']}: {action['action']}")
```

---

## Performance Tips

1. **Use Direct Mode for Simple Responses**
   ```python
   # Faster
   await agent.respond_to(context, scene, use_agent=False)

   # Slower but more sophisticated
   await agent.respond_to(context, scene, use_agent=True)
   ```

2. **Cache Agents**
   ```python
   factory = get_agent_factory()
   agent = factory.get_cached_agent(character_id)
   ```

3. **Limit Memory Retrieval**
   ```python
   memories = await tools.query_memories(query, limit=3)  # Not 10
   ```

4. **Batch Character Actions**
   ```python
   # Let story generator handle all characters at once
   scene = await generator.generate_next_scene(story_id, choice)
   # Instead of calling individual agents
   ```

---

## See Also

- [LangChain Agents Guide](LANGCHAIN_AGENTS.md) - Full implementation details
- [Database Models](../src/database/models.py) - Schema reference
- [Tests](../tests/test_langchain_agents.py) - Usage examples
