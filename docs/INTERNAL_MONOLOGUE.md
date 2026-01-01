# NPC Internal Monologue System

## Overview

The Internal Monologue system adds narrative depth to LivingWorld by giving NPCs private thoughts that reveal their true motivations, feelings, and reactions. These thoughts can be displayed to players to create dramatic irony and deeper character engagement.

## Features

### What Are Internal Thoughts?

Internal thoughts are what characters are REALLY thinking vs. what they say aloud. They reveal:
- Hidden motivations
- True feelings about the player or other characters
- Secrets and worries
- Plans and scheming
- Emotional reactions that aren't expressed

### Example Output

```
Maya Chen: "I'd be happy to show you to the guest house."
*Internal thought: This traveler seems capable. Maybe they can help with the recent 
disappearances. I shouldn't be too forward though - don't want to scare them off.*
```

## Usage

### Enabling Internal Thoughts

There are several ways to enable the display of internal thoughts:

#### 1. At Story Creation
When starting a new story, you'll be prompted:
```
Show NPC internal thoughts? (adds narrative depth) [yes/no]: yes
```

#### 2. From Settings Menu
From the main menu, select "Settings" then option 1 to toggle.

#### 3. In-Game Commands
While playing, type:
- `thoughts on` - Enable internal thoughts display
- `thoughts off` - Disable internal thoughts display

### What You'll See

When enabled:
- Character table shows a "Thoughts" column
- Character responses include italicized internal thoughts
- Thoughts are stored in character memories for future reference

## Implementation

### Storage

Internal thoughts are stored in the `character_memories` table with:
- `memory_type`: "internal_thought"
- `content`: The thought text
- `emotional_valence`: Detected emotional score
- `importance`: 0.6 (slightly elevated importance)

### Generation Process

1. **When Character Speaks**: After generating dialogue, the system generates an internal thought about what was said vs. what was thought

2. **When Character Acts**: Before taking autonomous action, a thought reveals motivations

3. **When Character Observes**: After observing events, a thought shows their genuine reaction

### Prompt Engineering

The thought generation prompts follow this structure:
- Ask for the gap between public speech and private thought
- Request authentic psychological depth
- Require consistency with personality and goals
- Enforce conciseness (1-3 sentences)

### Methods

#### CharacterAgent.generate_internal_thought()
Generates a thought when a character speaks.

```python
thought = await agent.generate_internal_thought(
    what_was_said=response,
    situation=context,
    other_characters_present=["Player", "Maya"],
)
```

#### CharacterAgent._generate_action_thought()
Generates a thought when taking autonomous action.

```python
thought = await agent._generate_action_thought(
    situation="The village is under attack",
    intended_action="I'll hide the villagers in the cave",
)
```

#### CharacterAgent._generate_observation_thought()
Generates a thought after observing an event.

```python
thought = await agent._generate_observation_thought(
    event_observed="A stranger arrived at midnight",
)
```

## Configuration

### Per-Story Setting
Each story can have its own internal thoughts preference.

### Global Setting
The default setting can be configured in the AgentFactory:

```python
agent_factory = AgentFactory(
    show_internal_thoughts=True,  # Default to enabled
)
```

### Runtime Toggle
Thoughts can be toggled at any time:

```python
agent.set_show_internal_thoughts(True)
factory.set_show_internal_thoughts(True)
```

## Character Prompt Examples

The system uses these examples when generating thoughts:

**Example 1 - Hiding Motivations:**
- Said: "I'd be happy to show you to the guest house."
- Thought: "This traveler seems capable. Maybe they can help with the recent disappearances."

**Example 2 - Concealing Fear:**
- Said: "Of course I trust you, old friend."
- Thought: "If he knew what I did last night, he'd have my head. Keep it together."

**Example 3 - Hiding Secrets:**
- Said: "I have no idea what happened to the merchant's goods."
- Thought: "That gold is hidden beneath the floorboards. Three more days and I can leave forever."

## Narrative Impact

### Dramatic Irony
Players know things their character doesn't, creating tension and engagement.

### Character Depth
Thoughts reveal complexity beyond surface dialogue.

### Story Hooks
Secrets and worries in thoughts can inspire player investigations.

### Replay Value
Playing with thoughts on/off provides different experiences.

## Best Practices

### For Writers
- Keep thoughts concise (1-3 sentences)
- Make them emotionally authentic
- Ensure consistency with established personality
- Use them to hint at future plot developments

### For Players
- Thoughts may contain misinformation (character biases)
- Not every character has complex inner thoughts
- Thoughts are private to each character (not shared between them)
- Use thoughts to guide your interactions and investigations

## Technical Details

### Thought Quality
The system uses:
- Lower temperature (0.7) for more focused thoughts
- Character personality, goals, and background in prompts
- Current emotional state for authenticity
- Context about others present

### Memory Storage
Thoughts are:
- Stored as `internal_thought` type memories
- Given elevated importance (0.6)
- Embedding for semantic retrieval
- Available for future character reflection

### Performance
Thought generation adds ~1-2 seconds per character response.
Can be disabled for faster-paced gameplay.

## Future Enhancements

Potential improvements:
- Thought decay over time (old thoughts less relevant)
- Thought contradictions (character lying to themselves)
- Hierarchical thoughts (surface vs. deep thoughts)
- Thought visualization in UI
- Export thoughts with story
