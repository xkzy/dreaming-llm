## Knowledge Graph Persistence and Selective Saving

The DreamingReasoningLLM uses a dynamic, growable knowledge graph (managed with `networkx`) to connect all dream triplets and their relationships. To enable lifelong learning and avoid memory explosion, only the most important knowledge is periodically saved to disk in GraphML format.

### How It Works

- **In-Memory Graph:** The reasoning graph is built and updated in RAM as the model reasons and dreams.
- **Selective Saving:** Only the top-K most important nodes/edges (e.g., high attention, high reward, or frequently traversed) are saved to a GraphML file using `networkx.write_graphml`.
- **Importance Function:** You can define an `importance_fn(node, data)` that returns a score for each node. The system saves only the nodes with the highest scores (default: 1000 nodes).
- **Persistence:** The saved GraphML file can be reloaded in future sessions to continue growing the knowledge base.

### Example Usage

```python
def importance_fn(node, data):
  # Example: prioritize nodes with high reward or attention
  return data.get("reward", 0.0) + data.get("attention", 0.0)

reasoner.save_important_knowledge_to_graphml(
  dream_sequences,
  filepath="important_knowledge.graphml",
  importance_fn=importance_fn,
  max_nodes=1000
)
```

### Loading Persistent Knowledge

```python
import networkx as nx
G = nx.read_graphml("important_knowledge.graphml")
# Use G to seed or augment the next reasoning session
```

### Benefits

- Prevents memory explosion by limiting graph size
- Enables lifelong, incremental learning
- Only the most valuable knowledge is retained
# Dreaming-Based Reasoning LLM Architecture

## Overview

The **Dreaming-Based Reasoning LLM** is a novel multi-modal architecture that performs reasoning entirely in "image space" through sequences of visual state transitions. Unlike traditional LLMs that reason with text tokens, this model thinks in visual scenes, enabling more intuitive spatial and causal reasoning.

## Key Innovation

**All reasoning happens as "dreams" – sequences of image triplets that are connected via graph-based attention.**

### Core Concept

1. **Universal Image Tokenization**: All inputs (text or images) are converted into image triplets `(what, action, result)`
2. **Dreaming Phase**: Generate multiple parallel sequences of image triplets representing different reasoning paths
3. **Graph Reasoning**: Connect dream sequences via temporal and causal edges, aggregate insights with multi-hop attention
4. **Multi-Modal Output**: Decode reasoning into text, images, or both

## Recent Enhancements (November 2025)

### Mixture of Experts (MoE) Integration

The Dream Generator now uses **4 specialized expert networks** that focus on different reasoning types:

- **Expert 0: Spatial Reasoning** - Physical relationships, positions, movements
- **Expert 1: Temporal Reasoning** - Time-based sequences, duration, ordering
- **Expert 2: Causal Reasoning** - Cause-effect chains, consequences
- **Expert 3: Abstract Reasoning** - Conceptual patterns, symbolic relationships

A **gating network** automatically selects the most appropriate expert(s) for each input based on the initial triplet. This allows the model to specialize its thinking for different types of questions.

### Reinforcement Learning (RL) Integration

The model now supports **continuous learning from feedback** using policy gradient methods:

- **Policy Network**: Guides dream generation toward higher-quality outputs
- **Reward Model**: Evaluates generation quality and provides feedback signals
- **Experience Replay**: Trajectory buffer stores reasoning paths for learning
- **RLHF Support**: Learn from human feedback via `learn_from_feedback()` method

This enables the model to improve over time based on user preferences and task-specific rewards.

## Architecture Components

### 1. Input Tokenizer

Converts any input into image triplets:

```python
Input: "What happens when you drop a ball?"
       ↓
Output: (what, action, result) image embeddings
        what:   [ball in hand]
        action: [releasing motion]
        result: [ball falling]
```

**Implementation:**
- Text → Image: Learned projection network maps text embeddings to visual space
- Images → Triplets: Scene decomposition network splits images into (what, action, result)
- Role embeddings distinguish the three components

### 2. Dream Generator (with MoE)

Creates multiple parallel dream sequences exploring different reasoning paths using specialized expert networks:

```python
Dream 1 (Spatial Expert): ball → release → fall → bounce → settle
Dream 2 (Temporal Expert): ball → release → fall → roll → stop
Dream 3 (Causal Expert): ball → release → fall → splash (in water)
Dream 4 (Abstract Expert): ball → release → fall → break (glass object)
```

**Key Features:**
- **num_dream_sequences**: Number of parallel reasoning paths (default: 4)
- **dream_length**: Steps in each sequence (default: 5)
- **MoE Experts**: 4 specialized dream generators for different reasoning types
- **Gating Network**: Automatically selects expert(s) based on input characteristics
- **Diversity**: Learned offsets create different starting points for exploration
- **Recurrent Generation**: GRU-based state transitions predict next triplets

### 3. Graph Reasoner

Connects dreams into a unified reasoning graph:

```
Nodes: Each image triplet from all dreams
Edges: 
  - Temporal (→): Within-dream sequential flow
  - Causal (⋯): Cross-dream connections at same time step

Reasoning:
  1. Build graph from all dream nodes
  2. Apply multi-hop graph attention (3 hops)
  3. Aggregate all nodes → unified reasoning embedding
```

**Why Graphs?**
- Captures both sequential (temporal) and cross-cutting (causal) relationships
- Multi-hop attention spreads information across reasoning paths
- Learns which dream paths are most relevant

### 4. Output Decoder

Decodes reasoning into final output:

**Text Mode:**
```python
reasoning_embedding → MLP → vocab logits → tokens
Output: "The ball falls due to gravity, bounces, and settles."
```

**Image Mode:**
```python
reasoning_embedding → MLP → image triplet
Output: (what, action, result) visual embeddings
```

**Both Mode:**
```python
Output: {
  "text": generated description,
  "image": (what, action, result) triplet
}
```

## Data Flow Example

### Input: "What happens when you open a door?"

#### Step 1: Input Tokenization
```
Text → Text Embedder → (B, 512)
     → InputTokenizer → 
        what:   door closed, person standing
        action: hand on handle, turning
        result: door opening, view through doorway
```

#### Step 2: Dream Generation (4 sequences × 5 steps)

**Dream Sequence 1 (typical case):**
```
T0: [closed door, person] → [hand on handle] → [turning handle]
T1: [handle turned] → [pushing] → [door opening]
T2: [door ajar] → [walking] → [person stepping through]
T3: [person in doorway] → [continuing] → [entering room]
T4: [person inside] → [releasing door] → [door closing behind]
```

**Dream Sequence 2 (locked door):**
```
T0: [closed door] → [hand on handle] → [turning attempt]
T1: [locked] → [pushing] → [no movement]
T2: [resistance] → [checking] → [looking for key]
T3: [searching] → [finding key] → [inserting key]
T4: [unlocking] → [turning] → [door opening]
```

**Dream Sequence 3 (automatic door):**
```
T0: [closed door] → [approaching] → [motion detected]
T1: [sensor active] → [automatic opening] → [door slides]
T2: [door open] → [walking through] → [passing]
T3: [past sensor] → [automatic closing] → [door slides back]
T4: [door closed] → [waiting] → [ready for next person]
```

**Dream Sequence 4 (emergency exit):**
```
T0: [emergency door] → [pushing bar] → [alarm triggering]
T1: [door opening] → [alarm sound] → [exit accessed]
T2: [person exiting] → [moving outside] → [in exterior]
T3: [door swinging] → [closing] → [lock engaging]
T4: [locked from outside] → [secure] → [cannot reenter]
```

#### Step 3: Graph Reasoning

**Build Graph:**
```
Nodes: 20 total (4 sequences × 5 steps)
Edges:
  - Temporal: 16 edges (sequential flow within each dream)
  - Causal: ~60 edges (connections between dreams at same time step)
```

**Multi-Hop Attention:**
```
Hop 1: Each node attends to temporal neighbors
Hop 2: Nodes attend to causal connections across dreams
Hop 3: Global aggregation weighs most likely paths
```

**Reasoning Embedding:**
```
All nodes → Mean pooling → MLP → (B, 512)
Captures: Most common patterns (typical door opening),
          Edge cases (locked, automatic),
          Context awareness (emergency vs normal)
```

#### Step 4: Output Decoding

**Text Output:**
```
reasoning_embedding → Text Decoder → Tokens
"When you open a door, you turn the handle, push it open, and can walk 
through to the other side. If locked, you need a key first. Some doors 
open automatically when you approach."
```

## Configuration

```python
from image_token_llm.config import DreamingConfig

config = DreamingConfig(
    num_dream_sequences=4,    # Number of parallel reasoning paths
    dream_length=5,            # Steps per dream sequence
    graph_reasoning_hops=3,    # Multi-hop attention iterations
    output_mode="text",        # "text", "image", or "both"
    enable_visualization=True  # Return dream data for visualization
)
```

## Usage Examples

### Basic Text Generation

```python
from image_token_llm.dreaming_model import DreamingReasoningLLM

# Create model with RL enabled
model = DreamingReasoningLLM(device="cuda", enable_rl=True)

# Generate response
output = model.generate(
    prompt="What causes rain?",
    max_length=50,
    temperature=0.8
)
print(output)
# "Rain occurs when water vapor in clouds condenses..."
```

### Learning from Feedback (RL)

```python
# Collect examples with rewards
examples = [
    {"prompt": "Explain photosynthesis", "reward": 0.9},
    {"prompt": "What is gravity?", "reward": 0.7},
    {"prompt": "How do birds fly?", "reward": 0.85},
]

# Continuous learning from feedback
metrics = model.learn_from_feedback(examples, num_epochs=10)
print(f"Average reward: {metrics['avg_reward']:.3f}")
print(f"Policy loss: {metrics['avg_policy_loss']:.3f}")
```

### With Dream Visualization

```python
result = model.generate(
    prompt="How does a plant grow?",
    return_dreams=True
)

print(f"Output: {result['output']}")
print(f"Dreams: {len(result['dreams'])} sequences")
print(f"Graph: {result['graph_data']['num_nodes']} nodes")
```

### Image Input → Text Description

```python
import torch

# Input images (B=1, N=3 images, C=3, H=W=224)
images = load_images(["scene1.jpg", "scene2.jpg", "scene3.jpg"])

description = model.generate(
    images=images,
    output_mode="text",
    max_length=100
)
print(description)
```

### Text → Image Generation

```python
triplet = model.generate(
    prompt="A bird flying over water",
    output_mode="image"
)

what, action, result = triplet
# what: (B, 512) - bird in sky
# action: (B, 512) - flapping wings
# result: (B, 512) - moving forward
```

## Advantages Over Traditional Architectures

### 1. Intuitive Spatial Reasoning
- Visual thinking enables better understanding of physical relationships
- Natural representation for "what happens if" scenarios
- Grounded in perceptual primitives

### 2. Multi-Path Exploration
- Generates multiple reasoning hypotheses in parallel
- Discovers edge cases and alternative outcomes
- More robust to ambiguous situations

### 3. Graph-Based Integration
- Connects related reasoning steps across different paths
- Learns which paths are most probable
- Aggregates insights from diverse explorations

### 4. Multi-Modal Native
- Seamlessly handles text and image inputs/outputs
- No separate encoder/decoder for each modality
- Unified reasoning in visual space

### 5. Interpretable Thinking
- Can visualize the reasoning process (dream sequences)
- See which paths were explored
- Understand how conclusion was reached

## Training Strategy

### Phase 1: Input Tokenization Pre-training
```bash
# Train text→image and image→triplet projections
python scripts/train_input_tokenizer.py \
    --data ./data/mixed_modality \
    --epochs 20 \
    --device cuda
```

### Phase 2: Dream Generation Training
```bash
# Train dream sequence generators on visual reasoning tasks
python scripts/train_dream_generator.py \
    --data ./data/reasoning_sequences \
    --num_dreams 4 \
    --dream_length 5 \
    --device cuda
```

### Phase 3: End-to-End Fine-tuning
```bash
# Fine-tune full model on question-answering tasks
python scripts/train_dreaming_model.py \
    --data ./data/qa_dataset \
    --output ./models/dreaming_llm \
    --device cuda
```

## Limitations and Future Work

### Current Limitations

1. **Computational Cost**: Generating multiple dream sequences is expensive
   - Solution: Adaptive dreaming (vary num_dreams based on complexity)

2. **Dream Quality**: Dreams may not always be coherent
   - Solution: Add consistency losses, dream validation networks

3. **Long Sequences**: Limited by dream_length parameter
   - Solution: Hierarchical dreaming, chunked reasoning

### Future Enhancements

1. **Hierarchical Dreaming**: Multi-level reasoning (coarse → fine)
2. **Attention Visualization**: Interactive dream path exploration
3. **Conditional Dreaming**: User-guided reasoning paths
4. **Real Image Generation**: Integrate diffusion models for actual images
5. **Multi-Agent Dreams**: Multiple dreamers with different perspectives

## Comparison to Other Architectures

| Feature | Traditional LLM | MoE LLM | Dreaming LLM |
|---------|----------------|---------|--------------|
| Reasoning Space | Text tokens | Text tokens | Image triplets |
| Multi-Path | No | Via experts | Via dreams |
| Graph Structure | Sequential | Hierarchical | Fully connected |
| Spatial Understanding | Limited | Limited | Native |
| Interpretability | Low | Medium | High (visualizable) |
| Multi-Modal | Adapter-based | Modality experts | Unified space |

## References

- Graph Neural Networks for reasoning: [Battaglia et al. 2018]
- Visual reasoning in AI: [Johnson et al. 2017]
- Multi-modal transformers: [Radford et al. 2021 - CLIP]
- Chain-of-thought prompting: [Wei et al. 2022]
- Diffusion models for generation: [Ho et al. 2020]

## Citation

```bibtex
@software{dreaming_reasoning_llm,
  title={Dreaming-Based Reasoning LLM: Visual Thinking for Multi-Modal AI},
  author={Your Team},
  year={2025},
  url={https://github.com/your-repo/dreaming-llm}
}
```
