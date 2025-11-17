# 🌟 Dreaming-Based Reasoning LLM

## Revolutionary Architecture Update

### What Changed?

We've completely redesigned the architecture to use **visual thinking** instead of traditional text-based reasoning. The new model performs all reasoning in "image space" through sequences of visual state transitions called "dreams."

## 🔑 Key Innovation

**All reasoning happens as dreams – sequences of image triplets `(what, action, result)` that are connected via graph-based attention.**

## 🏗️ Architecture Flow

```
Input (text or images)
    ↓
Input Tokenizer
    ↓ (converts everything to image triplets)
Dream Generator (4 parallel sequences × 5 steps)
    ↓ (explores multiple reasoning paths)
Graph Reasoner (temporal + causal edges)
    ↓ (multi-hop attention aggregation)
Output Decoder
    ↓
Output (text, images, or both)
```

## 📦 New Components

### 1. **InputTokenizer** (`dreaming.py`)
- Converts text prompts → image triplets
- Converts images → image triplets
- Universal representation: `(what, action, result)`

### 2. **DreamGenerator** (`dreaming.py`)
- Generates 4 parallel dream sequences
- Each sequence: 5 reasoning steps
- Explores diverse paths with learned offsets
- GRU-based recurrent state transitions

### 3. **DreamGraphReasoner** (`dream_graph_reasoner.py`)
- Builds reasoning graph from all dreams
- Temporal edges: within-dream flow (→)
- Causal edges: cross-dream connections (⋯)
- Multi-hop graph attention (3 hops)
- Aggregates insights across all paths

### 4. **OutputDecoder** (`dreaming.py`)
- Text mode: reasoning → text tokens
- Image mode: reasoning → image triplets
- Both mode: simultaneous text + images
- Autoregressive text generation

### 5. **DreamingReasoningLLM** (`dreaming_model.py`)
- Main model orchestrator
- `generate()`: Text/image generation with dream visualization
- `visualize_thinking()`: Export reasoning process
- `save_pretrained()` / `load_pretrained()`: Model persistence

## 🚀 Quick Start

### Basic Usage

```python
from image_token_llm.dreaming_model import DreamingReasoningLLM

# Create model
model = DreamingReasoningLLM(device="cuda")

# Generate text response
output = model.generate(
    prompt="What happens when you drop a ball?",
    max_length=50,
    temperature=0.8
)
print(output)
# "The ball falls due to gravity, bounces on impact, and settles on the ground."
```

### With Dream Visualization

```python
result = model.generate(
    prompt="How does a plant grow?",
    return_dreams=True
)

print(f"Output: {result['output']}")
print(f"Dreams explored: {len(result['dreams'])} paths")
print(f"Reasoning graph: {result['graph_data']['num_nodes']} nodes")
```

### 🔮 Live Dream Viewing (NEW!)

**Watch what the model is dreaming in real-time during generation!**

```python
# Create model with dream viewer enabled
model = DreamingReasoningLLM(
    device="cuda",
    enable_dream_viewer=True  # Enable dream viewing!
)

# Watch dreams stream live during generation
result = model.generate(
    prompt="What happens when you drop a ball?",
    watch_dreams=True  # Live stream to console!
)

# Export recorded dreams
model.export_recorded_dreams("dreams.json", format="json")

# Create visualization heatmap
model.visualize_dream_evolution(save_path="dreams.png")

# Get dream statistics
viewer = model.get_dream_viewer()
summary = viewer.get_dream_summary()
print(f"Total dreams: {summary['total_dreams']}")
print(f"Avg mean: {summary['avg_mean']:.4f}")
```

**Run the interactive demo:**
```bash
# Watch dreams live
python scripts/demo_dream_viewer.py --watch

# Export dreams to JSON
python scripts/demo_dream_viewer.py --export dreams.json

# Create heatmap visualization
python scripts/demo_dream_viewer.py --visualize dreams.png

# Use pretrained model
python scripts/demo_dream_viewer.py \
    --model-path ./models/distilled_large_scale \
    --watch --visualize dreams.png
```

### Image Input

```python
import torch

# Load images (B=1, N=3 images, C=3, H=W=224)
images = torch.randn(1, 3, 3, 224, 224)

description = model.generate(
    images=images,
    output_mode="text"
)
print(description)
```

### Image Generation

```python
what, action, result = model.generate(
    prompt="A bird flying",
    output_mode="image"
)
# Returns image triplet embeddings (B, 512) each
```

## 📊 Architecture Diagram

See [`docs/dreaming_architecture.svg`](docs/dreaming_architecture.svg) for the complete visual diagram showing:
- Input tokenization flow
- 4 parallel dream sequences with 5 steps each
- Temporal and causal graph connections
- Multi-hop attention aggregation
- Text/image output decoding
- Key innovations and example flow

## 📖 Documentation

### Comprehensive Guide
- **[`docs/DREAMING_ARCHITECTURE.md`](docs/DREAMING_ARCHITECTURE.md)**: Complete architecture explanation
  - Design rationale
  - Component details
  - Data flow examples
  - Configuration guide
  - Training strategy
  - Comparisons to other architectures

### Examples
- **[`examples/dreaming_examples.py`](examples/dreaming_examples.py)**: 6 usage examples
  1. Text → Dreaming → Text
  2. Images → Dreaming → Text
  3. Text → Dreaming → Images
  4. Mixed output (text + images)
  5. Visualizing thinking process
  6. Save and load models

## ✅ Tests

All 21 tests pass! Run with:

```bash
pytest tests/test_dreaming_model.py -v
```

### Test Coverage
- ✅ InputTokenizer (text/image → triplets)
- ✅ DreamSequence (single dream generation)
- ✅ DreamGenerator (multiple parallel dreams)
- ✅ DreamGraphReasoner (graph construction + attention)
- ✅ OutputDecoder (text/image/both modes)
- ✅ DreamingReasoningLLM (full model)
- ✅ Integration tests (end-to-end pipelines)

## 🎯 Advantages Over Previous MoE Architecture

| Feature | MoE Architecture | Dreaming Architecture |
|---------|-----------------|----------------------|
| **Reasoning Space** | Text tokens | Image triplets |
| **Multi-Path Reasoning** | Via experts | Via dreams (more intuitive) |
| **Spatial Understanding** | Limited | Native (visual thinking) |
| **Interpretability** | Medium | High (visualizable dreams) |
| **Multi-Modal** | Separate experts | Unified image space |
| **Causality** | Sequential | Graph-based (temporal + causal) |

## 🔧 Configuration

```python
from image_token_llm.config import DreamingConfig

config = DreamingConfig(
    num_dream_sequences=4,    # Number of parallel reasoning paths
    dream_length=5,            # Steps per dream sequence
    graph_reasoning_hops=3,    # Multi-hop attention iterations
    output_mode="text",        # "text", "image", or "both"
    enable_visualization=True  # Return dream data
)
```

## 📁 New Files

### Core Implementation
- `src/image_token_llm/dreaming.py` - InputTokenizer, DreamGenerator, OutputDecoder
- `src/image_token_llm/dream_graph_reasoner.py` - Graph-based reasoning
- `src/image_token_llm/dreaming_model.py` - Main model orchestrator

### Configuration
- Updated `src/image_token_llm/config.py` - Added `DreamingConfig`

### Documentation
- `docs/dreaming_architecture.svg` - Visual architecture diagram
- `docs/DREAMING_ARCHITECTURE.md` - Comprehensive guide

### Examples & Tests
- `examples/dreaming_examples.py` - 6 usage examples
- `tests/test_dreaming_model.py` - 21 comprehensive tests

## 🎨 Example: "What happens when you open a door?"

### Input Tokenization
```
Text prompt → Image triplet:
  what:   [closed door, person standing]
  action: [hand on handle, turning]
  result: [door opening, view through]
```

### Dreaming (4 sequences × 5 steps)

**Dream 1 (typical):** door closed → turning handle → door opening → walking through → door closing

**Dream 2 (locked):** door closed → turning fails → resistance → finding key → unlocking → opening

**Dream 3 (automatic):** approaching door → sensor detects → automatic opening → walking through → automatic closing

**Dream 4 (emergency):** pushing bar → alarm triggers → door opens → exiting → locks behind

### Graph Reasoning
- **Nodes:** 20 total (4 dreams × 5 steps)
- **Temporal edges:** Within each dream sequence
- **Causal edges:** Between dreams at same time steps
- **Aggregation:** Weighs most likely path (typical door opening)

### Output
```
"When you open a door, you turn the handle, push it open, and can walk 
through to the other side. If locked, you need a key first. Some doors 
open automatically when you approach."
```

## 🚧 Migration from MoE Model

The legacy MoE model is still available:

```python
# Old MoE model (still works)
from image_token_llm.model import ImageTokenReasoningLLM

# New dreaming model (recommended)
from image_token_llm.dreaming_model import DreamingReasoningLLM
```

## 🎓 Key Concepts

### Why "Dreams"?
Dreams are sequences of visual states representing possible reasoning paths. Like human imagination, the model "dreams" about what might happen, exploring multiple scenarios in parallel.

### Why Image Triplets?
`(what, action, result)` captures causality:
- **what**: Current state/scene
- **action**: Transformation/event
- **result**: New state after action

This is more intuitive than text tokens for spatial/physical reasoning.

### Why Graphs?
Graphs connect related reasoning steps across different paths. Temporal edges preserve sequential flow, causal edges link similar states across dreams.

## 📈 Performance Characteristics

### Strengths
- ✅ Intuitive spatial/physical reasoning
- ✅ Multi-path exploration (robustness)
- ✅ Interpretable thinking process
- ✅ Native multi-modal support
- ✅ Captures causal relationships

### Trade-offs
- ⚠️ Higher computational cost (multiple dreams)
- ⚠️ Requires visual scene understanding
- ⚠️ Limited by dream_length parameter

## 🔮 Future Enhancements

1. **Hierarchical Dreaming** - Multi-level reasoning (coarse → fine)
2. **Attention Visualization** - Interactive dream path exploration
3. **Conditional Dreaming** - User-guided reasoning
4. **Real Image Generation** - Integrate diffusion models
5. **Multi-Agent Dreams** - Multiple perspectives

## 🙏 Acknowledgments

This architecture is inspired by:
- Graph Neural Networks for reasoning
- Visual reasoning in cognitive science
- Chain-of-thought prompting
- Diffusion models for image generation

## 📄 License

Same as the main project.

---

**Ready to think visually? Try the dreaming model today!** 🌟
