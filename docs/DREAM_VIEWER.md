# 🔮 Dream Viewer - Peek Into Model's Mind

## Overview

The Dream Viewer is a revolutionary visualization system that lets you watch what the DreamingReasoningLLM is "thinking" in real-time during generation. It captures and displays the model's internal dream states as they evolve through the reasoning process.

## Features

### 1. **Live Dream Streaming**
Watch dreams unfold in real-time as the model generates output:
- Real-time console output of dream statistics
- Step-by-step progression tracking
- Graph reasoning visualization

### 2. **Dream Recording**
Automatically record all dream states for later analysis:
- JSON export format
- Text export format
- Complete state preservation

### 3. **Dream Analytics**
Get comprehensive statistics about dream evolution:
- Mean/std/min/max tracking
- Distribution analysis
- Temporal evolution patterns

### 4. **Visual Heatmaps**
Generate matplotlib heatmaps showing dream evolution:
- Mean value tracking
- Standard deviation patterns
- Min/max ranges over time
- Multi-panel visualization

## Quick Start

### Basic Usage

```python
from image_token_llm.dreaming_model import DreamingReasoningLLM

# Create model with dream viewer enabled
model = DreamingReasoningLLM(
    device="cuda",
    enable_dream_viewer=True  # Enable viewer!
)

# Watch dreams stream live during generation
result = model.generate(
    prompt="What happens when you drop a ball?",
    watch_dreams=True  # Enable live streaming
)
```

Output:
```
============================================================
🔮 LIVE DREAM STREAM - ACTIVE
============================================================

🌙 Dream Step 0
--------------------------------------------------
  Shape: [2, 3, 128]
  Mean:  0.0234
  Std:   0.9876
  Graph: 4 nodes, 6 edges
  Active: [0, 1, 2]...

🌙 Dream Step 1
--------------------------------------------------
  Shape: [2, 3, 128]
  Mean:  0.0456
  Std:   0.9543
  Graph: 8 nodes, 12 edges
  Active: [2, 3, 4]...

...

============================================================
✨ STREAM ENDED
============================================================
Captured 4 dream states
Avg Mean: 0.0345
Avg Std:  0.9712
```

### Export Dreams

```python
# Export recorded dreams to JSON
model.export_recorded_dreams("dreams.json", format="json")

# Export as human-readable text
model.export_recorded_dreams("dreams.txt", format="txt")
```

### Create Visualizations

```python
# Generate heatmap showing dream evolution
model.visualize_dream_evolution(save_path="dreams.png")
```

This creates a multi-panel visualization showing:
- Dream mean evolution
- Standard deviation changes
- Min/max ranges
- Combined heatmap

### Access Dream Data Programmatically

```python
# Get the dream viewer instance
viewer = model.get_dream_viewer()

# Get summary statistics
summary = viewer.get_dream_summary()
print(f"Total dreams: {summary['total_dreams']}")
print(f"Avg mean: {summary['avg_mean']:.4f}")
print(f"Mean range: {summary['mean_range']}")

# Access recorded dreams directly
for dream in viewer.recorded_dreams:
    print(f"Step {dream['step']}: mean={dream['dream_mean']:.4f}")

# Clear recorded dreams
viewer.clear_recording()
```

## Command-Line Demo

### Run Interactive Demo

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

# Custom prompt
python scripts/demo_dream_viewer.py \
    --prompt "How does photosynthesis work?" \
    --watch --export dreams.json --visualize evolution.png
```

## API Reference

### DreamViewer

Main class for dream visualization and recording.

**Methods:**

```python
capture_dream_state(
    step: int,
    dream_state: torch.Tensor,
    graph_state: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> Dict
```
Capture a single dream state snapshot.

```python
format_dream_state(
    snapshot: Dict,
    verbose: bool = True
) -> str
```
Format a dream snapshot as human-readable text.

```python
watch_dreams(
    dreams: List[Dict],
    interval: float = 0.5,
    live: bool = False
) -> None
```
Watch dream sequences unfold (can be live or batch).

```python
export_dreams(
    output_path: str,
    format: str = "json"
) -> None
```
Export recorded dreams to file (json or txt).

```python
get_dream_summary() -> Dict
```
Get summary statistics of recorded dreams.

```python
create_dream_heatmap(
    dream_sequence: List[Dict],
    save_path: Optional[str] = None
) -> Optional[str]
```
Create a heatmap visualization of dream evolution.

```python
clear_recording() -> None
```
Clear all recorded dreams.

### LiveDreamStream

Context manager for streaming dreams in real-time.

**Usage:**

```python
from image_token_llm.dream_viewer import DreamViewer, LiveDreamStream

viewer = DreamViewer(enable_recording=True)

with LiveDreamStream(viewer) as stream:
    # Dreams are automatically captured and displayed
    for step, dream_state in enumerate(dream_states):
        stream.stream_dream(
            dream_state=dream_state,
            step=step,
            metadata={"info": "example"}
        )
```

### DreamingReasoningLLM Integration

The model has built-in dream viewer support:

```python
# Enable viewer during initialization
model = DreamingReasoningLLM(
    enable_dream_viewer=True  # Enable viewing
)

# Generate with live streaming
result = model.generate(
    prompt="...",
    watch_dreams=True  # Stream dreams live
)

# Access viewer
viewer = model.get_dream_viewer()

# Export dreams
model.export_recorded_dreams("dreams.json")

# Visualize evolution
model.visualize_dream_evolution("dreams.png")
```

## Dream State Format

Each captured dream state contains:

```json
{
  "step": 0,
  "timestamp": 0,
  "dream_shape": [2, 5, 128],
  "dream_mean": 0.0234,
  "dream_std": 0.9876,
  "dream_min": -2.3456,
  "dream_max": 2.4567,
  "graph": {
    "num_nodes": 20,
    "num_edges": 40,
    "active_nodes": [0, 1, 2, 3, 4]
  },
  "metadata": {
    "prompt": "What happens when you drop a ball?"
  }
}
```

## Use Cases

### 1. **Debugging**
Understand why the model produces certain outputs by watching its reasoning process.

### 2. **Research**
Analyze how the model's internal representations evolve during different types of reasoning tasks.

### 3. **Education**
Demonstrate how neural reasoning works by visualizing the dream states.

### 4. **Monitoring**
Track model behavior in production to detect anomalies or unexpected patterns.

### 5. **Optimization**
Identify bottlenecks or inefficiencies in the reasoning process.

## Performance Notes

- **Recording Overhead**: Minimal (<1% slowdown) when enabled
- **Memory Usage**: ~50KB per dream state for typical models
- **Visualization**: Heatmaps require matplotlib (optional dependency)
- **Streaming**: Real-time display adds ~10ms per step

## Examples

### Example 1: Compare Different Prompts

```python
model = DreamingReasoningLLM(enable_dream_viewer=True)

prompts = [
    "What is 2+2?",
    "Explain quantum mechanics",
    "Write a poem about love"
]

for prompt in prompts:
    model.dream_viewer.clear_recording()
    
    result = model.generate(
        prompt=prompt,
        watch_dreams=True
    )
    
    summary = model.dream_viewer.get_dream_summary()
    print(f"\n{prompt}")
    print(f"  Dreams: {summary['total_dreams']}")
    print(f"  Avg Mean: {summary['avg_mean']:.4f}")
    print(f"  Complexity: {summary['std_range']}")
```

### Example 2: Track Learning Progress

```python
model = DreamingReasoningLLM(enable_dream_viewer=True)

for epoch in range(10):
    # Train the model...
    
    # Test and visualize
    result = model.generate(
        prompt="Test reasoning",
        watch_dreams=False
    )
    
    model.visualize_dream_evolution(
        save_path=f"epoch_{epoch}_dreams.png"
    )
    
    model.export_recorded_dreams(
        f"epoch_{epoch}_dreams.json"
    )
```

### Example 3: Real-Time Monitoring Dashboard

```python
import time

model = DreamingReasoningLLM(enable_dream_viewer=True)
viewer = model.get_dream_viewer()

while True:
    # Get user input
    prompt = input("Enter prompt: ")
    
    # Generate with live display
    result = model.generate(
        prompt=prompt,
        watch_dreams=True
    )
    
    # Show statistics
    summary = viewer.get_dream_summary()
    print(f"\nStats: {summary['total_dreams']} dreams, "
          f"mean={summary['avg_mean']:.4f}")
    
    time.sleep(1)
```

## Contributing

The dream viewer is modular and extensible. To add new visualization modes:

1. Extend `DreamViewer` class
2. Add new capture methods
3. Implement custom formatters
4. Create specialized visualizations

See `src/image_token_llm/dream_viewer.py` for implementation details.

## License

Same as DreamingReasoningLLM project.
