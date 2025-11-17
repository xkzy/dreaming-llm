# Image-Token Reasoning LLM - Development Guide

## Project Overview
Multi-modal LLM combining vision encoding, graph reasoning, RL continuous learning, and text generation with Ollama integration.

## Key Components
- **Vision Encoder** (`vision_encoder.py`): CLIP/ResNet/lite backbones for image triplet encoding
- **Graph RAG** (`graph_attention.py`, `graph_rag.py`): Multi-hop attention over knowledge graphs
- **RL Learning** (`rl_learning.py`): Policy networks, reward models, online learning
- **Text Generation** (`text_generation.py`): Transformer decoder with streaming support
- **Composite Model** (`model.py`): `ImageTokenReasoningLLM` orchestrating all components
- **Knowledge Transfer** (`knowledge_transfer.py`): Distillation from Ollama teachers
- **Continuous Learning** (`continuous_learning.py`): EWC, experience replay, incremental updates

## Quick Start Commands

### Run Tests
```bash
pytest
# or use VS Code task: "run tests"
```

### Create Pretrained Model
```bash
python scripts/create_pretrained_model.py \
    --teacher llama2 \
    --prompts 100 \
    --output ./pretrained \
    --device cuda
```

### Run Online RL Demo
```bash
python examples/online_rl_inference.py
```

### Use the Model
```python
from image_token_llm.model import ImageTokenReasoningLLM

# Initialize or load
model = ImageTokenReasoningLLM(device="cuda")
# or: model = ImageTokenReasoningLLM.load_from_bundle("./pretrained")

# Generate text
output = model.generate(
    prompt="Describe the scene",
    image_triplets=[(what_img, action_img, result_img)],
    temperature=0.8,
    stream=False
)

# Distill knowledge
metrics = model.distill_from_ollama(
    prompts=["What is AI?", "Explain machine learning."],
    teacher_model="llama2"
)

# Export bundle
model.export_ollama_bundle(
    output_dir="./my_bundle",
    bundle_name="my-model"
)
```

## Architecture Notes
- Image triplets: `(what, action, result)` tensors shape `(B, 3, H, W)`
- Graph traversal uses attention scores for neighborhood selection
- RL updates happen inline during `generate()` when `enable_rl=True`
- Text decoder is conditioned on fused image+graph memory
- Ollama bundles include weights, tokenizer, config, and Modelfile

## Development Guidelines

### Code Style
- Python 3.11+ with type hints
- Line length limit: 79 characters
- Use spaces (4) not tabs
- Follow PEP 8 conventions
- Type annotations for public APIs

### Testing
- All new features require tests in `tests/`
- Run `pytest` before committing
- Mock external services (Ollama) for unit tests
- Use fixtures for common test data

### Module Dependencies
- `vision_encoder.py` → standalone (optional transformers)
- `graph_rag.py` → networkx, torch
- `text_generation.py` → torch only
- `model.py` → imports all above modules
- `knowledge_transfer.py` → requests (Ollama API)

### Common Patterns
- Use `ExperimentConfig` for all configurable parameters
- Device handling: check CUDA availability, fallback to CPU
- Lazy imports for optional dependencies (e.g., CLIP)
- Return metrics dicts from training/inference methods
- Save/load via state_dict for PyTorch modules

### Debugging Tips
- Set `enable_rl=False` for faster model initialization
- Use `vision_backbone="lite"` to avoid downloading large models
- Check `model.last_metadata` for generation stats
- Monitor `policy_loss`, `value_loss`, `reward` for RL convergence
- TensorBoard logs in `runs/` directory

### Known Limitations
- Text tokenizer is character-level (simple but slow)
- Graph reasoning requires seeding with initial nodes
- RL updates may be unstable with small batch sizes
- Ollama distillation requires Ollama server running locally
