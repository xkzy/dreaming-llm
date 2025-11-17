# Image-Token Reasoning LLM - Quick Start Guide

## ✅ Project Status: Complete & Ready

All core components have been implemented and tested:

- ✅ Vision encoding (triplet images)
- ✅ Graph-based reasoning with attention
- ✅ RL continuous learning
- ✅ Text generation with transformer decoder
- ✅ Knowledge transfer from Ollama models
- ✅ Model export/import (Ollama bundles)
- ✅ Pretrained model creation script
- ✅ Tests passing
- ✅ Documentation updated

## Quick Start

### 1. Run Tests
```bash
pytest
# or
python -m pytest tests/
```

### 2. Create a Pretrained Model
```bash
# Using an available Ollama model
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 100 \
    --output ./my_pretrained \
    --device cuda \
    --no-rl
```

**Note**: Make sure Ollama is installed and the teacher model is available:
```bash
ollama pull llama3.2:1b
```

### 3. Test the Pretrained Model
```bash
python scripts/test_pretrained.py ./my_pretrained \
    --prompt "What is machine learning?" \
    --max-tokens 50
```

### 4. Use the Model in Code
```python
from image_token_llm.model import ImageTokenReasoningLLM
import torch

# Load from bundle
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./my_pretrained",
    device="cuda",
)

# Generate text
output = model.generate(
    prompt="Describe the process of learning",
    max_new_tokens=100,
    temperature=0.8,
)
print(output)

# With image triplets (if available)
# what_img, action_img, result_img = load_images()
# output = model.generate(
#     prompt="What happened?",
#     image_triplets=[(what_img, action_img, result_img)],
#     temperature=0.7,
# )
```

## Architecture Highlights

### Components
1. **Vision Encoder** - Processes (what, action, result) image triplets
2. **Graph RAG** - Knowledge graph with attention-based traversal
3. **Text Decoder** - Transformer conditioned on visual/graph context
4. **RL Learner** - Online learning during inference
5. **Knowledge Transfer** - Distillation from Ollama teachers

### Key Methods
- `generate()` - Text generation with optional streaming
- `distill_from_ollama()` - Learn from teacher models
- `export_ollama_bundle()` - Save complete model package
- `load_from_bundle()` - Restore from bundle

## Training Tips

### Start Small
- Use `--no-rl` flag for faster initialization during distillation
- Start with 10-50 prompts to test the pipeline
- Use CPU for small experiments, GPU for production

### Scale Up
```bash
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 1000 \
    --samples-per-prompt 2 \
    --output ./production_model \
    --device cuda \
    --vision-backbone clip
```

### Monitor Progress
- Distillation loss should decrease over prompts
- Check `model.last_metadata` for generation stats
- Use TensorBoard for detailed metrics (coming soon)

## Model Outputs

The model currently produces:
- Text tokens from character-level tokenizer
- Graph reasoning traces (in metadata)
- RL reward signals (when enabled)
- Attention-weighted knowledge retrieval

**Note**: For coherent text output, train with 100+ diverse prompts. The model
learns token patterns from the teacher through distillation.

## Next Steps

1. **More Training Data**: Increase prompts to 500-1000 for better quality
2. **Better Tokenizer**: Switch to BPE/WordPiece for real text
3. **Visual Training**: Add image-text datasets (COCO, Visual Genome)
4. **RL Fine-tuning**: Enable RL for task-specific adaptation
5. **Ollama Integration**: Deploy as Ollama-compatible server

## Files Created

- `scripts/create_pretrained_model.py` - Distillation script
- `scripts/test_pretrained.py` - Bundle testing utility
- `src/image_token_llm/model.py` - Composite LLM class
- `src/image_token_llm/text_generation.py` - Tokenizer and decoder
- Updated `README.md` and `.github/copilot-instructions.md`

## Troubleshooting

### "model not found" error
```bash
ollama list  # Check available models
ollama pull llama3.2:1b  # Download if needed
```

### IndexError during distillation
Fixed! Tokens are now clamped to valid vocabulary range.

### Out of memory
- Use `--device cpu` for testing
- Reduce `--prompts` count
- Use `--no-rl` to disable RL components

## Example Session
```
$ python scripts/create_pretrained_model.py --teacher llama3.2:1b --prompts 5 --no-rl
[INFO] Initializing model...
[INFO] Model initialized successfully
[INFO] Starting distillation from llama3.2:1b...
Generating traces: 100%|████████████| 5/5 [00:30<00:00]
[INFO] Distillation complete!
[INFO]   Distillation loss: 6.4899
[INFO]   Traces processed: 5
[INFO] Bundle exported successfully to: pretrained_llama3
```

## Success! 🎉

Your Image-Token Reasoning LLM is ready to use. Check the README.md for 
advanced usage, architecture details, and deployment options.
