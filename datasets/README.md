# Distillation Datasets Cache

This directory contains cached prompt-response pairs generated from Ollama teacher models for knowledge distillation into DreamingReasoningLLM.

## Purpose

Caching datasets allows:
- **Resumable generation**: Continue if interrupted
- **Reusable data**: Train multiple models on same dataset
- **Faster experiments**: No need to regenerate from Ollama
- **Reproducibility**: Same dataset across runs
- **Sharing**: Transfer datasets between machines

## File Naming Convention

`{teacher_model}_{num_pairs}.json`

Examples:
- `llama3.2_1b_100000.json` - 100,000 pairs from llama3.2:1b
- `qwen2-5-coder-1-5b_50000.json` - 50,000 pairs from qwen2-5-coder-1-5b

## Dataset Format

```json
[
  {
    "prompt": "What is gravity?",
    "response": "Gravity is a fundamental force of nature...",
    "index": 0
  },
  {
    "prompt": "Explain photosynthesis.",
    "response": "Photosynthesis is the process by which plants...",
    "index": 1
  }
]
```

## Usage

### Generate New Dataset

```bash
python scripts/distill_large_scale.py \
    --teacher llama3.2:1b \
    --num-pairs 100000 \
    --cache-dir ./datasets/distillation_cache
```

### Resume from Cache

```bash
# Automatically resumes if cache exists
python scripts/distill_large_scale.py \
    --teacher llama3.2:1b \
    --num-pairs 100000
```

### Start Fresh (Ignore Cache)

```bash
python scripts/distill_large_scale.py \
    --teacher llama3.2:1b \
    --num-pairs 100000 \
    --no-resume
```

## Checkpointing

The system automatically saves checkpoints every 1,000 pairs during generation. This means:
- If interrupted, at most 1,000 pairs need to be regenerated
- Large datasets can be generated over multiple sessions
- Network failures won't lose all progress

## Disk Space

Approximate sizes:
- 1,000 pairs: ~2 MB
- 10,000 pairs: ~20 MB
- 100,000 pairs: ~200 MB
- 1,000,000 pairs: ~2 GB

## Loading Datasets Programmatically

```python
import json
from pathlib import Path

# Load cached dataset
cache_path = Path("./datasets/distillation_cache/llama3.2_1b_100000.json")
with open(cache_path, "r") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} pairs")

# Access individual pairs
for item in dataset[:5]:
    print(f"Prompt: {item['prompt']}")
    print(f"Response: {item['response'][:100]}...")
    print()
```

## Maintenance

### Check Dataset Size

```bash
# Count pairs in dataset
cat llama3.2_1b_100000.json | jq '. | length'

# Check file size
ls -lh llama3.2_1b_100000.json
```

### Validate Dataset

```python
import json

with open("llama3.2_1b_100000.json", "r") as f:
    data = json.load(f)

# Check all entries have required fields
for i, item in enumerate(data):
    assert "prompt" in item, f"Entry {i} missing prompt"
    assert "response" in item, f"Entry {i} missing response"
    assert "index" in item, f"Entry {i} missing index"

print(f"✓ Dataset valid: {len(data)} pairs")
```

### Remove Old Caches

```bash
# Remove all cached datasets
rm -rf datasets/distillation_cache/*.json

# Remove specific cache
rm datasets/distillation_cache/llama3.2_1b_100000.json
```

## Best Practices

1. **Keep caches organized**: Use descriptive names and document what each dataset contains

2. **Version control**: For important datasets, consider:
   - Saving to separate location
   - Compressing for storage
   - Documenting generation parameters

3. **Backup large datasets**: 100k+ pair datasets take hours/days to generate, so back them up

4. **Share datasets**: Cached datasets can be shared with team members to avoid redundant generation

5. **Monitor disk space**: Large datasets can consume significant disk space

## Current Datasets

<!-- Auto-updated by training scripts -->
- `llama3.2_1b_100000.json` - 100,000 diverse prompt-response pairs
  - Teacher: llama3.2:1b
  - Topics: Science, History, Geography, Technology, Arts, Everyday
  - Generation date: 2025-11-17
  - Status: In progress...

## Notes

- Datasets are specific to the teacher model used
- Different models may give different responses to same prompts
- Cache files are plain JSON for easy inspection and manipulation
- Large caches (1M+ pairs) may require streaming or chunked processing
