# Image-Token Reasoning LLM - Usage Examples

This document provides practical examples for using the Image-Token Reasoning LLM.

## Table of Contents
1. [Basic Setup](#basic-setup)
2. [Text Generation](#text-generation)
3. [Knowledge Distillation](#knowledge-distillation)
4. [Model Export/Import](#model-exportimport)
5. [Graph Reasoning](#graph-reasoning)
6. [RL Learning](#rl-learning)
7. [Advanced Configuration](#advanced-configuration)

## Basic Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Load Pretrained Model
```python
from image_token_llm.model import ImageTokenReasoningLLM

# Load from bundle
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./pretrained_llama3",
    device="cuda",  # or "cpu"
    enable_rl=False,  # Set True for RL features
)
```

### Create New Model
```python
from image_token_llm.config import ExperimentConfig
from image_token_llm.model import ImageTokenReasoningLLM

config = ExperimentConfig()
model = ImageTokenReasoningLLM(
    config=config,
    device="cuda",
    vision_backbone="lite",  # or "clip", "resnet"
    enable_rl=True,
)
```

## Text Generation

### Simple Generation
```python
output = model.generate(
    prompt="Explain quantum computing",
    max_new_tokens=100,
    temperature=0.8,
)
print(output)
```

### Streaming Generation
```python
result = model.generate(
    prompt="Write a story about space exploration",
    max_new_tokens=200,
    temperature=0.9,
    stream=True,
)

for chunk in result:
    print(chunk, end="", flush=True)
print()
```

### With Custom Temperature
```python
# More creative (higher temperature)
creative_output = model.generate(
    prompt="Imagine a future city",
    temperature=1.2,
    max_new_tokens=150,
)

# More deterministic (lower temperature)
factual_output = model.generate(
    prompt="What is photosynthesis?",
    temperature=0.3,
    max_new_tokens=100,
)
```

## Knowledge Distillation

### Distill from Ollama Teacher
```python
# Prepare training prompts
prompts = [
    "What is artificial intelligence?",
    "Explain machine learning",
    "How does neural network work?",
    # ... add more prompts
]

# Distill knowledge
metrics = model.distill_from_ollama(
    prompts=prompts,
    teacher_model="llama3.2:1b",
    num_samples=1,
)

print(f"Distillation loss: {metrics['distillation_loss']:.4f}")
print(f"Traces processed: {metrics['traces']}")
```

### Using Script
```bash
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 100 \
    --output ./my_pretrained \
    --device cuda \
    --no-rl
```

## Model Export/Import

### Export Bundle
```python
bundle_path = model.export_ollama_bundle(
    output_dir="./my_models",
    bundle_name="custom-image-llm",
)
print(f"Model exported to: {bundle_path}")
```

### Load Bundle
```python
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./my_models/custom-image-llm",
    device="cuda",
)
```

### Bundle Contents
```
my-models/custom-image-llm/
├── custom-image-llm_weights.pt  # PyTorch weights
├── tokenizer.json                # Tokenizer vocabulary
├── config.json                   # Model configuration
├── Modelfile                     # Ollama deployment template
└── README.md                     # Usage instructions
```

## Graph Reasoning

### Add Knowledge Triplets
```python
from image_token_llm.graph_rag import Triplet

# Add knowledge to graph
triplets = [
    Triplet("Paris", "capital_of", "France"),
    Triplet("France", "located_in", "Europe"),
    Triplet("Eiffel_Tower", "located_in", "Paris"),
]

model.ingest_graph(triplets)
```

### Generate with Graph Context
```python
output = model.generate(
    prompt="Tell me about Paris",
    seeds=["Paris", "France", "Eiffel_Tower"],
    max_new_tokens=100,
)

# Check metadata
print(f"Seeds used: {model.last_metadata['seeds']}")
print(f"Graph nodes: {model.last_metadata['num_graph_nodes']}")
```

## RL Learning

### Enable RL During Generation
```python
import torch

# Create dummy image triplets (in practice, load real images)
what_img = torch.randn(3, 224, 224)
action_img = torch.randn(3, 224, 224)
result_img = torch.randn(3, 224, 224)

# Initialize with RL
model = ImageTokenReasoningLLM(
    device="cuda",
    enable_rl=True,
)

# Generate with RL updates
output = model.generate(
    prompt="Analyze this sequence",
    image_triplets=[(what_img, action_img, result_img)],
    max_new_tokens=50,
)

# Check RL metrics
if model.last_metadata.get("rl_metrics"):
    print("RL Metrics:")
    for key, value in model.last_metadata["rl_metrics"].items():
        print(f"  {key}: {value:.4f}")
```

### Online RL Learning
```python
from image_token_llm.rl_learning import RLContinuousLearner

# Setup RL components
learner = model.rl_learner

# Continuous learning loop
for what, action, result in data_stream:
    prediction, metrics = learner.online_inference_with_learning(
        what, action, result
    )
    
    print(f"Reward: {metrics['final_reward']:.3f}")
    print(f"Policy loss: {metrics['policy_loss']:.3f}")
```

## Advanced Configuration

### Custom Configuration
```python
from image_token_llm.config import (
    ExperimentConfig,
    TextDecoderConfig,
    TokenizerConfig,
    GraphRAGConfig,
)

# Create custom config
config = ExperimentConfig(
    text_decoder=TextDecoderConfig(
        vocab_size=8192,
        max_seq_len=512,
        num_layers=6,
        num_heads=8,
    ),
    tokenizer=TokenizerConfig(
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    ),
    graph_rag=GraphRAGConfig(
        top_k_neighbors=10,
        max_hops=4,
    ),
)

model = ImageTokenReasoningLLM(config=config)
```

### Check Model Metadata
```python
# After generation
print("Last Generation Metadata:")
print(f"  Seeds: {model.last_metadata.get('seeds', [])}")
print(f"  RL metrics: {model.last_metadata.get('rl_metrics', {})}")
print(f"  Graph nodes: {model.last_metadata.get('num_graph_nodes', 0)}")
```

### Save/Load Weights Manually
```python
# Save
torch.save({
    'model_state': model.state_dict(),
    'config': model.config.model_dump(),
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model = ImageTokenReasoningLLM(
    config=ExperimentConfig(**checkpoint['config'])
)
model.load_state_dict(checkpoint['model_state'])
```

## Practical Examples

### Question Answering System
```python
def qa_system(question: str) -> str:
    """Simple QA system using the LLM."""
    model = ImageTokenReasoningLLM.load_from_bundle("./pretrained_llama3")
    
    answer = model.generate(
        prompt=f"Question: {question}\nAnswer:",
        max_new_tokens=100,
        temperature=0.5,
    )
    
    return answer

# Use it
answer = qa_system("What causes rain?")
print(answer)
```

### Batch Processing
```python
questions = [
    "What is DNA?",
    "How do vaccines work?",
    "Explain climate change",
]

model = ImageTokenReasoningLLM.load_from_bundle("./pretrained_llama3")

for q in questions:
    answer = model.generate(q, max_new_tokens=80)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
```

### Interactive Chat
```python
model = ImageTokenReasoningLLM.load_from_bundle("./pretrained_llama3")

print("Chat with the AI (type 'quit' to exit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    response = model.generate(user_input, max_new_tokens=100)
    print(f"AI: {response}")
```

## Performance Tips

1. **Use GPU for faster inference**
   ```python
   model = ImageTokenReasoningLLM.load_from_bundle(
       bundle_dir="./model",
       device="cuda"  # 10-50x faster than CPU
   )
   ```

2. **Disable RL when not needed**
   ```python
   # Faster initialization
   model = ImageTokenReasoningLLM(enable_rl=False)
   ```

3. **Use lite vision backbone for speed**
   ```python
   model = ImageTokenReasoningLLM(vision_backbone="lite")
   ```

4. **Batch multiple prompts**
   - Process similar prompts together
   - Reuse model instance
   - Avoid reloading

5. **Tune temperature**
   - Lower (0.1-0.5) for factual/deterministic
   - Higher (0.8-1.2) for creative
   - Very high (>1.5) for randomness

## Troubleshooting

### Import Errors
```python
# Make sure package is installed
import sys
sys.path.insert(0, '/path/to/vs/src')

from image_token_llm.model import ImageTokenReasoningLLM
```

### CUDA Out of Memory
```python
# Use CPU or smaller model
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./model",
    device="cpu",
    enable_rl=False,
)
```

### Slow Generation
- Use GPU
- Reduce max_new_tokens
- Disable RL features
- Use lite vision backbone

For more examples, see the `scripts/` directory.
