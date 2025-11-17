# Complete Workflow Guide

This guide walks through the complete workflow from installation to deployment.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Creating a Pretrained Model](#creating-a-pretrained-model)
3. [Testing Your Model](#testing-your-model)
4. [Using the Model](#using-the-model)
5. [Advanced Training](#advanced-training)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## 1. Environment Setup

### Prerequisites
- Python 3.11 or higher
- (Optional) CUDA for GPU acceleration
- (Required for distillation) Ollama with llama3.2:1b

### Install Ollama
```bash
# On Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Pull the teacher model
ollama pull llama3.2:1b
```

### Setup Project
```bash
# Clone repository
git clone <repo-url>
cd vs

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
pytest
```

---

## 2. Creating a Pretrained Model

### Quick Model (5 prompts, ~30 seconds)
```bash
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 5 \
    --output ./models/quick \
    --device cpu \
    --no-rl
```

### Production Model (100+ prompts, ~10-15 minutes)
```bash
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 100 \
    --output ./models/production \
    --device cuda \
    --no-rl \
    --bundle-name my-production-llm
```

### Custom Prompts
Create a file `my_prompts.txt`:
```
What is artificial intelligence?
Explain quantum physics
How do neural networks work?
...
```

Then use it:
```bash
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts-file my_prompts.txt \
    --output ./models/custom \
    --device cpu
```

---

## 3. Testing Your Model

### Test Script
```bash
# Quick validation
python scripts/test_pretrained.py ./models/production/

# Expected output:
# ✓ Model loaded successfully
# Testing generation...
# [Generated text appears here]
```

### Benchmark Performance
```bash
# Test on 10 diverse prompts
python scripts/benchmark_model.py ./models/production/

# View results
# Average time per generation: 0.XX seconds
# Average output length: XX characters
```

### Compare Models
```bash
# Side-by-side comparison
python scripts/compare_models.py \
    ./models/quick/ \
    ./models/production/ \
    --prompts 10

# Results saved to model_comparison_results.json
```

---

## 4. Using the Model

### Command-Line Chat
```bash
# Start interactive chat
python scripts/llm_chat.py --model ./models/production/

# Chat commands:
You: /temp 0.9        # Set temperature
You: /tokens 200      # Set max tokens
You: /stream on       # Enable streaming
You: Hello, how are you?
AI: [Response appears here]
```

### Single Prompt
```bash
python scripts/llm_chat.py \
    --model ./models/production/ \
    --prompt "Explain photosynthesis" \
    --temp 0.7 \
    --tokens 150
```

### Python Script
Create `my_script.py`:
```python
from image_token_llm.model import ImageTokenReasoningLLM

# Load model
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./models/production",
    device="cuda",
)

# Generate
prompts = [
    "What is DNA?",
    "How does rain form?",
    "Explain gravity",
]

for prompt in prompts:
    output = model.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.7,
    )
    print(f"Q: {prompt}")
    print(f"A: {output}\n")
```

Run it:
```bash
python my_script.py
```

---

## 5. Advanced Training

### Incremental Training
```python
# Load existing model
model = ImageTokenReasoningLLM.load_from_bundle("./models/production")

# Add more training
new_prompts = [
    "What is blockchain?",
    "Explain climate change",
    # ... 50 more prompts
]

metrics = model.distill_from_ollama(
    prompts=new_prompts,
    teacher_model="llama3.2:1b",
)

# Export updated model
model.export_ollama_bundle(
    output_dir="./models/production_v2",
    bundle_name="my-llm-v2",
)
```

### Multi-Teacher Distillation
```python
# Train with multiple teachers
teachers = ["llama3.2:1b", "phi3", "mistral"]

model = ImageTokenReasoningLLM(device="cuda")

for teacher in teachers:
    print(f"Learning from {teacher}...")
    metrics = model.distill_from_ollama(
        prompts=domain_prompts,
        teacher_model=teacher,
    )

model.export_ollama_bundle(output_dir="./models/multi_teacher")
```

### Domain-Specific Fine-Tuning
```python
# Medical domain example
medical_prompts = [
    "What are the symptoms of diabetes?",
    "Explain how vaccines work",
    "What is MRI imaging?",
    # ... 100+ medical prompts
]

model = ImageTokenReasoningLLM(device="cuda")
model.distill_from_ollama(
    prompts=medical_prompts,
    teacher_model="llama3.2:1b",
)

model.export_ollama_bundle(
    output_dir="./models/medical_llm",
    bundle_name="medical-assistant",
)
```

---

## 6. Deployment

### Option 1: Direct Python Integration
```python
# In your application
from image_token_llm.model import ImageTokenReasoningLLM

class AIAssistant:
    def __init__(self):
        self.model = ImageTokenReasoningLLM.load_from_bundle(
            bundle_dir="/path/to/model",
            device="cuda",
        )
    
    def answer(self, question: str) -> str:
        return self.model.generate(
            prompt=question,
            max_new_tokens=150,
            temperature=0.7,
        )
```

### Option 2: REST API
Create `api_server.py`:
```python
from fastapi import FastAPI
from pydantic import BaseModel
from image_token_llm.model import ImageTokenReasoningLLM

app = FastAPI()
model = ImageTokenReasoningLLM.load_from_bundle("./models/production")

class Query(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
def generate(query: Query):
    output = model.generate(
        prompt=query.prompt,
        max_new_tokens=query.max_tokens,
        temperature=query.temperature,
    )
    return {"output": output}
```

Run with:
```bash
pip install fastapi uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is AI?", "max_tokens": 100}'
```

### Option 3: Ollama Integration (Future)
The exported bundle includes a `Modelfile` for Ollama:
```bash
# Copy bundle to Ollama models directory
cp -r ./models/production ~/.ollama/models/my-llm

# Import to Ollama
ollama create my-llm -f ~/.ollama/models/my-llm/Modelfile

# Use with Ollama
ollama run my-llm "What is quantum computing?"
```

---

## 7. Troubleshooting

### Problem: Model generates gibberish
**Solution:**
- Train with more prompts (100+ recommended)
- Lower temperature (0.3-0.5 for factual content)
- Check tokenizer vocabulary size

```python
# Retrain with more data
python scripts/create_pretrained_model.py \
    --prompts 200 \
    --teacher llama3.2:1b \
    --output ./models/improved
```

### Problem: Slow generation
**Solution:**
- Use GPU instead of CPU
- Reduce max_new_tokens
- Disable RL features

```python
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./model",
    device="cuda",  # Use GPU
    enable_rl=False,  # Disable RL
)
```

### Problem: CUDA out of memory
**Solution:**
- Switch to CPU
- Use smaller batch sizes
- Close other GPU applications

```python
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./model",
    device="cpu",
)
```

### Problem: Ollama connection error
**Solution:**
```bash
# Check Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Test connection
curl http://localhost:11434/api/generate \
    -d '{"model": "llama3.2:1b", "prompt": "test"}'
```

### Problem: Import errors
**Solution:**
```bash
# Reinstall package
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/vs/src"
```

---

## Best Practices

### 1. Training
- Start with 50-100 prompts for baseline
- Use diverse prompts covering different topics
- Train on domain-specific data for specialized models
- Save checkpoints regularly

### 2. Generation
- Use temperature 0.3-0.5 for factual content
- Use temperature 0.8-1.2 for creative content
- Limit max_tokens to avoid long, unfocused outputs
- Test streaming for better UX

### 3. Deployment
- Load model once at startup (expensive operation)
- Reuse model instance for multiple generations
- Monitor memory usage
- Implement request queuing for high load

### 4. Monitoring
```python
# Track generation metadata
output = model.generate(prompt="test")
print(model.last_metadata)
# Shows: seeds, tokens, time, etc.
```

---

## Performance Targets

### Training
| Prompts | Time (CPU) | Time (GPU) | Quality |
|---------|------------|------------|---------|
| 5       | 30s        | 10s        | Basic   |
| 50      | 5min       | 2min       | Good    |
| 100     | 10min      | 4min       | Better  |
| 500+    | 50min      | 20min      | Best    |

### Generation
| Device | Init Time | Per Token | Per Response (100 tokens) |
|--------|-----------|-----------|---------------------------|
| CPU    | 0.5s      | 0.3s      | 30s                       |
| GPU    | 0.2s      | 0.03s     | 3s                        |

---

## Next Steps

1. **Complete the tutorial:**
   - Train a 100-prompt model
   - Run benchmarks
   - Compare with baseline

2. **Build an application:**
   - REST API server
   - Chat interface
   - Domain-specific assistant

3. **Improve the model:**
   - Better tokenizer (BPE)
   - More training data
   - Fine-tuning

4. **Share your work:**
   - Export bundle
   - Document your process
   - Contribute improvements

---

## Additional Resources

- **Documentation:** [docs/](docs/)
- **Examples:** [examples/](examples/)
- **Scripts:** [scripts/](scripts/)
- **Tests:** [tests/](tests/)

For questions or issues, see [PROJECT_STATUS.md](docs/PROJECT_STATUS.md).
