### Image Tokenizer (CLIP-based, Production Ready)

The `ImageTokenizer` now uses a pretrained CLIP vision encoder by default to produce semantic tokens for each image, with support for large batches, device selection, and production scaling.

- **Default:** CLIP-based semantic tokens (requires `transformers` and `torch`)
- **Legacy:** Patch-based (deprecated)

**Usage:**

```python
from image_token_llm.image_tokenizer import ImageTokenizer
from image_token_llm.config import ImageTokenizerConfig
import torch

config = ImageTokenizerConfig(embedding_dim=512, patch_size=16)
tokenizer = ImageTokenizer(config, device="cuda")  # Uses CLIP by default, on GPU
images = [torch.randn(3, 224, 224) for _ in range(128)]
tokens = tokenizer.tokenize(images, batch_size=32)  # Efficient large-batch
print(tokens[0].shape)  # (512,)
```

If you want to use a different backbone (e.g., ResNet or lite), pass `backbone="resnet"` or `backbone="lite"` to the constructor.

#### Dictionary Size and Scaling

The effective "dictionary size" (vocabulary of unique image tokens) is determined by the CLIP embedding space (default: 512-dim). For larger-scale or more granular tokenization, you can:
- Increase the embedding dimension (with a larger CLIP model)
- Use clustering or quantization on CLIP embeddings to build a discrete codebook
- Adjust batch size and device for high-throughput production
# Image-Token Reasoning LLM Playground

## Vision
Design a research-grade sandbox for an experimental large language model that reasons over **image-token triplets** (what + action = result), integrates a **graph-based retrieval-augmented generation (Graph RAG)** subsystem, and performs **parallel situation simulations** before committing to an answer.

## 🚀 Latest: Dreaming-Based Architecture (November 2025)

The project now features a **Dreaming-Based Reasoning LLM** (`DreamingReasoningLLM`) that:
- 🧠 **Reasons in visual space**: All thinking happens as sequences of image triplets
- 🎯 **MoE (Mixture of Experts)**: 4 specialized dream generators (spatial, temporal, causal, abstract)
- 🔄 **RL Integration**: Continuous learning from feedback with policy networks
- 🌐 **Graph Reasoning**: Multi-hop attention over dream sequences
- 📊 **Fully interpretable**: Visualize the model's reasoning process

See [DREAMING_ARCHITECTURE.md](docs/DREAMING_ARCHITECTURE.md) for details.

## Core Requirements
1. **Image Tokenization Pipeline**
   - Convert textual or multi-modal inputs into canonical image tokens representing `(what, action, result)` triplets.
   - Maintain a shared embedding space for downstream reasoning modules.
2. **Graph RAG Knowledge Fabric**
   - Persist entities, actions, and outcomes as typed nodes/edges.
   - Support neighborhood expansion, contextual walks, and subgraph sampling during inference.
3. **Parallel Situation Simulation**
   - Launch multiple hypothetical rollouts against the knowledge graph and generative policy network.
   - Track provenance for each simulation branch.
4. **Evaluation & Arbitration**
   - Score simulated trajectories with learned or rule-based evaluators.
   - Select the highest-confidence reasoning chain as the final inference output.
5. **GPT-5.1-Codex (Preview) Enablement**
   - Provide configuration hooks so every client session can toggle the preview-capable backend.
   - Default to the preview model for experiments while retaining a fallback path.

## Technical Approach
- **Primary Language:** Python 3.11+
- **Core Libraries (planned):** PyTorch, torchvision, networkx (or graph-tool), FAISS or similar vector index, Hydra/OmegaConf for configs.
- **Project Layout Goals:**
  - `src/` for modular components (tokenizer, graph_rag, simulator, evaluator, orchestration).
  - `tests/` for pytest suites covering pipelines and reasoning flows.
  - `notebooks/` for exploratory experiments.
  - `data/` for small fixture assets (placeholders only; real datasets supplied externally).

## Next Steps
1. Scaffold the workspace structure and dependency manifests.
2. Implement minimal viable modules and stubs.
3. Add automation (tasks, tests, launch configuration) and documentation.

_Assumptions:_ GPU acceleration is desirable but optional for initial scaffolding; external services (e.g., GPT-5.1-Codex preview) expose standard API credentials supplied outside this repository.

## Getting Started

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd vs

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### 2. Quick Start - Dreaming Model

```python
from image_token_llm.dreaming_model import DreamingReasoningLLM

# Create model with RL + MoE
model = DreamingReasoningLLM(device="cuda", enable_rl=True)

# Generate with visualization
result = model.generate(
    prompt="What happens when you drop a ball?",
    return_dreams=True
)

print(f"Output: {result['output']}")
print(f"Dreams generated: {len(result['dreams'])}")
print(f"Graph nodes: {result['graph_data']['num_nodes']}")
```

### 3. Use Legacy Pretrained Model
```bash
# Load and use pretrained ImageTokenReasoningLLM
python scripts/llm_chat.py --model ./pretrained_llama3/

# Or try a single prompt
python scripts/llm_chat.py \
    --model ./pretrained_llama3/ \
    --prompt "Explain photosynthesis" \
    --temp 0.7
```

### 3. Create Your Own Pretrained Model
```bash
# Requires Ollama running locally with llama3.2:1b
python scripts/create_pretrained_model.py \
    --teacher llama3.2:1b \
    --prompts 50 \
    --output ./my_model \
    --device cpu
```

### 4. Interactive Chat
```bash
# Start interactive session
python scripts/llm_chat.py --model ./pretrained_llama3/

# In chat mode, use commands:
#   /temp 0.9     - Set temperature
#   /tokens 150   - Set max tokens
#   /stream on    - Enable streaming
#   /help         - Show help
#   /quit         - Exit
```

### 5. Compare Models
```bash
# Compare two pretrained models
python scripts/compare_models.py \
    ./pretrained_llama3/ \
    ./pretrained_enhanced/ \
    --prompts 5
```

### 6. Run Tests
```bash
pytest
# or use VS Code task: "run tests"
```

## Usage Examples

### Python API
```python
from image_token_llm.model import ImageTokenReasoningLLM

# Load pretrained model
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./pretrained_llama3",
    device="cuda",  # or "cpu"
)

# Generate text
output = model.generate(
    prompt="What is quantum computing?",
    max_new_tokens=100,
    temperature=0.7,
)
print(output)

# Streaming generation
result = model.generate(
    prompt="Write a story",
    max_new_tokens=200,
    temperature=0.9,
    stream=True,
)
for chunk in result:
    print(chunk, end="", flush=True)
```

### Knowledge Distillation
```python
# Create new model from Ollama teacher
model = ImageTokenReasoningLLM(device="cuda")

prompts = [
    "What is AI?",
    "Explain machine learning",
    # ... more prompts
]

metrics = model.distill_from_ollama(
    prompts=prompts,
    teacher_model="llama3.2:1b",
)

# Export bundle
model.export_ollama_bundle(
    output_dir="./my_models",
    bundle_name="custom-llm",
)
```

See [USAGE_EXAMPLES.md](docs/USAGE_EXAMPLES.md) for comprehensive examples.

## Documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Beginner-friendly guide
- **[docs/USAGE_EXAMPLES.md](docs/USAGE_EXAMPLES.md)** - API patterns and examples
- **[docs/architecture.md](docs/architecture.md)** - Technical design details
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Development status

## Running the Reasoning Demo
- Launch the orchestrated simulation with configurable branch count:

   ```bash
   python -m image_token_llm.cli --branches 3
   ```

- The CLI seeds the Graph RAG with a minimal motif, tokenizes synthetic `(what, action, result)` tensors, fans out multiple simulation branches, and prints the scored arbitration report.

- Tweak runtime defaults (device selection, GPT-5.1-Codex preview toggle, simulator depth, etc.) in `configs/default.yaml`.

## Testing & Diagnostics
- Run the automated suite locally:

   ```bash
   pytest
   ```

- Inside VS Code, use the pre-created **run tests** task (`Terminal → Run Task → run tests`) for a one-click regression check.

- Add more cases under `tests/` as components mature (tokenizer fixtures, graph-walk verifications, evaluator scoreboards).

## Architecture Overview

### Multi-Modal Vision System
- **Vision Encoder**: ResNet/ViT-based encoder that transforms images into dense embeddings
- **Triplet Encoder**: Processes `(what, action, result)` image triplets with cross-attention fusion
- **Image Tokenization**: Converts visual concepts into tokens for reasoning

### Graph RAG with Attention
- **Graph Attention Networks**: Multi-head attention over knowledge graph neighborhoods
- **Dynamic Retrieval**: Context-aware subgraph extraction with multi-hop traversal
- **Embedding Propagation**: Node embeddings updated via message passing

### Reinforcement Learning Inference
- **Policy Network**: Learns to select optimal reasoning paths through graph traversal
- **Reward Model**: Scores reasoning trajectories for correctness and confidence
- **Online Learning**: Continuous adaptation during inference with policy gradients
- **Experience Replay**: Prioritized buffer for efficient learning from past interactions

### Knowledge Transfer from Ollama
- **Distillation Pipeline**: Extract knowledge from Ollama models (llama2, mistral, codellama)
- **Soft Targets**: Use teacher logits to train student encoder
- **Triplet Mining**: Generate synthetic training data from language model outputs

### Continuous Learning
- **Incremental Updates**: EWC-based learning to prevent catastrophic forgetting
- **Graph Evolution**: Dynamic knowledge graph updates from new observations
- **Meta-Learning**: Adapt quickly to new concepts with few examples

## Training the Model

### 1. Knowledge Transfer from Ollama

```bash
python -m image_token_llm.knowledge_transfer \
    --teacher_model llama2:7b \
    --output_dir ./checkpoints \
    --num_epochs 10 \
    --batch_size 16
```

### 2. Full Training with Open Datasets

```bash
python -m image_token_llm.trainer \
    --config configs/default.yaml \
    --dataset coco \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 3. Continuous Learning Mode

```python
from image_token_llm.rl_learning import RLContinuousLearner
from image_token_llm.vision_encoder import TripletEncoder
from image_token_llm.graph_attention import GraphRAGEnhanced

# Initialize components
encoder = TripletEncoder(config)
graph_rag = GraphRAGEnhanced(config)
reward_model = RewardModel()
policy_network = PolicyNetwork()

learner = RLContinuousLearner(encoder, graph_rag, reward_model, policy_network)

# Online inference with learning
prediction, metrics = learner.online_inference_with_learning(
    what_images, action_images, result_images,
    ground_truth_label=("cat", "chases", "mouse")
)

print(f"Reward: {metrics['final_reward']:.3f}")
print(f"Policy Loss: {metrics['policy_loss']:.3f}")
```

## Key Features

✅ **Multi-modal reasoning** - Think in images, not just text  
✅ **Graph-based memory** - Structured knowledge with relational reasoning  
✅ **RL-driven inference** - Learn optimal reasoning paths dynamically  
✅ **Continuous learning** - Always improving from new experiences  
✅ **Knowledge distillation** - Transfer knowledge from larger models  
✅ **Open datasets** - Train on COCO, Visual Genome, ConceptNet  

## Debug & Launch Guidance
- Use `src/image_token_llm/orchestrator.py` as the central composition point when stepping through simulations; attach a debugger to the CLI entrypoint for full reasoning traces.
- Hydra configs (`configs/`) let you pin seeds, branch factors, or switch between CPU/GPU without code changes.
- The `runtime.enable_gpt51_codex_preview` flag enables the GPT-5.1-Codex (Preview) backend for every client session; set it to `false` for fallback testing or when preview capacity is unavailable.
- Graph RAG artifacts (nodes, edges, embeddings) are abstracted via `graph_rag.py`; swap in a persistent backend by implementing the same interface and wiring it through the orchestrator constructor.
- For RL training, monitor `policy_loss`, `value_loss`, and `final_reward` metrics to track learning progress.
- Use `test_rl_learning.py` for unit tests of RL components without full data loading.

## Full Model Architecture

### Multi-Modal Components
1. **Vision Encoder** (`vision_encoder.py`)
   - CLIP-based transformer for semantic image understanding
   - Triplet encoder for (what, action, result) image sequences
   - Multi-head attention fusion across triplet components
   - Learnable role embeddings for component differentiation

2. **Graph Attention Network** (`graph_attention.py`)
   - Multi-layer graph attention for relational reasoning
   - Neighborhood aggregation with attention weights
   - Support for multi-hop traversal and subgraph extraction
   - Node and edge embedding management

3. **Text Generation** (`text_generation.py`)
   - Simple tokenizer with character-level fallback
   - Transformer decoder conditioned on image/graph context
   - Streaming generation support for real-time inference
   - Configurable temperature and top-k sampling

4. **Composite LLM** (`model.py`)
   - `ImageTokenReasoningLLM`: Unified model orchestrating all components
   - Integrates vision encoder, graph reasoning, RL learning, and text decoder
   - Methods for generation, distillation, export, and bundle loading
   - Supports online RL updates during inference

5. **Knowledge Transfer** (`knowledge_transfer.py`)
   - Distillation pipeline from Ollama models (llama2, mistral, phi)
   - Teacher-student architecture with feature alignment
   - Soft target learning with temperature scaling
   - Knowledge triplet extraction from teacher outputs

6. **Continuous Learning** (`continuous_learning.py`)
   - Experience replay buffer with prioritized sampling
   - Elastic Weight Consolidation (EWC) to prevent forgetting
   - Incremental graph knowledge updates
   - Online learning loop for streaming data

7. **Data Loaders** (`data_loaders.py`)
   - Visual Genome scene graph triplets
   - COCO captions adapted for triplet learning
   - ConceptNet knowledge graph integration
   - Synthetic data generation for testing

## Training the Model

### Prerequisites
1. Install Ollama (for knowledge distillation):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama2
   ```

2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Training
Train on synthetic data (no external dependencies):
```bash
python -m image_token_llm.trainer \
    --epochs 10 \
    --batch-size 16 \
    --dataset synthetic \
    --device cuda
```

### Training with Knowledge Distillation
Extract knowledge from Ollama teacher model:
```bash
python -m image_token_llm.trainer \
    --epochs 20 \
    --batch-size 8 \
    --dataset synthetic \
    --use-distillation \
    --teacher-model llama2 \
    --device cuda
```

### Training with Real Datasets
Use Visual Genome or COCO (requires HuggingFace datasets):
```bash
python -m image_token_llm.trainer \
    --epochs 50 \
    --batch-size 32 \
    --dataset visual_genome \
    --max-samples 10000 \
    --use-distillation \
    --device cuda
```

### Resume from Checkpoint
```bash
python -m image_token_llm.trainer \
    --resume checkpoints/epoch_10.pt \
    --epochs 20
```

## Continuous Learning Mode

The model supports online learning from streaming experiences:

```python
from image_token_llm.continuous_learning import IncrementalLearner, OnlineLearningLoop
from image_token_llm.vision_encoder import TripletEncoder
from image_token_llm.graph_attention import GraphRAGEnhanced

# Initialize components
encoder = TripletEncoder(config)
graph = GraphRAGEnhanced(graph_config)
learner = IncrementalLearner(encoder, graph)
online_loop = OnlineLearningLoop(learner)

# Process streaming data
for what_img, action_img, result_img, label in data_stream:
    metrics = online_loop.step(what_img, action_img, result_img, label)
```

## Architecture Highlights

### Image-Token Reasoning
- Input text/images → Vision encoder → Semantic tokens
- Tokens represent (what, action, result) causal triplets
- Graph stores relationships between concepts
- Multi-hop attention reasoning over knowledge graph

### Knowledge Transfer
- Teacher: Ollama open-source models (llama2, mistral, phi, etc.)
- Student: Our lightweight multi-modal architecture
- Distillation: Soft targets + feature matching + hard labels
- Continuous extraction of new knowledge from teacher

### Always Learning
- Experience buffer with prioritized replay
- Elastic Weight Consolidation prevents catastrophic forgetting
- Incremental graph updates as new triplets are observed
- Online adaptation to distribution shifts

## Model Outputs
- **Embeddings**: Dense vectors for images, triplets, and graph nodes
- **Reasoning Chains**: Multi-step graph traversals with attention scores
- **Triplet Predictions**: Generate (subject, relation, object) from images
- **Knowledge Graph**: Dynamic graph of learned concepts and relations
- **Text Generation**: Natural language output conditioned on visual/graph context
- **Ollama Bundles**: Export complete model packages for deployment

## Export & Deployment

### Export to Ollama-Compatible Bundle
```python
from image_token_llm.model import ImageTokenReasoningLLM

model = ImageTokenReasoningLLM()
# ... train or load weights ...
bundle_path = model.export_ollama_bundle(
    output_dir="./ollama_bundles",
    bundle_name="my-image-llm"
)
print(f"Bundle exported to: {bundle_path}")
```

### Load from Bundle
```python
model = ImageTokenReasoningLLM.load_from_bundle(
    bundle_dir="./ollama_bundles/my-image-llm",
    device="cuda"
)
result = model.generate("Describe the scene", image_triplets=triplets)
```

### Quick-Start Distillation
Create a pretrained model using knowledge from an Ollama teacher:
```bash
python scripts/create_pretrained_model.py \
    --teacher llama2 \
    --prompts 100 \
    --output ./pretrained \
    --device cuda
```

## Performance Monitoring
Training metrics logged to TensorBoard:
```bash
tensorboard --logdir runs/multimodal_training
```

Metrics tracked:
- Reconstruction loss (triplet coherence)
- Distillation loss (teacher-student alignment)
- EWC loss (forgetting prevention)
- Graph reasoning accuracy
- Online learning adaptation rate
- Text generation perplexity
- RL reward signals
