# Changelog

All notable changes to the Image-Token Dreaming LLM project.

## [2.0.0] - 2025-11-17

### 🎉 Major Release: Dreaming-Based Architecture

Complete redesign of the reasoning system with Mixture of Experts, Graph Transformers, and PPO-based RL.

### Added

#### Core Architecture
- **DreamingReasoningLLM**: New main model class with visual reasoning
- **MoE Dream Generator**: 4 specialized experts (spatial, temporal, causal, abstract)
- **Graph Transformer**: Multi-hop attention with learned edge types
- **PPO Trainer**: Full reinforcement learning implementation
- **Multi-Component Rewards**: 4-head reward model (faithfulness, coherence, correctness, creativity)

#### Input/Tokenization Improvements
- CLIP-style contrastive loss with learned temperature (τ=0.07)
- Vision Transformer (ViT) backbone with 3-level feature pyramids
- Projection bottleneck (D→D/2) for regularization
- Noise robustness: 15% dropout + 10% token masking

#### MoE Enhancements
- Noisy top-k gating (k=2) inspired by Switch Transformers
- Transformer expert blocks (2 layers, 8 heads, pre-norm)
- Cross-expert attention for information sharing
- Load balancing loss to prevent expert collapse

#### Graph Reasoning Upgrades
- Learned edge predictor with 4 relation types
- Graph Transformer layers with edge-aware attention
- Node memory states using GRU cells
- Multi-head attention (8 heads) for relational reasoning

#### Training Infrastructure
- Curriculum learning scheduler (4 stages)
- ALiBi positional bias for extended context (512→2048+ tokens)
- RoPE with YaRN scaling for rotary embeddings
- SparseMax activation for sparse outputs
- Comprehensive training utilities module

#### Bug Fixes
- Fixed sequence length bug in DreamSequence (was generating n-1 instead of n steps)
- Fixed graph retention issue in DreamGenerator.last_load_loss (detached tensors)
- Fixed graph retention issue in DreamGraphReasoner._node_memory_states (detached tensors)

### Changed

#### Documentation Reorganization
- **Unified README.md**: Clean, focused main documentation
- **Archived legacy docs**: Moved 15+ old docs to `archive/` folder
- **Kept essential docs**:
  - `QUICK_REFERENCE.md` - Developer quick start
  - `IMPROVEMENTS_COMPLETE.md` - Technical specifications
  - `QUICKSTART.md` - Beginner guide
  - `docs/DREAMING_ARCHITECTURE.md` - Architecture deep dive
  - `docs/USAGE_EXAMPLES.md` - API patterns
  - `docs/DREAM_VIEWER.md` - Visualization guide

#### Configuration
- Expanded `DreamingConfig` from 8 to 32+ parameters
- Added 24 new hyperparameters for v2.0 features
- Backward compatible (all new features opt-in)

#### Testing
- All 25 tests passing
- Fixed test expectations for new architecture
- Added tests for MoE, Graph Transformer, PPO components

### Removed (Archived)
- `CLEANUP_COMPLETE.md` → `archive/`
- `COMPLETE.md` → `archive/`
- `TRAINING_PROGRESS.md` → `archive/`
- `DREAM_VIEWER_IMPLEMENTATION.md` → `archive/`
- `LARGE_SCALE_DISTILLATION.md` → `archive/`
- `DREAMING_README.md` → `archive/`
- `docs/COCO_IMPLEMENTATION.md` → `archive/`
- `docs/COCO_TRAINING.md` → `archive/`
- `docs/FINAL_SUMMARY.md` → `archive/`
- `docs/IMPLEMENTATION_COMPLETE.md` → `archive/`
- `docs/PROJECT_STATUS.md` → `archive/`
- `docs/WORKFLOW.md` → `archive/`
- `docs/MOE_ARCHITECTURE_EXPLAINED.md.old` → `archive/`
- `docs/moe_architecture.svg.old` → `archive/`
- Old `README.md` → `archive/README.old.md`

### Performance
- Model size: ~13-15M parameters (configurable)
- Training: Converges successfully on synthetic data (Loss 6.95→6.93 over 20 epochs)
- Inference: Real-time generation with GPU acceleration

## [1.0.0] - 2025-11-16

### Initial Release

#### Core Components
- Vision Encoder (ResNet/CLIP backbones)
- Graph RAG (knowledge graph reasoning)
- RL Learning (policy networks, reward models)
- Text Generation (transformer decoder)
- Knowledge Transfer (Ollama distillation)

#### Features
- Image triplet tokenization
- Multi-hop graph reasoning
- Online RL updates
- Knowledge distillation from Ollama models
- Export to Ollama-compatible bundles

#### Scripts
- `create_pretrained_model.py` - Create models via distillation
- `llm_chat.py` - Interactive chat interface
- `test_pretrained.py` - Model validation
- `compare_models.py` - Model comparison
- `benchmark_model.py` - Performance evaluation

#### Documentation
- Initial README with project overview
- Architecture documentation
- Usage examples
- Workflow guide
- API reference

---

## File Structure Summary

### Current Active Documentation
```
README.md                      # Main documentation (v2.0)
QUICK_REFERENCE.md            # Developer quick start
IMPROVEMENTS_COMPLETE.md      # Technical specifications
QUICKSTART.md                 # Beginner guide
docs/
  ├── DREAMING_ARCHITECTURE.md   # Architecture deep dive
  ├── USAGE_EXAMPLES.md          # API patterns
  ├── DREAM_VIEWER.md            # Visualization
  ├── architecture.md            # Legacy reference
  └── architecture_detailed.svg  # Architecture diagram
```

### Archived Documentation
```
archive/
  ├── README.old.md                      # Previous README
  ├── CLEANUP_COMPLETE.md                # Old cleanup notes
  ├── COMPLETE.md                        # Implementation notes
  ├── TRAINING_PROGRESS.md               # Training logs
  ├── DREAM_VIEWER_IMPLEMENTATION.md     # Old implementation
  ├── LARGE_SCALE_DISTILLATION.md        # Distillation notes
  ├── DREAMING_README.md                 # Old dreaming docs
  ├── COCO_IMPLEMENTATION.md             # COCO integration
  ├── COCO_TRAINING.md                   # COCO training
  ├── FINAL_SUMMARY.md                   # Old summary
  ├── IMPLEMENTATION_COMPLETE.md         # Old completion
  ├── PROJECT_STATUS.md                  # Old status
  ├── WORKFLOW.md                        # Old workflow
  ├── MOE_ARCHITECTURE_EXPLAINED.md.old  # Old MoE docs
  └── moe_architecture.svg.old           # Old diagram
```

---

## Migration Guide

### For Users of v1.0

The new `DreamingReasoningLLM` is the recommended model for new projects. Legacy `ImageTokenReasoningLLM` is still available for backward compatibility.

**Migrating to v2.0:**

```python
# Old (v1.0)
from image_token_llm.model import ImageTokenReasoningLLM
model = ImageTokenReasoningLLM(device="cuda")

# New (v2.0)
from image_token_llm.dreaming_model import DreamingReasoningLLM
model = DreamingReasoningLLM(device="cuda")

# Both models support similar APIs
output = model.generate(prompt="Your prompt", max_length=100)
```

### Configuration Changes

New parameters available in `DreamingConfig` (all optional, backward compatible):

```python
from image_token_llm.config import DreamingConfig

config = DreamingConfig(
    # New v2.0 parameters
    moe_top_k=2,
    use_contrastive_loss=True,
    vision_backbone="vit",
    use_ppo=True,
    # ... see IMPROVEMENTS_COMPLETE.md for full list
)
```

### Training Changes

Training now supports all v2.0 features:

```bash
# v1.0 training (still works)
python scripts/create_pretrained_model.py --teacher llama2

# v2.0 training (new script)
python scripts/train_dreaming_model.py \
    --device cuda \
    --epochs 20 \
    --enable-rl
```

---

Last Updated: November 17, 2025
