# Documentation Index

Complete guide to all documentation in the Image-Token Dreaming LLM project.

## 📖 Getting Started

Start here if you're new to the project:

1. **[README.md](../README.md)** - Main project documentation
   - Overview of the Dreaming-Based Architecture
   - Quick start examples
   - Installation instructions
   - Architecture diagram
   - Key features and components

2. **[QUICKSTART.md](../QUICKSTART.md)** - Beginner-friendly guide
   - Step-by-step installation
   - First examples
   - Basic usage patterns
   - Common tasks

3. **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Developer quick start
   - Feature-by-feature usage
   - Training loop examples
   - Hyperparameter tuning
   - Debugging checklist
   - Monitoring metrics

## 🔬 Technical Documentation

Deep dives into architecture and implementation:

1. **[IMPROVEMENTS_COMPLETE.md](../IMPROVEMENTS_COMPLETE.md)** - V2.0 Technical Specs
   - Complete list of 19 improvements
   - Mathematical formulas and algorithms
   - Implementation details
   - Performance impact analysis
   - Configuration parameters
   - Migration guide

2. **[DREAMING_ARCHITECTURE.md](DREAMING_ARCHITECTURE.md)** - Architecture Deep Dive
   - Visual reasoning system
   - MoE (Mixture of Experts) design
   - Graph Transformer details
   - RL integration
   - Component interactions
   - Design decisions

3. **[architecture.md](architecture.md)** - Legacy Reference
   - Original architecture overview
   - Historical context
   - Comparison with v2.0

## 🎨 Visual Resources

Architecture diagrams and visualizations:

1. **[architecture_detailed.svg](architecture_detailed.svg)** - Complete Architecture Diagram
   - All v2.0 components visualized
   - Data flow diagrams
   - Formulas and equations
   - Color-coded sections:
     - Blue: Input/Output
     - Red: MoE Experts
     - Green: Graph Reasoning
     - Orange: RL Components
     - Purple: New v2.0 Features

2. **[dreaming_architecture.svg](dreaming_architecture.svg)** - Simplified Overview
   - High-level system flow
   - Major components only
   - Easy to understand

## 💻 Usage & Examples

Practical guides and code examples:

1. **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - API Patterns & Examples
   - Model initialization patterns
   - Training examples
   - Generation examples
   - Advanced usage
   - Best practices
   - Common patterns

2. **[DREAM_VIEWER.md](DREAM_VIEWER.md)** - Visualization Guide
   - Dream recording
   - Real-time visualization
   - Interactive viewer
   - Graph visualization
   - Heatmap generation

## 📝 Project Management

Track project history and changes:

1. **[CHANGELOG.md](../CHANGELOG.md)** - Version History
   - Release notes
   - Feature additions
   - Bug fixes
   - Breaking changes
   - Migration guides
   - File organization

## 📂 Archived Documentation

Historical documents (in `archive/` folder):

### Implementation Notes
- **COMPLETE.md** - Original implementation completion notes
- **IMPLEMENTATION_COMPLETE.md** - Legacy completion summary
- **CLEANUP_COMPLETE.md** - Old cleanup documentation
- **FINAL_SUMMARY.md** - Previous project summary

### Training & Progress
- **TRAINING_PROGRESS.md** - Historical training logs
- **LARGE_SCALE_DISTILLATION.md** - Large-scale distillation notes
- **COCO_TRAINING.md** - COCO dataset training

### Feature Documentation
- **DREAM_VIEWER_IMPLEMENTATION.md** - Old viewer implementation
- **DREAMING_README.md** - Previous dreaming architecture docs
- **COCO_IMPLEMENTATION.md** - COCO integration details
- **PROJECT_STATUS.md** - Old status tracking
- **WORKFLOW.md** - Previous workflow guide

### Architecture Archives
- **MOE_ARCHITECTURE_EXPLAINED.md.old** - Original MoE documentation
- **moe_architecture.svg.old** - Old MoE diagram
- **README.old.md** - Previous main README

## 🗺️ Documentation Map

### By Audience

**For Beginners:**
1. README.md → Overview
2. QUICKSTART.md → First steps
3. USAGE_EXAMPLES.md → Code examples

**For Developers:**
1. QUICK_REFERENCE.md → Quick patterns
2. IMPROVEMENTS_COMPLETE.md → Technical details
3. DREAMING_ARCHITECTURE.md → Architecture deep dive

**For Researchers:**
1. IMPROVEMENTS_COMPLETE.md → Algorithms & formulas
2. DREAMING_ARCHITECTURE.md → Design decisions
3. architecture_detailed.svg → Visual reference
4. CHANGELOG.md → Evolution & history

**For Contributors:**
1. QUICK_REFERENCE.md → Development setup
2. IMPROVEMENTS_COMPLETE.md → Codebase structure
3. CHANGELOG.md → Recent changes
4. .github/copilot-instructions.md → Dev guidelines

### By Topic

**Architecture:**
- README.md (§ Architecture)
- DREAMING_ARCHITECTURE.md (complete)
- architecture_detailed.svg (visual)
- IMPROVEMENTS_COMPLETE.md (technical)

**Training:**
- QUICK_REFERENCE.md (§ Training Loop)
- IMPROVEMENTS_COMPLETE.md (§ Training Infrastructure)
- README.md (§ Training)

**Usage:**
- USAGE_EXAMPLES.md (complete)
- QUICKSTART.md (basics)
- QUICK_REFERENCE.md (patterns)
- README.md (§ Quick Start)

**MoE (Mixture of Experts):**
- IMPROVEMENTS_COMPLETE.md (§ Dream Generator MoE)
- DREAMING_ARCHITECTURE.md (§ MoE Design)
- architecture_detailed.svg (§ Dream Generator)

**Graph Reasoning:**
- IMPROVEMENTS_COMPLETE.md (§ Graph Reasoner)
- DREAMING_ARCHITECTURE.md (§ Graph Transformer)
- architecture_detailed.svg (§ Graph Reasoner)

**Reinforcement Learning:**
- IMPROVEMENTS_COMPLETE.md (§ RL Components)
- DREAMING_ARCHITECTURE.md (§ RL Integration)
- architecture_detailed.svg (§ RL Components)

**Visualization:**
- DREAM_VIEWER.md (complete)
- README.md (§ Dream Visualization)
- USAGE_EXAMPLES.md (§ Visualization)

## 📚 External Resources

### Dependencies
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NetworkX Documentation](https://networkx.org/documentation/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

### Research Papers
- Switch Transformers (noisy top-k gating)
- Graph Attention Networks (GAT)
- Proximal Policy Optimization (PPO)
- ALiBi (Attention with Linear Biases)
- RoPE (Rotary Position Embeddings)

### Related Projects
- Ollama (knowledge distillation source)
- CLIP (vision encoder)
- Vision Transformers (ViT)

## 🔍 Finding Information

### Quick Lookup

**"How do I..."**
- ...install? → QUICKSTART.md
- ...train a model? → README.md § Training
- ...use the API? → USAGE_EXAMPLES.md
- ...debug issues? → QUICK_REFERENCE.md § Debugging

**"What is..."**
- ...the architecture? → DREAMING_ARCHITECTURE.md
- ...MoE? → IMPROVEMENTS_COMPLETE.md § Dream Generator
- ...PPO? → IMPROVEMENTS_COMPLETE.md § RL Components
- ...new in v2.0? → CHANGELOG.md

**"Where is..."**
- ...the config? → src/image_token_llm/config.py
- ...the main model? → src/image_token_llm/dreaming_model.py
- ...the training script? → scripts/train_dreaming_model.py
- ...the tests? → tests/test_dreaming_model.py

## 🎯 Recommended Reading Paths

### Path 1: Quick Start (30 minutes)
1. README.md - Overview (10 min)
2. QUICKSTART.md - Installation & basics (10 min)
3. Try the training script (10 min)

### Path 2: Development (2 hours)
1. QUICK_REFERENCE.md - Developer guide (30 min)
2. USAGE_EXAMPLES.md - API patterns (30 min)
3. IMPROVEMENTS_COMPLETE.md - Technical specs (1 hour)

### Path 3: Research (4 hours)
1. DREAMING_ARCHITECTURE.md - Full architecture (1 hour)
2. IMPROVEMENTS_COMPLETE.md - All features (1.5 hours)
3. architecture_detailed.svg - Visual study (30 min)
4. Code review - Core modules (1 hour)

---

Last Updated: November 17, 2025

## 📧 Need Help?

Can't find what you're looking for? Check:
1. This index for the right document
2. The document's table of contents
3. The CHANGELOG for recent changes
4. The archived docs for historical info
