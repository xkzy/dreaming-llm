# Quick Reference Guide - Model Improvements

## 🚀 Quick Start

### Enable All Improvements
```python
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.config import DreamingConfig

# Create config with all improvements enabled
config = DreamingConfig(
    # MoE improvements
    moe_top_k=2,
    moe_noise_std=0.1,
    load_balance_loss_weight=0.01,
    use_cross_expert_attention=True,
    
    # Input improvements
    use_contrastive_loss=True,
    token_mask_prob=0.1,
    embedding_dropout=0.15,
    
    # Graph improvements
    use_learned_edges=True,
    use_node_memory=True,
    
    # RL improvements
    use_ppo=True,
    reward_components=True,
)

# Initialize model
model = DreamingReasoningLLM(config=config, device="cuda")
```

---

## 🎯 Feature-by-Feature Usage

### 1. Contrastive Alignment Loss
```python
from image_token_llm.dreaming import InputTokenizer

tokenizer = InputTokenizer(config)

# During training
loss = tokenizer.contrastive_loss(text_embeddings, image_embeddings)
total_loss = task_loss + 0.1 * loss  # Weight: 0.1
```

### 2. Vision Transformer with Multi-Scale Features
```python
from image_token_llm.vision_encoder import VisionEncoder

# Use ViT instead of CLIP
encoder = VisionEncoder(config, backbone="vit")
features = encoder(images)  # Automatically uses pyramid

# Or use CLIP (default)
encoder = VisionEncoder(config, backbone="clip")
```

### 3. MoE with Top-K Gating and Load Balancing
```python
from image_token_llm.dreaming import DreamGenerator

generator = DreamGenerator(config)

# Get load balancing loss during training
dreams = generator(what, action, result)
load_loss = generator.last_load_loss  # Access stored loss

# Add to training objective
total_loss = main_loss + 0.01 * load_loss
```

### 4. Graph Transformer with Learned Edges
```python
from image_token_llm.dream_graph_reasoner import DreamGraphReasoner

reasoner = DreamGraphReasoner(config)

# Automatically uses:
# - Learned edge predictor
# - Graph Transformer layers
# - Node memory states
reasoning_emb = reasoner(dream_sequences)
```

### 5. Multi-Component Rewards
```python
from image_token_llm.rl_learning import RewardModel

reward_model = RewardModel()

# Returns all components
total_reward, confidence, components = reward_model(what, action, result)

# Components dict contains:
# - faithfulness: How faithful to input
# - coherence: How coherent the reasoning
# - correctness: How correct the output
# - creativity: How creative/novel

print(f"Faithfulness: {components['faithfulness'].item():.3f}")
print(f"Coherence: {components['coherence'].item():.3f}")
```

### 6. PPO Training
```python
from image_token_llm.rl_learning import PPOTrainer, PolicyNetwork, RewardModel

policy = PolicyNetwork(embedding_dim=512)
value = PolicyNetwork(embedding_dim=512)  # Can reuse same architecture
reward_model = RewardModel()

ppo = PPOTrainer(
    policy_network=policy,
    value_network=value,
    reward_model=reward_model,
    clip_epsilon=0.2,
    gamma=0.99,
    lam=0.95,
)

# Collect trajectories first
states, actions, old_log_probs = collect_trajectories()

# Compute advantages using GAE
advantages, returns = ppo.compute_gae(rewards, values, dones)

# Update with multiple epochs
metrics = ppo.ppo_update(
    states, actions, old_log_probs,
    advantages, returns,
    neighbor_embs, masks,
    num_epochs=4,
    batch_size=64,
)

print(f"Policy Loss: {metrics['policy_loss']:.4f}")
print(f"Value Loss: {metrics['value_loss']:.4f}")
print(f"Clip Fraction: {metrics['clip_fraction']:.2%}")
```

### 7. Curriculum Learning
```python
from image_token_llm.training_utils import CurriculumLearningScheduler

scheduler = CurriculumLearningScheduler()

for step in range(total_steps):
    # Get current stage
    stage = scheduler.step()
    
    # Get stage-specific config
    stage_config = scheduler.get_stage_config()
    
    # Adjust model parameters
    model.config.dream_length = stage_config['dream_length']
    model.config.num_dream_sequences = stage_config['num_dreams']
    
    # Check progress
    progress = scheduler.get_progress()
    print(f"Stage: {stage}, Progress: {progress:.1%}")
    
    # Train...
```

### 8. Extended Context with ALiBi
```python
from image_token_llm.training_utils import create_extended_context_model

# Wrap model to support longer context
extended_model = create_extended_context_model(
    base_model,
    use_alibi=True,
    context_length=2048,  # Up from 512
)

# Now supports 4x longer sequences without retraining
long_sequence = torch.randn(1, 2048, embedding_dim)
output = extended_model(long_sequence)
```

### 9. RoPE Positional Encoding
```python
from image_token_llm.training_utils import RoPEPositionalEncoding

rope = RoPEPositionalEncoding(
    embedding_dim=512,
    max_positions=2048,
    scaling_factor=2.0,  # YaRN scaling for 2x context
)

# Apply to embeddings
embeddings = torch.randn(batch_size, seq_len, embedding_dim)
rotated = rope(embeddings, seq_len)
```

### 10. SparseMax Activation
```python
from image_token_llm.training_utils import SparseMaxActivation

sparsemax = SparseMaxActivation(dim=-1)

# Use instead of softmax for sparse attention
logits = torch.randn(batch_size, num_heads, seq_len, seq_len)
sparse_weights = sparsemax(logits)

# Many weights will be exactly 0
print(f"Sparsity: {(sparse_weights == 0).float().mean():.1%}")
```

---

## 📊 Training Loop Example

```python
from image_token_llm.dreaming_model import DreamingReasoningLLM
from image_token_llm.training_utils import CurriculumLearningScheduler

model = DreamingReasoningLLM(config=config, device="cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
curriculum = CurriculumLearningScheduler()

for step in range(total_steps):
    # Get current curriculum stage
    stage = curriculum.step()
    stage_config = curriculum.get_stage_config()
    
    # Load batch based on stage
    if stage_config['use_images']:
        batch = load_multimodal_batch()
    else:
        batch = load_text_batch()
    
    # Forward pass
    output = model(
        text_input=batch['text'],
        image_input=batch.get('images'),
    )
    
    # Compute losses
    task_loss = criterion(output, batch['target'])
    
    # Add contrastive loss if using images
    contrastive_loss = 0.0
    if stage_config['use_images']:
        contrastive_loss = model.input_tokenizer.contrastive_loss(
            batch['text_emb'], batch['image_emb']
        )
    
    # Add load balancing loss for MoE
    load_loss = model.dream_generator.last_load_loss
    
    # Total loss
    total_loss = (
        task_loss +
        0.1 * contrastive_loss +
        0.01 * load_loss
    )
    
    # Backward and optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Log
    if step % 100 == 0:
        print(f"Step {step}, Stage: {stage}")
        print(f"  Task Loss: {task_loss.item():.4f}")
        print(f"  Contrastive: {contrastive_loss:.4f}")
        print(f"  Load Balance: {load_loss.item():.4f}")
```

---

## 🔧 Hyperparameter Tuning Guide

### Contrastive Loss
- **Temperature** (0.01-0.1): Lower = harder negatives
- **Weight** (0.05-0.2): Balance with main task
- Default: τ=0.07, weight=0.1

### MoE Gating
- **Top-K** (1-4): Number of active experts
- **Noise Std** (0.05-0.2): Exploration during training
- **Load Balance Weight** (0.001-0.05): Strength of balancing
- Default: k=2, noise=0.1, weight=0.01

### Graph Reasoning
- **Num Hops** (2-5): Reasoning depth
- **Num Heads** (4-16): Multi-head attention
- **Edge Types** (2-8): Relational categories
- Default: hops=3, heads=8, types=4

### PPO
- **Clip Epsilon** (0.1-0.3): Clipping range
- **GAE Lambda** (0.9-0.99): Advantage smoothing
- **Num Epochs** (3-10): Updates per batch
- Default: ε=0.2, λ=0.95, epochs=4

### Curriculum Learning
- **Steps per Stage**: Adjust based on dataset size
- **Stage Order**: Can customize beyond default
- Default: [1000, 2000, 3000, 5000]

---

## ⚡ Performance Tips

### Memory Optimization
1. **Use gradient accumulation** for larger effective batch size
2. **Enable gradient checkpointing** for Transformer layers
3. **Use mixed precision (FP16)** for faster training
4. **Reduce num_dreams** during early training stages

### Speed Optimization
1. **Use DataLoader with num_workers=4+**
2. **Pin memory** for faster GPU transfer
3. **Use compiled models** (torch.compile in PyTorch 2.0+)
4. **Batch inference** when possible

### Quality Optimization
1. **Start with pretrained ViT** weights
2. **Warm up learning rate** for first 1000 steps
3. **Use cosine annealing** for learning rate decay
4. **Regular validation** to detect overfitting
5. **Monitor expert usage** (should be balanced)

---

## 🐛 Debugging Checklist

### Training Instabilities
- [ ] Check gradient norms (should be < 10)
- [ ] Verify load balancing (experts should be used equally)
- [ ] Monitor reward variance (PPO should reduce this)
- [ ] Check for NaN/Inf in losses

### Poor Quality
- [ ] Verify contrastive loss is decreasing
- [ ] Check expert specialization (shouldn't all be identical)
- [ ] Monitor individual reward components
- [ ] Validate curriculum progression is appropriate

### Memory Issues
- [ ] Reduce batch size
- [ ] Reduce num_dreams or dream_length
- [ ] Use gradient checkpointing
- [ ] Clear graph memory periodically

---

## 📈 Monitoring Metrics

### Key Metrics to Track
```python
metrics = {
    # Main task
    'task_loss': task_loss.item(),
    'task_accuracy': accuracy,
    
    # Contrastive alignment
    'contrastive_loss': contrastive_loss,
    'text_image_similarity': similarity.mean().item(),
    
    # MoE
    'load_balance_loss': load_loss.item(),
    'expert_0_usage': expert_usage[0],
    'expert_1_usage': expert_usage[1],
    'expert_2_usage': expert_usage[2],
    'expert_3_usage': expert_usage[3],
    
    # Rewards
    'total_reward': total_reward.mean().item(),
    'faithfulness': components['faithfulness'].mean().item(),
    'coherence': components['coherence'].mean().item(),
    'correctness': components['correctness'].mean().item(),
    'creativity': components['creativity'].mean().item(),
    
    # PPO
    'policy_loss': metrics['policy_loss'],
    'value_loss': metrics['value_loss'],
    'clip_fraction': metrics['clip_fraction'],
    'entropy': metrics['entropy'],
    
    # Curriculum
    'stage': stage,
    'stage_progress': curriculum.get_progress(),
}
```

---

## 🎯 Model Selection Guide

### Choose Architecture Based on Task

**Simple Reasoning (< 3 hops)**
```python
config = DreamingConfig(
    num_dream_sequences=2,
    dream_length=3,
    graph_reasoning_hops=2,
    expert_num_layers=1,
)
```

**Complex Reasoning (3-5 hops)**
```python
config = DreamingConfig(
    num_dream_sequences=4,
    dream_length=5,
    graph_reasoning_hops=3,
    expert_num_layers=2,
)
```

**Very Complex Reasoning (5+ hops)**
```python
config = DreamingConfig(
    num_dream_sequences=4,
    dream_length=7,
    graph_reasoning_hops=4,
    expert_num_layers=3,
)
```

---

## 💡 Best Practices

### Do's
✅ Start with curriculum learning  
✅ Monitor expert balance regularly  
✅ Use contrastive loss with paired data  
✅ Enable all stabilization features (pre-norm, residuals)  
✅ Validate on held-out set frequently  

### Don'ts
❌ Don't skip curriculum stages  
❌ Don't use very high noise_std (> 0.3)  
❌ Don't ignore load imbalance warnings  
❌ Don't forget to normalize rewards  
❌ Don't train without gradient clipping  

---

*Last Updated: 2025-11-17*
*Version: 2.0*
