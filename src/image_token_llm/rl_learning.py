"""Reinforcement learning for online inference-time learning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .graph_attention import GraphRAGEnhanced
from .vision_encoder import TripletEncoder


class RewardModel(nn.Module):
    """
    Multi-component reward model for evaluating reasoning trajectories.
    Evaluates: faithfulness, coherence, correctness, and creativity.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Component-specific reward heads
        self.faithfulness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.correctness_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.creativity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Learned component weights
        self.component_weights = nn.Parameter(torch.ones(4))
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        what_emb: torch.Tensor,
        action_emb: torch.Tensor,
        result_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-component reward and confidence for a triplet.
        
        Args:
            what_emb: [B, D]
            action_emb: [B, D]
            result_emb: [B, D]
        
        Returns:
            total_reward: [B, 1] - Weighted sum of all components
            confidence: [B, 1] - Confidence score
            components: Dict of individual reward components
        """
        combined = torch.cat([what_emb, action_emb, result_emb], dim=-1)
        features = self.feature_extractor(combined)
        
        # Compute individual reward components
        faithfulness = self.faithfulness_head(features)
        coherence = self.coherence_head(features)
        correctness = self.correctness_head(features)
        creativity = self.creativity_head(features)
        
        # Weighted sum with learned weights
        weights = F.softmax(self.component_weights, dim=0)
        total_reward = (
            weights[0] * faithfulness +
            weights[1] * coherence +
            weights[2] * correctness +
            weights[3] * creativity
        )
        
        confidence = self.confidence_net(features)
        
        components = {
            'faithfulness': faithfulness,
            'coherence': coherence,
            'correctness': correctness,
            'creativity': creativity,
        }
        
        return total_reward, confidence, components


class PolicyNetwork(nn.Module):
    """
    Policy network for selecting reasoning paths in the graph.
    Uses attention over graph neighbors to decide traversal.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_actions: int = 8,  # Max neighbors to consider
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Action-value network
        self.action_value = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Value baseline
        self.value_baseline = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        state_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution over neighbors.
        
        Args:
            state_emb: [B, D] - Current state embedding
            neighbor_embs: [B, N, D] - Neighbor node embeddings
            mask: [B, N] - Valid neighbor mask (1 = valid, 0 = invalid)
        
        Returns:
            action_probs: [B, N] - Action probability distribution
            value: [B, 1] - State value estimate
        """
        batch_size, num_neighbors, _ = neighbor_embs.shape
        
        # Encode state
        state_hidden = self.state_encoder(state_emb)  # [B, H]
        
        # Compute action values for each neighbor
        state_expanded = state_hidden.unsqueeze(1).expand(-1, num_neighbors, -1)  # [B, N, H]
        action_input = torch.cat([state_expanded, neighbor_embs], dim=-1)  # [B, N, H+D]
        
        action_logits = self.action_value(action_input).squeeze(-1)  # [B, N]
        
        # Apply mask
        if mask is not None:
            action_logits = action_logits.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get probabilities
        action_probs = F.softmax(action_logits, dim=-1)  # [B, N]
        
        # Value baseline
        value = self.value_baseline(state_hidden)  # [B, 1]
        
        return action_probs, value

    def sample_action(
        self,
        state_emb: torch.Tensor,
        neighbor_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Returns:
            action: [B] - Sampled action indices
            log_prob: [B] - Log probability of actions
        """
        action_probs, _ = self.forward(state_emb, neighbor_embs, mask)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        action = dist.sample()  # [B]
        log_prob = dist.log_prob(action)  # [B]
        
        return action, log_prob


class RLContinuousLearner(nn.Module):
    """
    Reinforcement learning-based continuous learner.
    Learns from rewards during inference using policy gradients.
    """

    def __init__(
        self,
        triplet_encoder: TripletEncoder,
        graph_rag: GraphRAGEnhanced,
        reward_model: RewardModel,
        policy_network: PolicyNetwork,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.triplet_encoder = triplet_encoder
        self.graph_rag = graph_rag
        self.reward_model = reward_model
        self.policy_network = policy_network
        
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        # Separate optimizers
        self.encoder_optimizer = torch.optim.AdamW(
            triplet_encoder.parameters(), lr=learning_rate
        )
        self.policy_optimizer = torch.optim.AdamW(
            policy_network.parameters(), lr=learning_rate
        )
        self.reward_optimizer = torch.optim.AdamW(
            reward_model.parameters(), lr=learning_rate
        )
        
        # Trajectory buffer
        self.trajectory_buffer: List[Dict[str, Any]] = []

    def simulate_reasoning(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
        seed_nodes: List[str],
        max_steps: int = 5,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """
        Simulate a reasoning trajectory using the policy network.
        
        Args:
            what_images: [B, C, H, W]
            action_images: [B, C, H, W]
            result_images: [B, C, H, W]
            seed_nodes: Starting nodes in graph
            max_steps: Maximum trajectory length
        
        Returns:
            trajectory: List of states, actions, rewards
            final_reward: Final trajectory reward
        """
        trajectory = []
        
        # Encode triplet
        with torch.no_grad():
            fused_emb, (what_emb, action_emb, result_emb) = self.triplet_encoder(
                what_images, action_images, result_images
            )
        
        current_state = fused_emb.squeeze(0)  # [D]
        current_nodes = seed_nodes
        
        for step in range(max_steps):
            # Get neighbors from graph
            neighbors = []
            neighbor_embs = []
            
            for node in current_nodes:
                node_neighbors = list(self.graph_rag.graph.neighbors(node))[:8]
                if node_neighbors:
                    neighbors.extend(node_neighbors)
                    for neighbor in node_neighbors:
                        if neighbor in self.graph_rag.embeddings:
                            neighbor_embs.append(self.graph_rag.embeddings[neighbor])
                        else:
                            neighbor_embs.append(torch.zeros(self.triplet_encoder.embedding_dim))
            
            if not neighbor_embs:
                break
            
            # Pad to fixed size
            max_neighbors = 8
            while len(neighbor_embs) < max_neighbors:
                neighbor_embs.append(torch.zeros(self.triplet_encoder.embedding_dim))
                neighbors.append("__padding__")
            
            neighbor_embs = neighbor_embs[:max_neighbors]
            neighbors = neighbors[:max_neighbors]
            
            # Create mask
            mask = torch.tensor([1 if n != "__padding__" else 0 for n in neighbors], dtype=torch.float32)
            
            # Stack neighbor embeddings
            neighbor_tensor = torch.stack(neighbor_embs).unsqueeze(0)  # [1, N, D]
            state_tensor = current_state.unsqueeze(0)  # [1, D]
            
            # Move to device
            device = next(self.policy_network.parameters()).device
            neighbor_tensor = neighbor_tensor.to(device)
            state_tensor = state_tensor.to(device)
            mask = mask.unsqueeze(0).to(device)  # [1, N]
            
            # Sample action
            action_idx, log_prob = self.policy_network.sample_action(
                state_tensor, neighbor_tensor, mask
            )
            
            # Execute action
            selected_neighbor = neighbors[action_idx.item()]
            
            # Compute reward
            with torch.no_grad():
                selected_emb = neighbor_tensor[0, action_idx]
                reward, confidence = self.reward_model(
                    what_emb.squeeze(0).unsqueeze(0),
                    action_emb.squeeze(0).unsqueeze(0),
                    selected_emb.unsqueeze(0),
                )
            
            # Store transition
            trajectory.append({
                "state": state_tensor.cpu(),
                "action": action_idx.cpu(),
                "log_prob": log_prob.cpu(),
                "reward": reward.cpu(),
                "confidence": confidence.cpu(),
                "neighbors": neighbor_tensor.cpu(),
                "mask": mask.cpu(),
            })
            
            # Update state
            current_state = selected_emb
            current_nodes = [selected_neighbor] if selected_neighbor != "__padding__" else []
        
        # Compute final reward
        final_reward = sum(t["reward"].item() for t in trajectory) / max(len(trajectory), 1)
        final_reward_tensor = torch.tensor(final_reward)
        
        return trajectory, final_reward_tensor

    def learn_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        final_reward: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update policy using trajectory with policy gradient.
        
        Args:
            trajectory: List of transitions
            final_reward: Final cumulative reward
        
        Returns:
            Training metrics
        """
        if not trajectory:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        device = next(self.policy_network.parameters()).device
        
        # Compute returns (discounted rewards)
        returns = []
        R = final_reward.item()
        for t in reversed(trajectory):
            R = t["reward"].item() + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for i, transition in enumerate(trajectory):
            state = transition["state"].to(device)
            neighbors = transition["neighbors"].to(device)
            mask = transition["mask"].to(device)
            log_prob = transition["log_prob"].to(device)
            
            # Forward pass
            action_probs, value = self.policy_network(state, neighbors, mask)
            
            # Compute advantage
            advantage = returns[i] - value.squeeze()
            
            # Policy loss (policy gradient)
            policy_loss = -log_prob * advantage.detach()
            
            # Value loss
            value_loss = F.mse_loss(value.squeeze(), returns[i])
            
            # Entropy bonus
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
            entropy_loss = -self.entropy_coef * entropy
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)
        
        # Aggregate losses
        total_policy_loss = torch.stack(policy_losses).mean()
        total_value_loss = torch.stack(value_losses).mean()
        total_entropy_loss = torch.stack(entropy_losses).mean()
        
        total_loss = total_policy_loss + total_value_loss + total_entropy_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()
        
        return {
            "policy_loss": total_policy_loss.item(),
            "value_loss": total_value_loss.item(),
            "entropy": -total_entropy_loss.item() / self.entropy_coef,
            "final_reward": final_reward.item(),
        }

    def online_inference_with_learning(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
        ground_truth_label: Optional[Tuple[str, str, str]] = None,
        seed_nodes: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform inference while learning from the experience.
        
        Args:
            what_images: [B, C, H, W]
            action_images: [B, C, H, W]
            result_images: [B, C, H, W]
            ground_truth_label: Optional ground truth for reward computation
            seed_nodes: Starting nodes for graph traversal
        
        Returns:
            prediction: Final embedding
            metrics: Learning metrics
        """
        # Simulate reasoning trajectory
        if seed_nodes is None:
            seed_nodes = ["vision", "action", "perception"]
        
        trajectory, final_reward = self.simulate_reasoning(
            what_images, action_images, result_images, seed_nodes
        )
        
        # Learn from trajectory
        metrics = self.learn_from_trajectory(trajectory, final_reward)
        
        # If ground truth provided, update reward model
        if ground_truth_label is not None:
            self._update_reward_model(
                what_images, action_images, result_images, ground_truth_label
            )
            metrics["reward_updated"] = 1.0
        
        # Update graph with new knowledge
        with torch.no_grad():
            fused_emb, _ = self.triplet_encoder(
                what_images, action_images, result_images
            )
            
            if ground_truth_label is not None:
                self.graph_rag.ingest([ground_truth_label], [fused_emb.squeeze(0)])
        
        return fused_emb, metrics

    def _update_reward_model(
        self,
        what_images: torch.Tensor,
        action_images: torch.Tensor,
        result_images: torch.Tensor,
        ground_truth: Tuple[str, str, str],
    ) -> None:
        """Update reward model using ground truth feedback."""
        # Encode triplet
        fused_emb, (what_emb, action_emb, result_emb) = self.triplet_encoder(
            what_images, action_images, result_images
        )
        
        # Predict reward
        pred_reward, pred_confidence = self.reward_model(
            what_emb.squeeze(0).unsqueeze(0),
            action_emb.squeeze(0).unsqueeze(0),
            result_emb.squeeze(0).unsqueeze(0),
        )
        
        # Target reward (1.0 for correct, 0.0 for incorrect)
        # In real scenario, use actual correctness signal
        target_reward = torch.ones_like(pred_reward)
        
        # Loss
        reward_loss = F.binary_cross_entropy(pred_reward, target_reward)
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_reward)
        
        total_loss = reward_loss + 0.5 * confidence_loss
        
        # Update
        self.reward_optimizer.zero_grad()
        total_loss.backward()
        self.reward_optimizer.step()


class PPOTrainer(nn.Module):
    """
    Proximal Policy Optimization (PPO) trainer for more stable RL.
    Implements clipped surrogate objective with value function.
    """

    def __init__(
        self,
        policy_network: PolicyNetwork,
        value_network: nn.Module,
        reward_model: RewardModel,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,  # GAE lambda
        vf_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        super().__init__()
        self.policy_network = policy_network
        self.value_network = value_network
        self.reward_model = reward_model
        
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Single optimizer for both policy and value
        self.optimizer = torch.optim.AdamW(
            list(policy_network.parameters()) +
            list(value_network.parameters()),
            lr=learning_rate
        )
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: [T] - Rewards for each timestep
            values: [T+1] - Value estimates (includes bootstrap value)
            dones: [T] - Done flags
            
        Returns:
            advantages: [T] - GAE advantages
            returns: [T] - Discounted returns
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=rewards.device)
        
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        return advantages, returns
    
    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        neighbor_embs: torch.Tensor,
        masks: torch.Tensor,
        num_epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Perform PPO update with multiple epochs over the data.
        
        Args:
            states: [T, D] - State embeddings
            actions: [T] - Action indices
            old_log_probs: [T] - Log probs from old policy
            advantages: [T] - GAE advantages
            returns: [T] - Target returns
            neighbor_embs: [T, N, D] - Neighbor embeddings
            masks: [T, N] - Valid neighbor masks
            num_epochs: Number of update epochs
            batch_size: Mini-batch size
            
        Returns:
            Training metrics
        """
        T = states.shape[0]
        indices = torch.arange(T)
        
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'clip_fraction': 0.0,
        }
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Shuffle indices
            perm = torch.randperm(T, device=states.device)
            
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = perm[start:end]
                
                # Get batch
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_neighbors = neighbor_embs[batch_idx]
                batch_masks = masks[batch_idx]
                
                # Forward pass
                action_probs, new_values = self.policy_network(
                    batch_states, batch_neighbors, batch_masks
                )
                
                # Compute new log probs
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred = new_values.squeeze(-1)
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.vf_coef * value_loss -
                    self.entropy_coef * entropy
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_network.parameters()) +
                    list(self.value_network.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                with torch.no_grad():
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                    metrics['clip_fraction'] += clip_fraction.item()
                num_updates += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_updates
        
        return metrics
