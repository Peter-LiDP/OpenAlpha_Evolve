import logging
import random
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.interfaces import RLFineTunerInterface, BaseAgent

logger = logging.getLogger(__name__)

class _DQN(nn.Module):
    """Simple feed-forward network for DQN."""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RLFineTunerAgent(RLFineTunerInterface):
    """DQN-based agent for prompt fine tuning."""

    ACTIONS = [
        "keep_prompt",
        "add_example",
        "emphasize_quality",
        "increase_temperature",
        "decrease_temperature",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self.state_dim = 3  # correctness, runtime_ms, token_count
        self.action_dim = len(self.ACTIONS)
        self.gamma = self.config.get("gamma", 0.99)
        self.lr = self.config.get("learning_rate", 1e-3)
        self.batch_size = self.config.get("batch_size", 16)
        self.update_target_steps = self.config.get("update_target_steps", 50)

        self.policy_net = _DQN(self.state_dim, self.action_dim)
        self.target_net = _DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.buffer: List[Dict[str, Any]] = []
        self.step_count = 0
        logger.info("RLFineTunerAgent initialized with DQN.")

    def _metrics_to_tensor(self, metrics: Dict[str, float]) -> torch.Tensor:
        """Convert metric dictionary to tensor state representation."""
        correctness = metrics.get("correctness", 0.0)
        runtime = metrics.get("runtime_ms", 0.0)
        token_count = metrics.get("token_count", 0.0)
        return torch.tensor([correctness, runtime, token_count], dtype=torch.float32)

    def _compute_reward(self, metrics: Dict[str, float]) -> float:
        """Combine evaluation metrics into a scalar reward."""
        correctness = metrics.get("correctness", 0.0)
        runtime = metrics.get("runtime_ms", 0.0)
        token_count = metrics.get("token_count", 0.0)
        reward = correctness - 0.001 * runtime - 0.0001 * token_count
        return reward

    async def update_policy(self, experience_data: List[Dict]) -> None:
        """Update DQN policy from a batch of experiences."""
        if not experience_data:
            logger.warning("update_policy called with no experience data.")
            return
        self.buffer.extend(experience_data)

        while len(self.buffer) >= self.batch_size:
            batch = [self.buffer.pop(0) for _ in range(self.batch_size)]
            states = torch.stack([self._metrics_to_tensor(exp["state"]) for exp in batch])
            actions = torch.tensor([exp["action"] for exp in batch], dtype=torch.long)
            next_states = torch.stack([self._metrics_to_tensor(exp["next_state"]) for exp in batch])
            rewards = torch.tensor([self._compute_reward(exp["next_state"]) for exp in batch], dtype=torch.float32)

            q_values = self.policy_net(states)
            state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                targets = rewards + self.gamma * next_q
            loss = F.mse_loss(state_action_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.step_count += 1
            if self.step_count % self.update_target_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                logger.debug("Target network updated.")

        logger.info(f"Updated policy with {len(experience_data)} experiences.")

    def select_action(self, state_metrics: Dict[str, float], epsilon: float = 0.1) -> int:
        """Select an action using epsilon-greedy strategy."""
        state = self._metrics_to_tensor(state_metrics).unsqueeze(0)
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            return int(self.policy_net(state).argmax(1).item())

    async def execute(self, state_metrics: Dict[str, float]) -> Any:
        """Return an action recommendation for the given state metrics."""
        action_idx = self.select_action(state_metrics)
        action_name = self.ACTIONS[action_idx]
        logger.debug(f"Selected action {action_name} (index {action_idx}) for metrics {state_metrics}.")
        return {"action_index": action_idx, "action_name": action_name}
