import asyncio
import torch
import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_finetuner.agent import RLFineTunerAgent

class TestRLFineTunerAgent(unittest.IsolatedAsyncioTestCase):
    async def test_policy_update_changes_weights(self):
        agent = RLFineTunerAgent(config={'batch_size': 2, 'update_target_steps': 1})
        experiences = [
            {
                'state': {'correctness': 0.2, 'runtime_ms': 100, 'token_count': 10},
                'action': 1,
                'next_state': {'correctness': 0.4, 'runtime_ms': 90, 'token_count': 9}
            },
            {
                'state': {'correctness': 0.4, 'runtime_ms': 90, 'token_count': 9},
                'action': 0,
                'next_state': {'correctness': 0.6, 'runtime_ms': 80, 'token_count': 8}
            }
        ]
        params_before = [p.clone() for p in agent.policy_net.parameters()]
        await agent.update_policy(experiences)
        params_after = list(agent.policy_net.parameters())
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
        self.assertTrue(changed)

    async def test_execute_returns_action(self):
        agent = RLFineTunerAgent()
        result = await agent.execute({'correctness': 0.5, 'runtime_ms': 100, 'token_count': 20})
        self.assertIn('action_index', result)
        self.assertIn('action_name', result)

