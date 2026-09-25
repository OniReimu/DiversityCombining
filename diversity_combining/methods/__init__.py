"""Reasoning method implementations.

Each method takes a model + prompt and returns:
- answer: str
- candidates: list[str]  (K candidate tokens/paths)
- logits: torch.Tensor   (raw logits at each step)
- hidden_states: torch.Tensor  (for geometric rate, interference)
- attention_patterns: torch.Tensor  (for head specialization)
- metadata: dict  (compute cost, method-specific info)
"""

from diversity_combining.methods.base import ReasoningMethod
from diversity_combining.methods.greedy import GreedyMethod
from diversity_combining.methods.self_consistency import SelfConsistencyMethod
from diversity_combining.methods.tree_of_thought import TreeOfThoughtMethod
from diversity_combining.methods.soft_k import SoftKMethod
from diversity_combining.methods.latent_loop import LatentLoopMethod

METHOD_REGISTRY = {
    "greedy": GreedyMethod,
    "sc": SelfConsistencyMethod,
    "tot": TreeOfThoughtMethod,
    "soft_k": SoftKMethod,
    "latent_loop": LatentLoopMethod,
}

def get_method(name: str, **kwargs) -> ReasoningMethod:
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](**kwargs)
