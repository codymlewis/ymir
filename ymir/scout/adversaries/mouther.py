from ymir import mp

import ymirlib


class GradientTransform:
    """
    Network controller that scales adversaries' gradients by the inverse of aggregation algorithm
    """
    def __init__(self, num_adversaries, victim, attack_type):
        self.num_adv = num_adversaries
        self.victim = victim
        self.attack_type = attack_type
        
    def __call__(self, all_grads):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        grad = all_grads[self.victim]
        if "bad" in self.attack_type:
            grad = ymirlib.tree_mul(grad, -1)
        all_grads[-self.num_adv:] = [ymirlib.tree_add_normal(grad, loc=0.0, scale=10e-4) for _ in range(self.num_adv)]
        return all_grads