from ymir import garrison
from ymir import mp
from ymir.scout import collaborator

class OnOffController:
    """
    Network controller that toggles an attack on or off respective to the result of the aggregation algorithm
    """
    def __init__(self, network, params, alg, adversaries, max_alpha, sharp, beta=1.0, gamma=0.85):
        self.alg = alg
        self.attacking = False
        self.max_alpha = max_alpha
        self.sharp = sharp
        self.beta = beta
        self.gamma = gamma
        self.server = getattr(garrison.aggregators, self.alg).Server(params, network)
        self.adversaries = adversaries
        self.num_adv = len(adversaries)
        
    def should_toggle(self, alpha):
        avg_syb_alpha = alpha[-self.num_adv:].mean()
        p = self.attacking and avg_syb_alpha < self.beta * self.max_alpha
        if self.sharp:
            q = not self.attacking and avg_syb_alpha > 0.4 * self.max_alpha
        else:
            q = not self.attacking and avg_syb_alpha > self.gamma * self.max_alpha
        return p or q

    def __call__(self, all_grads):
        """Update each connected client and return the generated gradients. Recursively call in connected controllers"""
        self.server.update(all_grads)
        alpha = self.server.scale(all_grads)
        if self.should_toggle(alpha):
            self.attacking = not self.attacking
            for a in self.adversaries:
                a.toggle()
        return all_grads


def make_onoff(client):
    client.shadow_update = client.update
    client.update = collaborator.update(client.opt, client.loss)
    client.toggle = toggle.__get__(client)


def toggle(self):
    self.update, shadow_update = self.shadow_update, self.update