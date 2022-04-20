from . import scout


class Scout(scout.Scout):
    """A federated learning client which performs just local learning."""

    def step(self, weights, return_weights=False):
        """
        Perform a single local training loop.
        """
        for _ in range(self.epochs):
            x, y = next(self.data)
            loss = self._step(x, y)
        return loss, weights, self.batch_size
