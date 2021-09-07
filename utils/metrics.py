import jax
import jax.numpy as jnp

def measurer(net):
    @jax.jit
    def accuracy(params, batch):
        predictions = net.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    @jax.jit
    def attack_success_rate(params, batch, attack_from, attack_to):
        preds = jnp.argmax(net.apply(params, batch), axis=-1)
        mask = jnp.where(batch['label'] == attack_from, 1, 0)
        return jnp.sum((preds * mask) == attack_to) / jnp.sum(mask)
    return {'acc': accuracy, 'asr': attack_success_rate}

def create_recorder(evals, train=False, test=False, add_evals=None):
    results = dict()
    if train:    
        results.update({f"train {e}": [] for e in evals})
    if test:
        results.update({f"test {e}": [] for e in evals})
    if add_evals is not None:
        results.update({e: [] for e in add_evals})
    return results

def record(results, evaluator, params, train_ds=None, test_ds=None, add_recs=None, **kwargs):
    for k, v in results.items():
        ds = train_ds if "train" in k else test_ds
        if "acc" in k:
            v.append(evaluator['acc'](params, next(ds)))
        if "asr" in k:
            v.append(evaluator['asr'](params, next(ds), kwargs['attack_from'], kwargs['attack_to']))
    for k, v in add_recs.items():
        results[k].append(v)
