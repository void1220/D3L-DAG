"""Construct the selected D3L model."""
from .model import GraphMaskModel

def get_d3l_variant(n, m, config):
    keys = ('width', 'layers', 'heads', 'dropout', 'steps', 'noise_schedule', 'feature_dim', 'feature_mode', 'cross_attention')
    kwargs = {k: config[k] for k in keys}
    if config['decoder'] == 'ar':
        kwargs['steps'] = 2 * n
    model = GraphMaskModel(n, m, **kwargs)
    model.unmasked_priority_logits = config['decoder'] != 'ar'
    return model
