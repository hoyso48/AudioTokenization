from .expert import UpstreamExpert as _UpstreamExpert

def dtmae_base(ckpt: str = None, model_config: str = None, **kwargs):
    """
    ckpt: DTM/PLE checkpoint path
    model_config: (optional) config path
    """
    return _UpstreamExpert(ckpt=ckpt, model_config=model_config, **kwargs)
