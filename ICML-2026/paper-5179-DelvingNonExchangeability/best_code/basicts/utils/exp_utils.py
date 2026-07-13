from omegaconf import OmegaConf


def cfg_to_python(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj
