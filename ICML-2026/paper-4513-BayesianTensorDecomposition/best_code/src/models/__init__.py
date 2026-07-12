from .edm.edm import create_edm_from_unet_adm

def get_model(name: str, **kwargs):
    if name == 'edm_from_unet_adm':
        return create_edm_from_unet_adm(**kwargs)
    else:
        raise NameError(f"Model {name} is not defined.")
