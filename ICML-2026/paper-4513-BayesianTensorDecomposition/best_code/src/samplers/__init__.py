from .pnp_edm.pnp_edm import PnPEDM

def get_sampler(config, model, operator, noiser, device):
    if config.name == 'pnp_edm':
        return PnPEDM(config, model, operator, noiser, device)
    else:
        raise NameError(f"Model {config.name} is not defined.")
