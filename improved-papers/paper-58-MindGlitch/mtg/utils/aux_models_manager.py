import lpips
from mtg.utils.loaders import cleandift_loader, inpainting_model_loader, groundingsam_loader


class ModelManager:
    def __init__(
        self, DEVICE, cfg, load_cleandift=False, load_inpainting=False, load_lpips=False, load_groundingsam=False
    ):
        self.DEVICE = DEVICE
        self.cfg = cfg
        self._load_all_models(load_cleandift, load_inpainting, load_lpips, load_groundingsam)

    def _load_all_models(self, load_cleandift, load_inpainting, load_lpips, load_groundingsam):
        # Load the CleanDIFT model
        self._cleandift_model = cleandift_loader(self.DEVICE) if load_cleandift else None
        # Load the Grounding DINO and SAM models
        self._groundingdino_model, self._sam_predictor = (
            groundingsam_loader(self.DEVICE) if load_groundingsam else (None, None)
        )
        # Load the inpainting model
        self._inpainting_pipe = (
            inpainting_model_loader(self.cfg.inpainting_model, self.DEVICE) if load_inpainting else None
        )
        # Load LPIPS model
        self._lpips_model = lpips.LPIPS(net="vgg").to(self.DEVICE) if load_lpips else None

    @property
    def cleandift_model(self):
        if self._cleandift_model is None:
            raise ValueError("Cleandift model is not loaded")
        return self._cleandift_model

    @property
    def groundingdino_model(self):
        if self._groundingdino_model is None:
            raise ValueError("Grounding DINO model is not loaded")
        return self._groundingdino_model

    @property
    def sam_predictor(self):
        if self._sam_predictor is None:
            raise ValueError("SAM predictor is not loaded")
        return self._sam_predictor

    @property
    def inpainting_pipe(self):
        if self._inpainting_pipe is None:
            raise ValueError("Inpainting pipe is not loaded")
        return self._inpainting_pipe

    @property
    def lpips_model(self):
        if self._lpips_model is None:
            raise ValueError("LPIPS model is not loaded")
        return self._lpips_model
