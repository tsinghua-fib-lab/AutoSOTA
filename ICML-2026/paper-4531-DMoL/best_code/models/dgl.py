from .dmol import DMoL_Network

class DGL_Network(DMoL_Network):
    """
    DGL 网络结构与 DMoL 相同，但用于贪婪解耦训练。
    结构: FeatureExtractor -> [Module 0] -> [Module 1] ...
    """
    def __init__(self, num_modules, num_classes, in_channels, feature_dim=128):
        super().__init__(num_modules, num_classes, in_channels, feature_dim)
