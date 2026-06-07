class ModelConfig:
    def __init__(self):
        self.name = 'flux'

        # ----- basic setting ----- #

        self.version = 'flux'
        self.img_size = 1024
        self.feat_size = 512 // 8
        self.t = 150

        # ----- running setting ----- #

        self.cross_attn_setting = {
            'task': 'goca',

            # head aggregation
            'head_method': 'average',
            # 'head_method': 'l2-norm',
            # 'head_method': 'cosine',
            # 'head_method': 'dot-product',
            # 'head_method': 'dot-product w/o clamp',
            # 'head_method': 'cosine + VW',
            # 'head_method': 'dot-product + VW',

            # layer aggregation
            # 'layer_method': 'dot-product similarity',
            # 'layer_method': 'mse',
            # 'layer_method': 'iou-like',
            'layer_method': 'vanilla',

            # rescaling
            # 'rescale_method': 'raw',
            # 'rescale_method': 'per-token renorm',
            'rescale_method': 'per-token renorm+',
            # 'rescale_method': 'sum-1 rescaling',
            # 'rescale_method': 'sum-1 rescaling + per-token renorm',
            # 'rescale_method': 'sum-1 rescaling + per-token renorm+',
            # 'rescale_method': 'sum-1 rescaling + per-token renorm x raw',
            # 'rescale_method': 'sum-1 rescaling + per-token renorm+ x raw',
            # 'rescale_method': 'sum-1 rescaling + per-token renorm+ x raw + renorm',

            'ref_layer': 'vit-block27-attn-out',
            'layer': [
                # 'vit-block0-cross-map',
                # 'vit-block1-cross-map',
                # 'vit-block2-cross-map',
                # 'vit-block3-cross-map',
                # 'vit-block4-cross-map',
                # 'vit-block5-cross-map',
                # 'vit-block6-cross-map',
                # 'vit-block7-cross-map',
                'vit-block8-cross-map',
                # 'vit-block9-cross-map',
                # 'vit-block10-cross-map',
                # 'vit-block11-cross-map',
                # 'vit-block12-cross-map',
                # 'vit-block13-cross-map',
                # 'vit-block14-cross-map',
                # 'vit-block15-cross-map',
                # 'vit-block16-cross-map',
                # 'vit-block17-cross-map',
                # 'vit-block18-cross-map',

                # 'vit-block19-cross-map',
                # 'vit-block20-cross-map',
                # 'vit-block21-cross-map',
                # 'vit-block22-cross-map',
                # 'vit-block23-cross-map',
                # 'vit-block24-cross-map',
                # 'vit-block25-cross-map',
                # 'vit-block26-cross-map',
                # 'vit-block27-cross-map',
                'vit-block28-cross-map',
                # 'vit-block29-cross-map',
                # 'vit-block30-cross-map',
                # 'vit-block31-cross-map',
                # 'vit-block32-cross-map',
                # 'vit-block33-cross-map',
                # 'vit-block34-cross-map',
                # 'vit-block35-cross-map',
                # 'vit-block36-cross-map',
                # 'vit-block37-cross-map',

                # 'vit-block38-cross-map',
                # 'vit-block39-cross-map',
                # 'vit-block40-cross-map',
                # 'vit-block41-cross-map',
                # 'vit-block42-cross-map',
                # 'vit-block43-cross-map',
                # 'vit-block44-cross-map',
                # 'vit-block45-cross-map',
                # 'vit-block46-cross-map',
                'vit-block47-cross-map',
                # 'vit-block48-cross-map',
                # 'vit-block49-cross-map',
                # 'vit-block50-cross-map',
                # 'vit-block51-cross-map',
                # 'vit-block52-cross-map',
                # 'vit-block53-cross-map',
                # 'vit-block54-cross-map',
                # 'vit-block55-cross-map',
                # 'vit-block56-cross-map',
            ],
        }

        self.space_attn_setting = {
            'task': 'self_attn',
            'layer': [
                'vit-block8-self-map',
                'vit-block28-self-map',
                'vit-block47-self-map',
            ],
        }

        self.postprocess_setting = [
            {
                'task': 'affinity',
                'order': 2,
            },
        ]

        self.downstream_setting = {
            'task': 'quantitative_evaluation',
            'save_path_root': '../run_output/flux-vanilla',
        }
