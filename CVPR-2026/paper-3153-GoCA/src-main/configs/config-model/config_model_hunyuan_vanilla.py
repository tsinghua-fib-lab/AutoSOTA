class ModelConfig:
    def __init__(self):
        self.name = 'hunyuan'

        # ----- basic setting ----- #

        self.version = 'hunyuan'
        self.img_size = 1024
        self.feat_size = 512 // 8
        self.t = 100

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

            'ref_layer': 'vit-block25-cross-q',

            'layer': [
                # 'vit-block0-cross-map',
                # 'vit-block1-cross-map',
                # 'vit-block2-cross-map',
                # 'vit-block3-cross-map',
                # 'vit-block4-cross-map',
                # 'vit-block5-cross-map',
                # 'vit-block6-cross-map',
                # 'vit-block7-cross-map',
                # 'vit-block8-cross-map',
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
                'vit-block20-cross-map',
                'vit-block21-cross-map',
                'vit-block22-cross-map',
                'vit-block23-cross-map',
                'vit-block24-cross-map',
                'vit-block25-cross-map',
                'vit-block26-cross-map',
                'vit-block27-cross-map',
                'vit-block28-cross-map',
                'vit-block29-cross-map',
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
            ],
        }

        self.space_attn_setting = {
            'task': 'self_attn',
            'layer': [
                'vit-block20-self-map',
                'vit-block21-self-map',
                'vit-block22-self-map',
                'vit-block23-self-map',
                'vit-block24-self-map',
                'vit-block25-self-map',
                'vit-block26-self-map',
                'vit-block27-self-map',
                'vit-block28-self-map',
                'vit-block29-self-map',
            ],
        }

        self.postprocess_setting = [
            {
                'task': 'affinity',
                'order': 4,
            },
        ]

        self.downstream_setting = {
            'task': 'quantitative_evaluation',
            'save_path_root': '../run_output/hunyuan-vanilla',
        }
