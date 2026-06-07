class ModelConfig:
    def __init__(self):
        self.name = 'xl'

        # ----- basic setting ----- #

        self.version = 'xl'
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

            'ref_layer': 'up-level1-repeat0-vit-block0-cross-q',

            'layer': [
                'down-level1-repeat0-vit-block0-cross-map',
                'down-level1-repeat0-vit-block1-cross-map',
                'down-level1-repeat1-vit-block0-cross-map',
                'down-level1-repeat1-vit-block1-cross-map',

                'down-level2-repeat0-vit-block0-cross-map',
                'down-level2-repeat0-vit-block1-cross-map',
                'down-level2-repeat0-vit-block2-cross-map',
                'down-level2-repeat0-vit-block3-cross-map',
                'down-level2-repeat0-vit-block4-cross-map',
                'down-level2-repeat0-vit-block5-cross-map',
                'down-level2-repeat0-vit-block6-cross-map',
                'down-level2-repeat0-vit-block7-cross-map',
                'down-level2-repeat0-vit-block8-cross-map',
                'down-level2-repeat0-vit-block9-cross-map',

                'down-level2-repeat1-vit-block0-cross-map',
                'down-level2-repeat1-vit-block1-cross-map',
                'down-level2-repeat1-vit-block2-cross-map',
                'down-level2-repeat1-vit-block3-cross-map',
                'down-level2-repeat1-vit-block4-cross-map',
                'down-level2-repeat1-vit-block5-cross-map',
                'down-level2-repeat1-vit-block6-cross-map',
                'down-level2-repeat1-vit-block7-cross-map',
                'down-level2-repeat1-vit-block8-cross-map',
                'down-level2-repeat1-vit-block9-cross-map',

                'mid-vit-block0-cross-map',
                'mid-vit-block1-cross-map',
                'mid-vit-block2-cross-map',
                'mid-vit-block3-cross-map',
                'mid-vit-block4-cross-map',
                'mid-vit-block5-cross-map',
                'mid-vit-block6-cross-map',
                'mid-vit-block7-cross-map',
                'mid-vit-block8-cross-map',
                'mid-vit-block9-cross-map',

                'up-level0-repeat0-vit-block0-cross-map',
                'up-level0-repeat0-vit-block1-cross-map',
                'up-level0-repeat0-vit-block2-cross-map',
                'up-level0-repeat0-vit-block3-cross-map',
                'up-level0-repeat0-vit-block4-cross-map',
                'up-level0-repeat0-vit-block5-cross-map',
                'up-level0-repeat0-vit-block6-cross-map',
                'up-level0-repeat0-vit-block7-cross-map',
                'up-level0-repeat0-vit-block8-cross-map',
                'up-level0-repeat0-vit-block9-cross-map',

                'up-level0-repeat1-vit-block0-cross-map',
                'up-level0-repeat1-vit-block1-cross-map',
                'up-level0-repeat1-vit-block2-cross-map',
                'up-level0-repeat1-vit-block3-cross-map',
                'up-level0-repeat1-vit-block4-cross-map',
                'up-level0-repeat1-vit-block5-cross-map',
                'up-level0-repeat1-vit-block6-cross-map',
                'up-level0-repeat1-vit-block7-cross-map',
                'up-level0-repeat1-vit-block8-cross-map',
                'up-level0-repeat1-vit-block9-cross-map',

                'up-level0-repeat2-vit-block0-cross-map',
                'up-level0-repeat2-vit-block1-cross-map',
                'up-level0-repeat2-vit-block2-cross-map',
                'up-level0-repeat2-vit-block3-cross-map',
                'up-level0-repeat2-vit-block4-cross-map',
                'up-level0-repeat2-vit-block5-cross-map',
                'up-level0-repeat2-vit-block6-cross-map',
                'up-level0-repeat2-vit-block7-cross-map',
                'up-level0-repeat2-vit-block8-cross-map',
                'up-level0-repeat2-vit-block9-cross-map',

                'up-level1-repeat0-vit-block0-cross-map',
                'up-level1-repeat0-vit-block1-cross-map',
                'up-level1-repeat1-vit-block0-cross-map',
                'up-level1-repeat1-vit-block1-cross-map',
                'up-level1-repeat2-vit-block0-cross-map',
                'up-level1-repeat2-vit-block1-cross-map',
            ],
        }

        self.space_attn_setting = {
            'task': 'self_attn',
            'layer': [
                'up-level1-repeat0-vit-block0-self-map',
                'up-level1-repeat0-vit-block1-self-map',
                'up-level1-repeat1-vit-block0-self-map',
                'up-level1-repeat1-vit-block1-self-map',
                'up-level1-repeat2-vit-block0-self-map',

                'down-level1-repeat0-vit-block0-self-map',
                'down-level1-repeat0-vit-block1-self-map',
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
            'save_path_root': '../run_output/xl-vanilla',
        }
