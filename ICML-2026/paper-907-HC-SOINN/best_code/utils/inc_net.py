import copy
import logging
import torch
from torch import nn
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, SimpleContinualLinear, TunaLinear, CosineLinearFeature

from backbone.prompt import CodaPrompt
from utils.kac_classifier import KACLayer
import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    # SimpleCIL or SimpleCIL w/ Finetune
    if name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    elif '_memo' in name:
        if args["model_name"] == "memo":
            from backbone import vit_memo
            _basenet, _adaptive_net = timm.create_model("vit_base_patch16_224_memo", pretrained=True, num_classes=0)
            _basenet.out_dim = 768
            _adaptive_net.out_dim = 768
            return _basenet, _adaptive_net
    # SSF 
    elif '_ssf' in name:
        if args["model_name"] == "aper_ssf"  or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone import vit_ssf
            if name == "pretrained_vit_b16_224_ssf":
                model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_ssf":
                model = timm.create_model("vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0)
                model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    
    # VPT
    elif '_vpt' in name:
        if args["model_name"] == "aper_vpt"  or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone.vpt import build_promptmodel
            if name == "pretrained_vit_b16_224_vpt":
                basicmodelname = "vit_base_patch16_224" 
            elif name == "pretrained_vit_b16_224_in21k_vpt":
                basicmodelname = "vit_base_patch16_224_in21k"
            
            print("modelname,", name, "basicmodelname", basicmodelname)
            VPT_type = "Deep"
            if args["vpt_type"] == 'shallow':
                VPT_type = "Shallow"
            Prompt_Token_num = args["prompt_token_num"]

            model = build_promptmodel(modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type)
            prompt_state_dict = model.obtain_prompt()
            model.load_prompt(prompt_state_dict)
            model.out_dim = 768
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    elif '_adapter' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "sema":
            from backbone import vit_sema
            from easydict import EasyDict
            device = args["device"][0] if isinstance(args["device"], list) else args["device"]
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                ffn_adapter_type=args.get("ffn_adapter_type", "adaptmlp"),
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                exp_threshold=args.get("exp_threshold", 2),
                adapt_start_layer=args.get("adapt_start_layer", 9),
                adapt_end_layer=args.get("adapt_end_layer", 11),
                rd_dim=args.get("rd_dim", 128),
                buffer_size=args.get("buffer_size", 500),
            )
            if name == "pretrained_vit_b16_224_adapter" or name == "vit_base_patch16_224_adapter":
                model = vit_sema.vit_base_patch16_224_sema(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config, device=device)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter" or name == "vit_base_patch16_224_in21k_adapter":
                model = vit_sema.vit_base_patch16_224_in21k_sema(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config, device=device)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        elif args["model_name"] == "aper_adapter" or args["model_name"] == "ranpac" or args["model_name"] == "fecam":
            from backbone import vit_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vit_adapter.vit_base_patch16_224_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vit_adapter.vit_base_patch16_224_in21k_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # L2P
    elif '_l2p' in name:
        if args["model_name"] == "l2p":
            from backbone import vit_l2p
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # dualprompt
    elif '_dualprompt' in name:
        if args["model_name"] == "dualprompt":
            from backbone import vit_dualprompt
            use_kac = args.get("use_kac", False)
            kac_config = args.get("kac_config", {})
            
            model = timm.create_model(
                args["backbone_type"],
                pretrained=args["pretrained"],
                num_classes=args["nb_classes"],
                drop_rate=args["drop"],
                drop_path_rate=args["drop_path"],
                drop_block_rate=None,
                prompt_length=args["length"],
                embedding_key=args["embedding_key"],
                prompt_init=args["prompt_key_init"],
                prompt_pool=args["prompt_pool"],
                prompt_key=args["prompt_key"],
                pool_size=args["size"],
                top_k=args["top_k"],
                batchwise_prompt=args["batchwise_prompt"],
                prompt_key_init=args["prompt_key_init"],
                head_type=args["head_type"],
                use_prompt_mask=args["use_prompt_mask"],
                use_g_prompt=args["use_g_prompt"],
                g_prompt_length=args["g_prompt_length"],
                g_prompt_layer_idx=args["g_prompt_layer_idx"],
                use_prefix_tune_for_g_prompt=args["use_prefix_tune_for_g_prompt"],
                use_e_prompt=args["use_e_prompt"],
                e_prompt_layer_idx=args["e_prompt_layer_idx"],
                use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_e_prompt"],
                same_key_value=args["same_key_value"],
                use_kac=use_kac,
                kac_config=kac_config,
            )
            return model
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    # Coda_Prompt
    elif '_coda_prompt' in name:
        if args["model_name"] == "coda_prompt":
            from backbone import vit_coda_promtpt
            model = timm.create_model(args["backbone_type"], pretrained=args["pretrained"])
            # model = vision_transformer_coda_prompt.VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
            #                 num_heads=12, ckpt_layer=0,
            #                 drop_path_rate=0)
            # from timm.models import vit_base_patch16_224
            # load_dict = vit_base_patch16_224(pretrained=True).state_dict()
            # del load_dict['head.weight']; del load_dict['head.bias']
            # model.load_state_dict(load_dict)
            return model
    elif '_ease' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ease" :
            from backbone import vit_ease
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_ease":
                model = vit_ease.vit_base_patch16_224_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_ease":
                model = vit_ease.vit_base_patch16_224_in21k_ease(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif '_lae' in name:
        from backbone import vit_lae
        model = timm.create_model(args["backbone_type"], pretrained=True)
        return model
    elif '_cllora' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "cllora" :
            from backbone import vit_cllora
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                use_distillation = args["use_distillation"],
                use_block_weight = args["use_block_weight"],
                msa_adapt = args["msa_adapt"],
                msa = args["msa"],
                specfic_pos = args["specfic_pos"],
                general_pos = args["general_pos"],
                attn_bn = ffn_num,  # Use ffn_num as attn_bn
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_cllora":
                model = vit_cllora.vit_base_patch16_224_cllora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            elif name == "vit_base_patch16_224_in21k_cllora":
                model = vit_cllora.vit_base_patch16_224_in21k_cllora(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")
    elif '_mos' in name:
        ffn_num = args["ffn_num"]
        if args["model_name"] == "mos":
            from backbone import vit_mos
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                _device = args["device"][0],
                adapter_momentum = args["adapter_momentum"],
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "vit_base_patch16_224_mos":
                model = vit_mos.vit_base_patch16_224_mos(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            elif name == "vit_base_patch16_224_in21k_mos":
                model = vit_mos.vit_base_patch16_224_in21k_mos(num_classes=args["nb_classes"],
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model
    elif '_tuna' in name:
        ffn_num = 16
        from backbone import vit_tuna
        from easydict import EasyDict
        tuning_config = EasyDict(
            # AdaptFormer
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            d_model=768,
            _device=args["device"][0],
            # VPT related
            vpt_on=False,
            vpt_num=0,
        )
        if name == "vit_base_patch16_224_tuna":
            model = vit_tuna.vit_base_patch16_224_adapter(num_classes=args["nb_classes"],
                                                             global_pool=False, drop_path_rate=0.0,
                                                             tuning_config=tuning_config,
                                                             r=args["r"])
        elif name == "vit_base_patch16_224_in21k_tuna":
            model = vit_tuna.vit_base_patch16_224_in21k_adapter(num_classes=args["nb_classes"],
                                                                   global_pool=False, drop_path_rate=0.0,
                                                                   tuning_config=tuning_config,
                                                                   r=args["r"])
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x["features"])
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.backbone.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.backbone.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num  ==  1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc

class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logits: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.backbones) == 0:
            self.backbones.append(get_backbone(self.args, self.pretrained))
        else:
            self.backbones.append(get_backbone(self.args, self.pretrained))
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        # for RanPAC
        self.W_rand = None
        self.RP_dim = None

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        if self.RP_dim is not None:
            feature_dim = self.RP_dim
        else:
            feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        if self.W_rand is not None:
            x = torch.nn.functional.relu(x @ self.W_rand)
        out = self.fc(x)
        out.update({"features": x})
        return out

class SEMAVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args
        self.ncm_fc = CosineLinear(768, args["nb_classes"], sigma=False)

    @property
    def feature_dim(self):
        """Handle feature dim."""
        return 768

    def extract_vector(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            return out["features"]
        return out

    def forward(self, x):
        out = self.backbone(x)
        x = out["features"]
        out.update({"logits": self.fc(x)})
        return out

# SimpleCIL-KNN
class SimpleVitNetKNN(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        pass

    def generate_fc(self, in_dim, out_dim):
        return None

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        out = {"features": x}
        return out

# l2p and dualprompt
from backbone.linears import SimpleLinear, SplitCosineLinear, CosineLinear, EaseCosineLinear, SimpleContinualLinear, TunaLinear, CosineLinearFeature
from utils.kac_classifier import KACLayer
import copy

class PromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(PromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        if args["get_original_backbone"]:
            self.original_backbone = self.get_original_backbone(args)
        else:
            self.original_backbone = None
        
        self.ncm_fc = CosineLinear(768, args["nb_classes"], sigma=False)
        self.feature_dim = 768
            
    def get_original_backbone(self, args):
        return timm.create_model(
            args["backbone_type"],
            pretrained=args["pretrained"],
            num_classes=args["nb_classes"],
            drop_rate=args["drop"],
            drop_path_rate=args["drop_path"],
            drop_block_rate=None,
        ).eval()

    def forward(self, x, task_id=-1, train=False):
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None

        x = self.backbone(x, task_id=task_id, cls_features=cls_features, train=train)
        return x

    def extract_vector(self, x, task_id=-1):
        """Handle extract vector."""
        with torch.no_grad():
            if self.original_backbone is not None:
                cls_features = self.original_backbone(x)['pre_logits']
            else:
                cls_features = None
        
        res = self.backbone(x, task_id=task_id, cls_features=cls_features, train=False)
        if isinstance(res, dict):
            if 'features' in res:
                return res['features']
            return res['pre_logits']
        return res

# coda_prompt
class CodaPromptVitNet(nn.Module):
    def __init__(self, args, pretrained):
        super(CodaPromptVitNet, self).__init__()
        self.args = args
        self.backbone = get_backbone(args, pretrained)
        
        self.use_kac = args.get("use_kac", False)
        kac_config = args.get("kac_config", {})
        
        if self.use_kac:
            logging.info("Using KAC classifier instead of linear classifier")
            self.fc = KACLayer(
                input_dim=768,
                output_dim=args["nb_classes"],
                grid_min=kac_config.get("grid_min", -2.0),
                grid_max=kac_config.get("grid_max", 2.0),
                num_grids=kac_config.get("num_grids", 16),
                spline_weight_init_scale=kac_config.get("spline_weight_init_scale", 0.1),
            )
        else:
            self.fc = nn.Linear(768, args["nb_classes"])
        
        self.prompt = CodaPrompt(768, args["nb_tasks"], args["prompt_param"])
        self.ncm_fc = CosineLinear(768, args["nb_classes"], sigma=False)
        self.feature_dim = 768

    # pen: get penultimate features  
    def forward(self, x, pen=False, train=False):
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:,0,:]
            out, prompt_loss = self.backbone(x, prompt=self.prompt, q=q, train=train)
            out = out[:,0,:]
        else:
            out, _ = self.backbone(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
    
    def extract_vector(self, x):
        """Handle extract vector."""
        if self.prompt is not None:
            with torch.no_grad():
                q, _ = self.backbone(x)
                q = q[:,0,:]
            out, _ = self.backbone(x, prompt=self.prompt, q=q, train=False)
            out = out[:,0,:]
        else:
            out, _ = self.backbone(x)
            out = out[:,0,:]
        out = out.view(out.size(0), -1)
        return out
    
    def update_fc(self, nb_classes):
        """Handle update fc."""
        if self.use_kac:
            if hasattr(self.fc, 'update'):
                self.fc.update(nb_classes)
            else:
                kac_config = self.args.get("kac_config", {})
                old_fc = self.fc
                self.fc = KACLayer(
                    input_dim=768,
                    output_dim=nb_classes,
                    grid_min=kac_config.get("grid_min", -2.0),
                    grid_max=kac_config.get("grid_max", 2.0),
                    num_grids=kac_config.get("num_grids", 16),
                    spline_weight_init_scale=kac_config.get("spline_weight_init_scale", 0.1),
                )
                if hasattr(old_fc, 'spline_linear') and hasattr(self.fc, 'spline_linear'):
                    old_out_dim = old_fc.output_dim
                    if old_out_dim <= nb_classes:
                        self.fc.spline_linear.weight.data[:old_out_dim, :] = old_fc.spline_linear.weight.data
        else:
            if self.fc is not None:
                nb_output = self.fc.out_features
                weight = copy.deepcopy(self.fc.weight.data)
                bias = copy.deepcopy(self.fc.bias.data)
            else:
                nb_output = 0
            
            fc = self.generate_fc(768, nb_classes)
            if self.fc is not None:
                fc.weight.data[:nb_output] = weight
                fc.bias.data[:nb_output] = bias
            
            del self.fc
            self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        """Handle generate fc."""
        if self.use_kac:
            kac_config = self.args.get("kac_config", {})
            return KACLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                grid_min=kac_config.get("grid_min", -2.0),
                grid_max=kac_config.get("grid_max", 2.0),
                num_grids=kac_config.get("num_grids", 16),
                spline_weight_init_scale=kac_config.get("spline_weight_init_scale", 0.1),
            )
        else:
            return nn.Linear(in_dim, out_dim)


class MultiBranchCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        
        # no need the backbone.
        
        print('Clear the backbone in MultiBranchCosineIncrementalNet, since we are using self.backbones with dual branches')
        self.backbone=torch.nn.Identity()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbones = nn.ModuleList()
        self.args=args
        self._feature_dim = None
        
        if 'resnet' in args['backbone_type']:
            self.model_type='cnn'
        else:
            self.model_type='vit'

    @property
    def feature_dim(self):
        """Handle feature dim."""
        if self._feature_dim is not None:
            return self._feature_dim
        if len(self.backbones) > 0:
            return self.backbones[0].out_dim * len(self.backbones)
        return super().feature_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self._feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self._feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]       

        features = torch.cat(features, 1)
        # import pdb; pdb.set_trace()
        out = self.fc(features)
        out.update({"features": features})
        return out

    def extract_vector(self, x):
        """Handle extract vector."""
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features
    
    def construct_dual_branch_network(self, tuned_model):
        if 'ssf' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_ssf','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without scale
        elif 'vpt' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_vpt','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without vpt
        elif 'adapter' in self.args['backbone_type']:
            newargs=copy.deepcopy(self.args)
            newargs['backbone_type']=newargs['backbone_type'].replace('_adapter','')
            print(newargs['backbone_type'])
            self.backbones.append(get_backbone(newargs)) #pretrained model without adapter
        else:
            self.backbones.append(get_backbone(self.args)) #the pretrained model itself

        self.backbones.append(tuned_model.backbone) #adappted tuned model
    
        self._feature_dim = self.backbones[0].out_dim * len(self.backbones) 
        self.fc=self.generate_fc(self._feature_dim,self.args['init_cls'])


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.backbones = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.backbones)

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        if self.model_type == 'cnn':
            features = [backbone(x)["features"] for backbone in self.backbones]
        else:
            features = [backbone(x) for backbone in self.backbones]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.backbones.append(get_backbone(self.args, self.pretrained))
        if self.out_dim is None:
            self.out_dim = self.backbones[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.backbones[-1].load_state_dict(self.backbones[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_backbone(self):
        for param in self.backbones.parameters():
            param.requires_grad = False
        self.backbones.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.backbones) == 1
        self.backbones[0].load_state_dict(model_infos['backbone'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = args["backbone_type"]
        self.TaskAgnosticExtractor , _ = get_backbone(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_backbone(self.args, self.pretrained)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            # logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].out_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["backbone_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['backbone']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class EaseNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = args["use_init_ptm"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]
            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            # print(name)
    
    @property
    def feature_dim(self):
        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)
    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[ : old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
    
    def generate_fc(self, in_dim, out_dim):
        fc = EaseCosineLinear(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
        else:
            x = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x)
            else:
                out = self.fc.forward_reweight(x, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)
            
        out.update({"features": x})
        return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())


class OurNet(BaseNet):
    def __init__(self, args, pretrained=True):
        super().__init__(args, pretrained)
        self.args = args
        self.inc = args["increment"]
        self.init_cls = args["init_cls"]
        self._cur_task = -1
        self.out_dim =  self.backbone.out_dim
        self.fc = None
        self.use_init_ptm = False
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.fc_list = nn.ModuleList()
        self.fc_list_task = nn.ModuleList()
        self.adapter_list = nn.ModuleList()
        self.init_proto = None

            
    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
    
    @property
    def feature_dim(self):

        if self.use_init_ptm:
            return self.out_dim * (self._cur_task + 2)
        else:
            return self.out_dim * (self._cur_task + 1)

    # (proxy_fc = cls * dim)

    def update_fc_task(self):
        self.proxy_fc_task = self.generate_fc(self.out_dim, 1).to(self._device)
        self.fc_list_task.append(self.proxy_fc_task.requires_grad_(True))

    def update_fc(self, nb_classes):
        self._cur_task += 1
        
        if self._cur_task == 0:
            self.proxy_fc = self.generate_fc(self.out_dim, self.init_cls).to(self._device)
        else:
            self.proxy_fc = self.generate_fc(self.out_dim, self.inc).to(self._device)
        init_proto = self.generate_fc(self.out_dim, nb_classes).to(self._device)

        if self.init_proto is not None:
            old_nb_classes = self.init_proto.out_features
            weight = copy.deepcopy(self.init_proto.weight.data)
            init_proto.weight.data[: old_nb_classes, :] = nn.Parameter(weight)
        del self.init_proto
        self.init_proto = init_proto

        
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        fc.reset_parameters_to_zero()
        
        if self.fc is not None:
            old_nb_classes = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            fc.weight.data[: old_nb_classes, : -self.out_dim] = nn.Parameter(weight)
        del self.fc
        self.fc = fc
        self.fc.requires_grad_(False)

    def add_fc(self):
        self.fc_list.append(self.proxy_fc.requires_grad_(False))
        del self.proxy_fc

    def remove_fc_task(self):
        self.fc_list_task.requires_grad_(False)
        del self.proxy_fc_task

    def remove_fc_init(self):
        self.init_proto_list.append(self.init_proto)
        del self.init_proto
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinearFeature(in_dim, out_dim)
        return fc
    
    def extract_vector(self, x):
        return self.backbone(x)

    def forward_kd(self, x, t_idx):
        x_new, x_teacher = self.backbone.forward_general_cls(x, t_idx)
        out_new, out_teacher = self.proxy_fc(x_new), self.proxy_fc(x_teacher)
        return  out_new, out_teacher

    def forward(self, x, test=False):
        if test == False:
            x = self.backbone.forward(x, False)
            out = self.proxy_fc(x)
            out.update({"features": x})
            return out
        else:
            x_input = self.backbone.forward(x, True, use_init_ptm=self.use_init_ptm)
            if self.args["moni_adam"] or (not self.args["use_reweight"]):
                out = self.fc(x_input)
            else:
                out = self.fc.forward_diagonal(x_input, cur_task=self._cur_task, alpha=self.alpha, init_cls=self.init_cls, inc=self.inc, use_init_ptm=self.use_init_ptm, beta=self.beta)

            out.update({"features": x_input})
            return out

    def show_trainable_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel())

class SLCANet(BaseNet):

    def __init__(self, args, pretrained=True, fc_with_ln=False):
        super().__init__(args, pretrained)
        self.old_fc = None
        self.fc_with_ln = fc_with_ln


    def extract_layerwise_vector(self, x, pool=True):
        with torch.no_grad():
            features = self.backbone(x, layer_feat=True)['features']
        for f_i in range(len(features)):
            if pool:
                features[f_i] = features[f_i].mean(1).cpu().numpy() 
            else:
                features[f_i] = features[f_i][:, 0].cpu().numpy() 
        return features


    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def save_old_fc(self):
        if self.old_fc is None:
            self.old_fc = copy.deepcopy(self.fc)
        else:
            self.old_fc.heads.append(copy.deepcopy(self.fc.heads[-1]))

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleContinualLinear(in_dim, out_dim)

        return fc

    def forward(self, x, bcb_no_grad=False, fc_only=False):
        if fc_only:
            fc_out = self.fc(x)
            if self.old_fc is not None:
                old_fc_logits = self.old_fc(x)['logits']
                fc_out['old_logits'] = old_fc_logits
            return fc_out
        if bcb_no_grad:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})

        return out

class LAE(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.backbone = get_backbone(args, pretrained=pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]
        self.args = args
        self.pet_cls = args["pet_cls"]
        self.pet_length = args["pet_length"]
        self.pets_emas = nn.ModuleList([])
        self.adapt_blocks = [0, 1, 2, 3, 4]
        self.down_sample_dim = args["down_sample_dim"]

        self.pets = self.create_pets()
        self.pets_emas = self.create_pets()
        for pq, pk in zip(self.pets.parameters(),self.pets_emas.parameters()):
            pk.data.copy_(pq.data)
            pk.requires_grad=False
        self.attach_pets_vit(self.pets)

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def create_pets(self):
        n = len(self.adapt_blocks)
        embed_dim = self.backbone.embed_dim
        if self.pet_cls == "Prefix":
            from backbone.vit_lae import Prefix
            return nn.ModuleList([Prefix(length=self.pet_length, dim=embed_dim) for _ in range(n)])
        if self.pet_cls == "Adapter":
            from backbone.vit_lae import Adapter
            return nn.ModuleList([Adapter(embed_dim=embed_dim, down_sample=self.down_sample_dim) for _ in range(n)])
        if self.pet_cls == "LoRA":
            from backbone.vit_lae import KVLoRA
            return nn.ModuleList([KVLoRA(in_features=embed_dim, out_features=embed_dim) for _ in range(n)] )

    def attach_pets_vit(self, pets: nn.ModuleList):
        assert self.pet_cls in ["Adapter", "LoRA", "Prefix"]
        
        if self.pet_cls == "Prefix":
            for i, b in enumerate(self.adapt_blocks):
                self.backbone.blocks[b].attn.attach_prefix(pets[i])
            return
        if self.pet_cls == "Adapter":
            for i, b in enumerate(self.adapt_blocks):
                self.backbone.blocks[b].attn.attach_adapter(attn=pets[i])
            return
        if self.cfg.pet_cls == "LoRA":
            for i, b in enumerate(self.cfg.adapt_blocks):
                self.model.backbone.blocks[b].attn.attach_adapter(qkv=pets[i])
            return
        
    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        x = x[:,0,:]
        out = self.fc(x)
        
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

class MOSNet(nn.Module):
    def __init__(self, args, pretrained):
        super(MOSNet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc.requires_grad_(False)
    
    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc
    
    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)['features']
        
        res = dict()
        res['features'] = features
        res['logits'] = self.fc(features)['logits']
                
        return res
        
    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        res = self.backbone(x, adapter_id, train, fc_only)

        return res

class TUNANet(nn.Module):
    def __init__(self, args, pretrained):
        super(TUNANet, self).__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = 768
        self.fc = None
        self._device = args["device"][0]
       

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes,nextperiod_initialization=None):
      
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def generate_fc(self, in_dim, out_dim): 
        fc = TunaLinear(in_dim, out_dim)
        return fc

    def forward_orig(self, x):
        features = self.backbone(x, adapter_id=0)['features']

        res = dict()
        res['features'] = features
        res['logits'] = self.fc(features)['logits']

        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        res = self.backbone(x, adapter_id, train, fc_only)
        return res