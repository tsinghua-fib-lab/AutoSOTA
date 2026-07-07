from collections import OrderedDict
from typing import Callable, List

import torch
import torch.nn as nn
from loguru import logger

import clip
from utils import process_class_names


def get_cls_embddings(
    text_tokenizer: Callable,
    text_embedding: torch.nn.Module,
    class_names: List,
    number_of_prompts: int,
    context_size: int,
    pad_emb: torch.Tensor
):

    tokenized_cls_names = [
        text_tokenizer(process_class_names(cls_name)) for cls_name in class_names
    ]

    eot_token_pos = torch.tensor(
        [number_of_prompts + (i > 0).sum().item() -
            1 for i in tokenized_cls_names]
    )

    length_cls_names = [(i > 0).sum().item() for i in tokenized_cls_names]

    # create duplicate padding to get to context_size after adding prompts
    per_cls_padding = [
        torch.cat(
            [pad_emb] * (context_size - length - number_of_prompts), dim=1
        )
        for length in length_cls_names
    ]

    embedded_cls_names_with_special_tokens = [text_embedding(
        tokens) for tokens in tokenized_cls_names]

    # only keep class emb, remove sot, eot and padding
    cls_lengths = [i - 2 for i in length_cls_names]

    embedded_cls_names = []
    for emb, length in zip(embedded_cls_names_with_special_tokens, cls_lengths):
        cut_emb = emb[:, 1: 1 + length]

        embedded_cls_names.append(
            cut_emb[None] if cut_emb.ndim == 2 else cut_emb
        )

    return embedded_cls_names, per_cls_padding, cls_lengths, eot_token_pos


def get_prompts_positions(
    prompt_position: str,
    num_classes: int,
    cls_lengths: int,
    number_of_prompts: int
):
    # class token position
    if prompt_position == "end":
        starts = [1] * num_classes
    elif prompt_position == "middle":
        starts = [1 + length // 2 for length in cls_lengths]
    else:
        starts = [1 + length for length in cls_lengths]

    cls_indx = (
        torch.arange(num_classes)
        .view(-1, 1)
        .repeat(1, number_of_prompts)
    )
    prompt_positions = torch.stack(
        [
            torch.arange(start, start + number_of_prompts)
            for start in starts
        ]
    )
    cond_positions = (cls_indx, prompt_positions)

    return cond_positions


class Prompter(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        train_class_names: List,
        test_class_names: List,
        clip_model: int,
        per_class_prompts: bool,
        prompt_position: str,
        number_of_prompts: int,
        use_conditioning: bool,
    ):
        super().__init__()

        assert prompt_position in ["start", "middle", "end"]

        self.zero_shot_case = not all(x == y for x, y in zip(
            train_class_names, test_class_names))

        if self.zero_shot_case:
            assert not per_class_prompts, \
                "Per class prompts not supported for zero shot case"

        text_tokenizer = clip.tokenize
        text_embedding = clip_model.token_embedding

        dummy_text = text_tokenizer([""])
        dummy_emb = text_embedding(dummy_text)
        _, self.context_size, emb_dim = dummy_emb.size()

        # dummy_text is tokenized as [sot, eot, pad, pad ....]
        self.sot_emb = dummy_emb[:, 0].unsqueeze(1)
        self.eot_emb = dummy_emb[:, 1].unsqueeze(1)
        pad_emb = dummy_emb[:, 2].unsqueeze(1)

        self.embedded_cls_names, self.per_cls_padding, \
            cls_lengths, self._eot_token_pos = get_cls_embddings(
                text_tokenizer=text_tokenizer,
                text_embedding=text_embedding,
                class_names=train_class_names,
                number_of_prompts=number_of_prompts,
                context_size=self.context_size,
                pad_emb=pad_emb
            )

        if self.zero_shot_case:
            # train and test classes are different
            self.test_embedded_cls_names, self.test_per_cls_padding, \
                test_cls_lengths, self._test_eot_token_pos = get_cls_embddings(
                    text_tokenizer=text_tokenizer,
                    text_embedding=text_embedding,
                    class_names=test_class_names,
                    number_of_prompts=number_of_prompts,
                    context_size=self.context_size,
                    pad_emb=pad_emb
                )

        # create learnable prompts
        self.train_num_classes = len(train_class_names)
        self.test_num_classes = len(test_class_names)
        self.prompt_position = prompt_position
        self.use_conditioning = use_conditioning
        self.per_class_prompts = per_class_prompts
        self.number_of_prompts = number_of_prompts

        if per_class_prompts:
            logger.info("Using per class prompts")
            prompt_shape = (self.num_classes, self.number_of_prompts, emb_dim)
        else:
            prompt_shape = (1, self.number_of_prompts, emb_dim)

        prompts = torch.randn(*prompt_shape)
        nn.init.normal_(prompts, std=0.01)
        self.prompts = nn.Parameter(prompts)

        if self.use_conditioning:
            logger.info("Using conditioning")
            vis_emb_dim = clip_model.visual.output_dim

            self.hyper_cond_net = nn.Sequential(
                OrderedDict(
                    [
                        ("linear1", nn.Linear(vis_emb_dim, vis_emb_dim // 16)),
                        ("relu", nn.ReLU(inplace=True)),
                        ("linear2", nn.Linear(vis_emb_dim // 16, emb_dim)),
                    ]
                )
            )

            self.cond_positions = get_prompts_positions(
                prompt_position=self.prompt_position,
                num_classes=self.train_num_classes,
                cls_lengths=cls_lengths,
                number_of_prompts=self.number_of_prompts
            )

            if self.zero_shot_case:
                self.test_cond_positions = get_prompts_positions(
                    prompt_position=self.prompt_position,
                    num_classes=self.test_num_classes,
                    cls_lengths=test_cls_lengths,
                    number_of_prompts=self.number_of_prompts
                )

    @property
    def eot_token_pos(self):
        if self.zero_shot_case and (not self.training):
            return self._test_eot_token_pos

        return self._eot_token_pos

    @property
    def trainable_parameters(self):
        if self.use_conditioning:
            return ["hyper_cond_net", "prompts"]
        return ["prompts"]

    def send_to_device(self, device):
        self.sot_emb = self.sot_emb.to(device)
        self.eot_emb = self.eot_emb.to(device)
        self.embedded_cls_names = [i.to(device)
                                   for i in self.embedded_cls_names]
        self.per_cls_padding = [i.to(device) for i in self.per_cls_padding]

        if self.zero_shot_case:
            self.test_embedded_cls_names = [i.to(device)
                                            for i in self.test_embedded_cls_names]
            self.test_per_cls_padding = [i.to(device)
                                         for i in self.test_per_cls_padding]

    def create_prompt(self):
        if self.prompts.device != self.sot_emb.device:
            self.send_to_device(self.prompts.device)

        num_classes = self.train_num_classes if self.training else self.test_num_classes

        if self.zero_shot_case and (not self.training):
            embedded_cls_names = self.test_embedded_cls_names
            per_cls_padding = self.test_per_cls_padding
        else:
            embedded_cls_names = self.embedded_cls_names
            per_cls_padding = self.per_cls_padding

        # the position of the class token
        if self.prompt_position == "end":
            text_emb = []
            for i in range(num_classes):
                text_emb.append(
                    torch.cat(
                        [
                            self.sot_emb,
                            self.prompts[i][None] if self.per_class_prompts else self.prompts,
                            embedded_cls_names[i],
                            self.eot_emb,
                            per_cls_padding[i],
                        ],
                        dim=1,
                    )
                )
            text_emb = torch.cat(text_emb)

        elif self.prompt_position == "middle":
            text_emb = []
            for i in range(num_classes):
                half_prompts = self.number_of_prompts // 2

                first_half = (
                    self.prompts[i, :half_prompts][None]
                    if self.per_class_prompts
                    else self.prompts[:, :half_prompts]
                )
                sec_half = (
                    self.prompts[i, half_prompts:][None]
                    if self.per_class_prompts
                    else self.prompts[:, half_prompts:]
                )

                text_emb.append(
                    torch.cat(
                        [
                            self.sot_emb,
                            first_half,
                            embedded_cls_names[i],
                            sec_half,
                            self.eot_emb,
                            per_cls_padding[i],
                        ],
                        dim=1,
                    )
                )

            text_emb = torch.cat(text_emb)

        else:
            text_emb = []
            for i in range(num_classes):
                text_emb.append(
                    torch.cat(
                        [
                            self.sot_emb,
                            embedded_cls_names[i],
                            self.prompts[i][None] if self.per_class_prompts else self.prompts,
                            self.eot_emb,
                            per_cls_padding[i],
                        ],
                        dim=1,
                    )
                )
            text_emb = torch.cat(text_emb)

        return text_emb

    def forward(
        self,
        image_features=None,
    ):
        prompts = self.create_prompt()

        if self.use_conditioning:
            assert image_features is not None

            num_classes = self.train_num_classes if \
                self.training else self.test_num_classes

            # batch, dim
            img_cond = self.hyper_cond_net(image_features)
            # batch, 1, 1, dim
            img_cond = img_cond.unsqueeze(1).unsqueeze(1)
            # batch, n_classes, num_prompts, dim
            img_cond = img_cond.repeat(
                1, num_classes, self.number_of_prompts, 1)

            # n_classes, context, dim
            prompts = prompts[None]
            # batch, n_classes, context, dim
            prompts = prompts.repeat(img_cond.size(0), 1, 1, 1)

            if self.zero_shot_case and (not self.training):
                cond_positions = self.test_cond_positions
            else:
                cond_positions = self.cond_positions

            # add the conditioning to the prompts positions
            for batch in range(img_cond.size(0)):
                prompts[batch][cond_positions] += img_cond[batch]

        return prompts
