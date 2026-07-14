from llama_model import VaLlamaForCausalLM
from vappt_llama_model import VaPPTLlamaForCausalLM
from torch import nn
import torch
from transformers import LlamaTokenizer, AutoTokenizer
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.checkpoint import checkpoint
from qwen_model import VaQwen3ForCausalLM
import os
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")
class EncoderWrapperSupervised(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)

        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 你之前就是这样写的，保持不动
        )

        # all_values 是一个长度为 num_layers 的列表，每个元素 [batch, seq_len, hidden]
        # 先堆成 [num_layers, batch, seq_len, hidden]
        all_values = torch.stack(output.all_values, dim=0)

        # 选取后 16 层，然后在层维度做平均 -> [batch, seq_len, hidden]
        # 如果总层数少于 16，这里可以改成 all_values[-16:] 也没有问题
        last_values = all_values[-16:]          # [16, B, T, H] 假设层数 >= 16
        token_values = last_values.mean(dim=0)  # [B, T, H]

        # 基于 attention_mask 的 mean pooling
        # attention_mask: [B, T] -> [B, T, 1]
        mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]

        # 将 padding 位置置零
        masked_values = token_values * mask          # [B, T, H]

        # 对序列维度求和
        sum_values = masked_values.sum(dim=1)        # [B, H]

        # 每个样本的有效 token 数
        lengths = mask.sum(dim=1)                    # [B, 1]
        lengths = lengths.clamp(min=1e-6)            # 避免除零

        # masked mean pooling: [B, H]
        embedding = sum_values / lengths

        return embedding  # 不在这里做归一化，在外面再 normalize


class EncoderWrapperSupervisedLastToken(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)

        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 你之前就是这样写的，保持不动
        )

        embedding = output.hidden_states[-1] #[B, T, H]

        lengths = attention_mask.sum(dim=1)          # [B]
        last_indices = (lengths - 1).clamp(min=0)    # [B]，最后一个有效 token 的下标

        batch_size = embedding.size(0)

        # 取每个样本最后一个有效 token 的 embedding
        last_token_embeddings = embedding[
            torch.arange(batch_size, device=embedding.device),
            last_indices
        ]  # [B, H]

        
        return last_token_embeddings  # 不在这里做归一化，在外面再 normalize
    

class EncoderWrapperSupervisedMeanPooling(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)

        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 你之前就是这样写的，保持不动
        )

        embedding = output.hidden_states[-1] #[B, T, H]

        # 将 attention_mask 扩展到 [B, T, 1]，并转成和 embedding 相同的 dtype
        mask = attention_mask.unsqueeze(-1).type_as(embedding)  # [B, T, 1]

        # 对 padding 位置置零，再对时间维求和
        masked_embeddings = embedding * mask  # [B, T, H]
        sum_embeddings = masked_embeddings.sum(dim=1)  # [B, H]

        # 有效 token 个数，用于做平均
        lengths = mask.sum(dim=1)  # [B, 1]
        lengths = lengths.clamp(min=1.0)  # 避免除以 0

        mean_embeddings = sum_embeddings / lengths  # [B, H]

        # 不在这里做归一化，在外面再 normalize
        return mean_embeddings



class EncoderWrapperVAPPT(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        # attention_masks: [batch, seq_len]，1 为有效 token，0 为 padding
        attention_mask = attention_masks.to(input_ids.device)
        ppt_begin_pos = attention_mask.sum(dim=1, keepdim=True)
        ppt_end_pos = attention_mask.sum(dim=1, keepdim=True) + 64
        # 调用底层编码器，VaLlamaForCausalLM 需要保证返回 all_values
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  
            ppt_begin_pos=ppt_begin_pos,
            ppt_end_pos=ppt_end_pos
        )
        last16 = (output.all_va_outputs.permute(0, 1, 3, 2, 4).reshape(32, attention_masks.size(0), attention_masks.size(1) + 64, 32*128))[-16:]
        value = torch.stack(output.va_value_states, dim=0).permute(0, 1, 3, 2, 4).reshape(32, attention_masks.size(0), attention_masks.size(1) + 64, 32*128)[-16:]
        all_layer_embeddings = []
        for j in range(ppt_begin_pos.shape[0]):
            now_begin = ppt_begin_pos[j][0] + 32
            now_embeddings = []
            value_embeddings = []
            for i in range(16):
                now_embeddings.append(last16[i, j, i + now_begin])
                value_embeddings.append(value[i, j, i + now_begin + 16])
            all_layer_embeddings.append(torch.mean(torch.stack(now_embeddings, dim=0), dim=0) + torch.mean(torch.stack(value_embeddings, dim=0), dim=0))
        return torch.stack(all_layer_embeddings, dim=0)



class Value_Aggregation_Gather(nn.Module):
    def __init__(self, local_rank=0) -> None:
        super().__init__()
        model_name = "Qwen/Qwen3-8B"
        if "llama" in model_name:
            self.model = VaLlamaForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        else:
            self.model = VaQwen3ForCausalLM.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=use_auth_token
            )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1

    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks(
            negative_input_ids, negative_attention_mask
        )

        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )




class VAPPT_Gather(nn.Module):
    def __init__(self, local_rank=0, o_layer=False) -> None:
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = VaPPTLlamaForCausalLM.from_pretrained(
            model_name,
            o_layer=o_layer,
            use_auth_token=use_auth_token
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        self.model.initialize_prompt(64)
        self.o_layer = o_layer

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperVAPPT(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1

    def _encode_in_chunks_vappt(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def forward(
        self,
        query_input_ids,
        positive_input_ids,
        negative_input_ids,
        query_attention_mask,
        positive_attention_mask,
        negative_attention_mask,
    ):
        query_embedding = self._encode_in_chunks_vappt(
            query_input_ids, query_attention_mask
        )
        positive_embedding = self._encode_in_chunks_vappt(
            positive_input_ids, positive_attention_mask
        )
        negative_embedding = self._encode_in_chunks_vappt(
            negative_input_ids, negative_attention_mask
        )

        # E5 / InfoNCE 风格：句向量 L2 归一化，用归一化后的点积当余弦
        query_weight_embedding_norm = F.normalize(query_embedding, p=2, dim=-1)
        positive_weight_embedding_norm = F.normalize(positive_embedding, p=2, dim=-1)
        negative_weight_embedding_norm = F.normalize(negative_embedding, p=2, dim=-1)

        return (
            query_weight_embedding_norm,
            positive_weight_embedding_norm,
            negative_weight_embedding_norm,
        )

class Value_Aggregation_Eval(nn.Module):
    def __init__(self, local_rank=0) -> None:
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = VaLlamaForCausalLM.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            use_auth_token=use_auth_token
        )
        torch.cuda.set_device(local_rank)
        self.dim = self.model.config.hidden_size
        self.global_step = 0
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        # 给 backbone 注入 LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model_wrapper()

    def _encode_in_chunks(self, input_ids, attention_mask):
        """
        手动按 batch 维切块 + checkpoint，以减小显存占用。
        """
        device = input_ids.device
        dummy_tensor = torch.ones(
            1,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        embeddings = []

        for start in range(0, input_ids.size(0), self.encoder_gpu_train_limit):
            end = start + self.encoder_gpu_train_limit
            sub_input_ids = input_ids[start:end]
            sub_attention_mask = attention_mask[start:end]

            sub_emb = checkpoint(
                self.model,  # EncoderWrapperSupervised
                sub_input_ids,
                sub_attention_mask,
                dummy_tensor,
            )  # [sub_B, H]
            embeddings.append(sub_emb)

        return torch.cat(embeddings, dim=0)  # [B, H]

    def model_wrapper(self):
        # 用 EncoderWrapperSupervised 包一下，forward 输出的是句子级 embedding
        self.model = EncoderWrapperSupervised(self.model)
        self.encoder_gpu_train_limit = 2
        self.temperature = 0.05
        self.scale = 1


    def forward(
        self,
        input_ids,
        attention_mask
    ):
        current_embedding = self._encode_in_chunks(
            input_ids, attention_mask
        )
        return current_embedding