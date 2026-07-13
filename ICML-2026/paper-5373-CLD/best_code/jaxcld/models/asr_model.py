import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForCTC, AutoProcessor, AutoModelForAudioClassification

import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk
import torch
import torchaudio
from typing import Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

dtype = torch.float16

ISO2_TO_ISO3 = {
    "en": "eng",
    "zh": "zho",
    "hi": "hin",
    "id": "ind",
    "ms": "zlm"
}

ISO3_TO_ISO2 = {
    "cdo": "zh",
    "cmn": "zh",
    "cpx": "zh",
    "czh": "zh",
    "hak": "zh",
    "hsn": "zh",
    "mnp": "zh",
    "nan": "zh",
    "wuu": "zh",
    "yue": "zh",
    "zho": "zh",
    "eng": "en",
    "hin": "hi",
    "zlm": "ms",
    "ind": "id"
}

# Used by MMS adapter selection to try multiple script-specific adapters.
POSSIBLE_SCRIPTS = ["", "Latn", "Cyrl", "Arab", "Deva", "Hans", "Hant"]

class ASRModel(ABC):
    def __init__(self, model_name, config):
        """Load model here"""
        self.model_name = model_name
        self.config = config
    
    @classmethod
    def from_pretrained(self, model_name, config={}):
        if model_name.startswith("openai/whisper"):
            return Whisper(model_name, config)
        elif model_name.startswith("facebook/mms"):
            return MMS(model_name, config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @abstractmethod
    def load_data(self, dataset_path: str, caller_script: str = None, data_seed: int = 42, dataset_split: str = "train"):
        pass
    
    def load_data_jax(self, dataset_path: str, caller_script: str = None, data_seed: int = 42, dataset_split: str = "train"):
        A, y = self.load_data(dataset_path, caller_script, data_seed, dataset_split)
        A = jnp.array(A)  # (n, 768)
        y = jnp.array(y)    # (n,)
        return A, y

    @abstractmethod
    def set_lang_detect_head(self, lang_detect_head):
        pass

    @abstractmethod
    def predict(self, audio):
        """Runs transcription on the audio, returns list of language tokens and transcriptions"""
        pass

    @abstractmethod
    def get_dimensions(self):
        pass

    @abstractmethod
    def get_device(self):
        pass
    


def whisper_custom_retrieve_init_tokens_creator(asr_model, languages, tokenizer):
    def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
        def lang_to_id(_, lang):
            return self.generation_config.lang_to_id[f"<|{lang}|>"]

        encoder_outputs = self.model.encoder(input_features, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        class_ids = asr_model.head.predict(hidden)
        
        if not languages:
            raise ValueError("config['languages'] must be provided (non-empty) when using a custom language detection head.")

        # Head predicts indices into config['languages']
        chosen_langs = []
        for cid in class_ids:
            try:
                chosen_langs.append(languages[int(cid)])
            except Exception:
                chosen_langs.append(languages[0])

        lang_tokens = [lang_to_id(self, lang) for lang in chosen_langs]
        asr_model.lang_tokens.extend(chosen_langs)
        
        sot_token_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        transcribe_token_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        init_tokens = [[sot_token_id, lang_token, transcribe_token_id] for lang_token in lang_tokens]
        
        init_tokens_tensor = torch.tensor(init_tokens, 
                                          dtype=torch.long, 
                                          device=input_features.device)

        return init_tokens_tensor

    return _custom_retrieve_init_tokens


class Whisper(ASRModel):
    def __init__(self, model_name, config={}):
        super().__init__(model_name, config)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.model.to(dtype=dtype)
        self.model.config.forced_decoder_ids = None
        self.processor = WhisperProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.head = None # default head


    def load_data(self, dataset_path: str, caller_script: str = None, data_seed: int = 42, dataset_split: str = "train", shuffle=True):
        """
        Load HF dataset, extract pooled model hidden states, return train/test splits.
        
        Args:
            dataset_path (str): Path to local HF dataset dir (splits: train, valid, test).
            caller_script (str): 'defrun' for 90% data (convex training); else full.
            data_seed (int): Seed for shuffle/split.
        
        Returns:
            Atr, ytr, Atst, ytst, ntr, ntst: JAX arrays for features/labels (pooled to 768 dim).
        """
        np.random.seed(data_seed)
        
        # Load train split (main data for training)
        dataset = load_from_disk(dataset_path)
        train_data = dataset[dataset_split]
        print(f"Loaded {len(train_data)} train samples")

        languages = self.config.get("languages")
        if not languages:
            # Infer languages from the dataset split and persist for downstream consumers.
            languages = sorted({sample.get("lang") for sample in train_data if sample.get("lang") is not None})
            self.config["languages"] = languages

        lang_to_index = {lang: i for i, lang in enumerate(languages)}
        
        # Load Whisper encoder
        self.model.eval()
        
        def extract_pooled_hidden(audio) -> np.ndarray:
            """Extract and pool last hidden states to (768,)."""
            # Handle audio dict or path
            if isinstance(audio, dict):
                if audio.get('array') is not None:
                    audio_arr = audio['array']
                    sr = audio['sampling_rate']
                else:
                    audio_path = audio['path']
                    if not os.path.exists(audio_path):
                        return None
                    waveform, sr = torchaudio.load(audio_path)
                    audio_arr = waveform.mean(0).numpy()
            else:
                # Assume path if not dict
                if not os.path.exists(audio):
                    return None
                waveform, sr = torchaudio.load(audio)
                audio_arr = waveform.mean(0).numpy()
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_arr = resampler(torch.tensor(audio_arr)).numpy()
            
            # Process to input_features
            inputs = self.processor(audio_arr, sampling_rate=16000, return_tensors='pt').to(self.get_device(), dtype=dtype)
            
            # Encoder last hidden
            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(inputs.input_features, output_hidden_states=True)
                hidden = encoder_outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
            
            # Pool: Mean over seq_len
            pooled = hidden.mean(0).cpu().numpy()  # (768,)
            return pooled
        
        # Extract features and labels for all train samples
        features = []
        labels = []
        valid_count = 0
        for sample in train_data:
            hidden = extract_pooled_hidden(sample['audio'])
            if hidden is None:
                continue  # Skip invalid audio
            
            label = lang_to_index.get(sample.get("lang"))
            if label is None:
                continue
            features.append(hidden)
            labels.append(label)
            valid_count += 1
        
        if valid_count == 0:
            raise ValueError("No valid audio samples found")
        print(f"Extracted {valid_count} valid samples across {len(languages)} language(s)")
        
        # Convert to arrays
        A = np.array(features)
        y = np.array(labels, dtype=int)
        
        # Shuffle
        if shuffle:
            perm = np.random.permutation(A.shape[0])
            A = A[perm]
            y = y[perm]

        return A, y

    def set_lang_detect_head(self, lang_detect_head):
        self.head = lang_detect_head
        if self.head:
            languages = self.config.get("languages")
            if not languages:
                raise ValueError("config['languages'] must be provided when using a custom language detection head.")
            self.model._retrieve_init_tokens = types.MethodType(
                whisper_custom_retrieve_init_tokens_creator(self, languages, self.processor.tokenizer),
                self.model,
            )
        
    def _detect_language_vanilla(self, input_features):
        # 50258 is the token for transcribing
        batch_size = input_features.shape[0]
        device = input_features.device
        decoder_input_ids = torch.full((batch_size, 1), 50258, dtype=torch.long, device=device)
        model_output = self.model(input_features, decoder_input_ids=decoder_input_ids)
        logits = model_output.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Language tokens in Whisper multilingual models are IDs 50263 to 50361 (99 languages)
        # Compute probabilities and detect the most likely language per batch item
        language_probs = torch.softmax(logits, dim=-1)
        language_indices = torch.argmax(language_probs, dim=-1)  # Shape: (batch_size,)
        
        # Map indices to language codes (sorted list of Whisper's 99 supported languages)
        detected_languages = [self.id_to_lang(x.item()) for x in language_indices]
        
        # Return list of detected languages (one per batch item); also return probs if needed
        return detected_languages  # e.g., ['en'] for batch_size=1
    
    def predict(self, audio):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.get_device(), dtype=dtype)

        self.lang_tokens = []
        predicted_ids = self.model.generate(input_features, repetition_penalty=1.1, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)        
        if(self.head is None or getattr(self.head, "SKIP", False)):
            self.lang_tokens = self._detect_language_vanilla(input_features)
        return self.lang_tokens, transcription
    
    def get_dimensions(self):
        return self.model.config.d_model

    def get_device(self):
        return next(self.model.model.encoder.layers[-1].parameters()).device

    def lang_to_id(self, lang):
        lang_code = f"<|{lang}|>"
        return self.model.generation_config.lang_to_id[lang_code]

    def id_to_lang(self, tid):
        id_to_lang_mapping =  dict(zip(self.model.generation_config.lang_to_id.values(), self.model.generation_config.lang_to_id.keys()))
        return id_to_lang_mapping.get(tid, "    ")[2:-2]


class MMS(ASRModel):
    def __init__(self, model_name: str, config: dict = {}):
        super().__init__(model_name, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device, dtype=dtype)
        self.lid_model = AutoModelForAudioClassification.from_pretrained("facebook/mms-lid-126").to(self.device, dtype=dtype)
        self.head = None
        self.current_adapter = None

        self.iso2_to_iso3 = ISO2_TO_ISO3
        self.languages = config.get("languages") or []
        # Back-compat: older config used "class_names" for iso2 codes.
        if not self.languages and config.get("class_names"):
            self.languages = list(config.get("class_names"))
        self.class_names = [self.iso2_to_iso3.get(cid, cid) for cid in self.languages]

    def load_data(self, dataset_path: str, caller_script: str = None, data_seed: int = 42, dataset_split: str = "train", shuffle=True):
        """
        Load HF dataset, extract pooled model hidden states, return train/test splits.
        
        Args:
            dataset_path (str): Path to local HF dataset dir (splits: train, valid, test).
            caller_script (str): 'defrun' for 90% data (convex training); else full.
            data_seed (int): Seed for shuffle/split.
        
        Returns:
            Atr, ytr, Atst, ytst, ntr, ntst: JAX arrays for features/labels (pooled to 768 dim).
        """
        np.random.seed(data_seed)
        
        # Load train split (main data for training)
        dataset = load_from_disk(dataset_path)
        train_data = dataset[dataset_split]
        print(f"Loaded {len(train_data)} train samples")

        languages = self.config.get("languages") or self.languages
        if not languages:
            languages = sorted({sample.get("lang") for sample in train_data if sample.get("lang") is not None})
            self.config["languages"] = languages
            self.languages = languages
            self.class_names = [self.iso2_to_iso3.get(cid, cid) for cid in self.languages]
        lang_to_index = {lang: i for i, lang in enumerate(languages)}
        
        
        def extract_pooled_hidden(audio) -> np.ndarray:
            """Extract and pool last hidden states to (768,)."""
            # Handle audio dict or path
            if isinstance(audio, dict):
                if audio.get('array') is not None:
                    audio_arr = audio['array']
                    sr = audio['sampling_rate']
                else:
                    audio_path = audio['path']
                    if not os.path.exists(audio_path):
                        return None
                    waveform, sr = torchaudio.load(audio_path)
                    audio_arr = waveform.mean(0).numpy()
            else:
                # Assume path if not dict
                if not os.path.exists(audio):
                    return None
                waveform, sr = torchaudio.load(audio)
                audio_arr = waveform.mean(0).numpy()
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_arr = resampler(torch.tensor(audio_arr)).numpy()
            
            # Process to input_features
            inputs = self.processor(audio_arr, sampling_rate=16000, return_tensors='pt').to(self.get_device(), dtype=dtype)
            
            # Encoder last hidden
            with torch.no_grad():
                encoder_outputs = self.model.wav2vec2(inputs.input_values, output_hidden_states=True)
                hidden = encoder_outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)

            # Pool: Mean over seq_len
            pooled = hidden.mean(0).cpu().numpy()  # (768,)
            return pooled
        
        # Extract features and labels for all train samples
        features = []
        labels = []
        valid_count = 0
        for sample in train_data:
            hidden = extract_pooled_hidden(sample['audio'])
            if hidden is None:
                continue  # Skip invalid audio
            
            label = lang_to_index.get(sample.get("lang"))
            if label is None:
                continue
            features.append(hidden)
            labels.append(label)
            valid_count += 1
        
        if valid_count == 0:
            raise ValueError("No valid audio samples found")
        print(f"Extracted {valid_count} valid samples across {len(languages)} language(s)")
        
        # Convert to arrays
        A = np.array(features)
        y = np.array(labels, dtype=int)
        
        # Shuffle
        if shuffle:
            perm = np.random.permutation(A.shape[0])
            A = A[perm]
            y = y[perm]

        return A, y, len(languages)

    def set_lang_detect_head(self, lang_detect_head):
        self.head = lang_detect_head

    def _detect_language_vanilla(self, audio_list):
        inputs = self.processor(audio_list, sampling_rate=16000, padding="longest", return_tensors="pt")
        input_values = inputs.input_values.to(self.device, dtype=dtype)
        with torch.no_grad():
            logits = self.lid_model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        return [self.lid_model.config.id2label[pid] for pid in pred_ids]

    def predict(self, audio):
        # Ensure audio is a list (single np.ndarray or list of them)
        if not isinstance(audio, list):
            audio = [audio]

        batch_size = len(audio)

        # Prepare batch once
        inputs = self.processor(audio, sampling_rate=16000, padding="longest", return_tensors="pt")
        input_values = inputs.input_values.to(self.device, dtype=dtype)

        # 1. Detect language(s)
        if self.head:
            # Run frozen encoder to get hidden states for the head.
            # NOTE: this batch is zero-padded to the longest clip ("padding=longest").
            # Training (load_data) pools each clip individually over real frames only,
            # so we must mask out padded frames here too -- otherwise the padding
            # dilutes the mean pool and the head sees out-of-distribution features
            # (this is why benchmark acc was ~0.85 while the head trains to ~0.98).
            attention_mask = inputs.get("attention_mask")
            with torch.no_grad():
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    encoder_out = self.model.wav2vec2(
                        input_values, attention_mask=attention_mask, output_hidden_states=True
                    )
                    hidden = encoder_out.last_hidden_state  # (B, T, D)
                    # Map the waveform-level mask onto the encoder's downsampled frame grid.
                    feat_lengths = self.model._get_feat_extract_output_lengths(
                        attention_mask.sum(-1)
                    ).long()
                    frame_idx = torch.arange(hidden.shape[1], device=self.device)
                    frame_mask = (frame_idx[None, :] < feat_lengths[:, None]).unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1.0)
                else:
                    encoder_out = self.model.wav2vec2(input_values, output_hidden_states=True)
                    pooled = encoder_out.last_hidden_state.mean(dim=1)  # (B, D)
                pooled = pooled.float().cpu().numpy()  # → numpy for sklearn/jax heads
                class_ids = self.head.predict(pooled)  # assume returns np.array of shape (B,)
            detected_langs = [self.class_names[cid] for cid in class_ids]
        else:
            detected_langs = self._detect_language_vanilla(audio)


        # 2. Transcribe – group by language to minimise adapter switching
        transcriptions = [None] * batch_size
        lang_to_indices = defaultdict(list)
        for i, lang in enumerate(detected_langs):
            lang_to_indices[lang].append(i)

        for lang, indices in lang_to_indices.items():
            batch_input = input_values[indices]
            if self.current_adapter != lang:
                self.set_adapter(lang)

            with torch.no_grad():
                logits = self.model(batch_input).logits

            pred_ids = torch.argmax(logits, dim=-1)
            trans = self.processor.batch_decode(pred_ids, skip_special_tokens=True)

            for k, orig_idx in enumerate(indices):
                transcriptions[orig_idx] = trans[k]

        detected_langs = [ISO3_TO_ISO2[token] if token in ISO3_TO_ISO2 else token for token in detected_langs]

        # Return single values if input was single audio, otherwise lists
        if batch_size == 1:
            return detected_langs[0], transcriptions[0]
        

        return detected_langs, transcriptions

    def get_dimensions(self):
        return self.model.config.hidden_size

    def get_device(self):
        return self.device
    
    def set_adapter(self, lang_id):
        for script in POSSIBLE_SCRIPTS:
            if script == "":
                new_lang_id = lang_id
            else:
                new_lang_id = lang_id+"-script_"+script

            try:
                self.processor.tokenizer.set_target_lang(new_lang_id)
                self.model.load_adapter(new_lang_id)
                self.current_adapter = lang_id
                return
            except ValueError:
                pass
        
        # raise ValueError(f"No adapter found for {lang_id}")
        new_lang_id = "eng"
        self.processor.tokenizer.set_target_lang(new_lang_id)
        self.model.load_adapter(new_lang_id)
        self.current_adapter = lang_id

