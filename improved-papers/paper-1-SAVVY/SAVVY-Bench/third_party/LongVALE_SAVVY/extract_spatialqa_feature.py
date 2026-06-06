"""
We adapted the extract_spatialqa_feature file to get input videos and audios from SAVVY-Bench data.
"""
import os
import torchaudio.compliance.kaldi as ta_kaldi
import torch.nn.functional as F

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip
import pickle
from tqdm import tqdm
import tempfile
import pickle

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import clip
from transformers import WhisperFeatureExtractor

import sys; sys.path = ["LongVALE/"] + sys.path
try:
    from longvalellm.constants import IMAGE_TOKEN_INDEX
    from longvalellm.conversation import conv_templates, SeparatorStyle
    from longvalellm.model.builder import load_pretrained_model, load_lora
    from longvalellm.utils import disable_torch_init
    from longvalellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
    from longvalellm.mm_utils import BEATSAudioExtractor
    from longvalellm.model.beats.BEATs import BEATs, BEATsConfig
    from longvalellm.mm_utils import VideoExtractor 
    from preprocess.beats_feature_extract import prepare_model as prepare_audio_model
    from preprocess.clip_feature_extract import prepare_model as prepare_visual_model
    from preprocess.whisper_feature_extract import prepare_model as prepare_speech_model
except ImportError:
    eval_logger.debug("LongVALE is not installed. Please install LongVALE to use this model.")


cache_path = "LongVALE/checkpoints"
visual_processor = prepare_visual_model("LongVALE/checkpoints/ViT-L-14.pt", 0)[0]
visual_processor.eval()
audio_extractor = BEATSAudioExtractor(is_eval=True)
audio_processor = prepare_audio_model(f"{cache_path}/BEATs_iter3_plus_AS20K.pt", 0)[0]
audio_processor.eval()
speech_processor = prepare_speech_model(snapshot_download("openai/whisper-large-v2"), 0)[0]
speech_processor.eval()
speech_transform = WhisperFeatureExtractor.from_pretrained(snapshot_download("openai/whisper-large-v2"))
        

def get_video_chunk_content(video_path, max_audio_seconds=180, device="cuda:0"):
    video = VideoFileClip(video_path)
    fps = video.fps
    sr = 16000
    audio_path = os.path.splitext(video_path)[0] + ".pkl"
    if os.path.exists(audio_path):
        audio_np = pickle.load(open(audio_path, "rb"))
    else:
        if video.audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                temp_audio_file_path = temp_audio_file.name
                video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=sr, verbose=False, logger=None)
                audio_np, sr = librosa.load(temp_audio_file_path, sr=sr, mono=True)
        else:
            audio_np = np.zeros((round(sr * video.duration)))
        pickle.dump(audio_np, open(audio_path, 'wb'))
    
    # audio
    waveform = torch.tensor(audio_np).float().unsqueeze(0)
    waveform = waveform * 2**15
    fbank = ta_kaldi.fbank(
            waveform,
            num_mel_bins=128,
            sample_frequency=sr,
            frame_length=25,
            frame_shift=10, # 10
        )
    fbank = (fbank - 15.41663) / (2 * 6.55582)
    frame_length = 512
    fbank_pad_len = fbank.shape[0] % frame_length
    if fbank_pad_len > 0:
        fbank = torch.nn.ZeroPad2d((0, 0, 0, fbank_pad_len))(fbank)
    curr_frames = fbank.shape[0] // frame_length
    frames = [fbank[i*frame_length:(i+1)*frame_length].unsqueeze(0) for i in range(curr_frames)]
    audio_features = torch.cat(frames, dim=0)
    audio_features = audio_processor.extract_features(audio_features.to(device))[0]
    audio_features = audio_features.mean(dim=1).squeeze(1).data.cpu()
    
    # speech 
    waveform = torch.tensor(audio_np).float()
    if len(waveform) > 30 * sr:
        audio_list = [waveform[i: i + 30 * sr] for i in range(0, len(waveform), 30 * sr)]
        spectrogram_list = []
        for audio_piece in audio_list:
            spectrogram_piece = speech_transform(
                audio_piece,
                sampling_rate=sr,
                return_tensors="pt",
                max_length=30 * sr,
            )
            spectrogram_list.append(spectrogram_piece["input_features"].squeeze())
        spectrogram = torch.stack(spectrogram_list, dim=0)
    else:
        spectrogram = speech_transform(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            max_length=30 * sr,
            )
        spectrogram = spectrogram["input_features"].squeeze()

    features = []
    for spec in spectrogram:
        features.append(speech_processor(spec.unsqueeze(0).to(device), return_dict=True).last_hidden_state.data.cpu()[0])
    features = torch.stack(features)
    dim = features.shape[-1]
    features = features.reshape(1, -1, dim)

    B, T, C = features.shape
    kernel = round(1500 * 5.12 / 30.0)
    stride = round(1500 * 5.12 / 30.0)
    kernel = (1, kernel)
    stride = (1, stride)
    speech_embeds_tr = features.transpose(1, 2).unsqueeze(2)
    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, stride=stride)
    _, _, L = speech_embeds_overlap.shape
    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L) 
    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1]) 
    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C) 
    
    speech_features = torch.mean(speech_embeds, dim=1) 

    # visual
    video_loader = VideoExtractor(N=32)
    _, images = video_loader.extract({'id': None, 'video': video_path})
    vis_transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    images = vis_transform(images / 255.0)
    images = images.to(torch.float16)
    with torch.no_grad():
        vis_features = visual_processor.encode_image(images.to(device)).data.cpu()
    
    return vis_features, audio_features, speech_features


lines = open("third_party/LongVale_SAVVY/video.txt", "r").readlines()
feature_dir = "data/spatial_avqa/av_feat_pkl/longvale_feat/"
os.makedirs(feature_dir, exist_ok=True)
for line in tqdm(lines):
    video_name = line.strip().split("/")[1]
    feat_path = f"{feature_dir}/{video_name}.pkl"
    if os.path.exists(feat_path):
        continue
    video_path = f"data/spatial_avqa/videos/{video_name}.mp4"
    vis_features, audio_features, speech_features = get_video_chunk_content(video_path)
    feat_dict = {
        "vis": vis_features.numpy(),
        "audio": audio_features.numpy(),
        "speech": speech_features.numpy()
    }
    pickle.dump(feat_dict, open(feat_path, "wb"))