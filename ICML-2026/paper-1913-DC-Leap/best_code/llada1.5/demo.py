import time
import queue
import threading
import torch
import numpy as np
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM 

DEVICE = 'cuda' 
MASK_ID = 126336

MODEL = None
TOKENIZER = None

def load_model():
    global MODEL, TOKENIZER
    if MODEL is None:
        MODEL = LLaDAModelLM.from_pretrained(
            '/path/to/your/LLaDA-1.5', 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(DEVICE).eval()
        TOKENIZER = AutoTokenizer.from_pretrained(
            '/path/to/your/LLaDA-1.5', 
            trust_remote_code=True
        )

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

def get_top1_info(logits: torch.Tensor):
    probs = F.softmax(logits, dim=-1)
    top1_probs, top1_indices = torch.max(probs, dim=-1)
    return top1_indices, top1_probs

def verify_and_commit(region_logits, region_x, mask_id, commit_thres, left_boundary_known):
    top1_indices, top1_probs = get_top1_info(region_logits)
    is_confident = (top1_probs > commit_thres)
    if left_boundary_known:
        contiguity_mask = torch.cumprod(is_confident.int(), dim=0).bool()
    else:
        contiguity_mask = torch.zeros_like(is_confident, dtype=torch.bool)
    is_mask = (region_x == mask_id)
    final_commit_mask = contiguity_mask & is_mask
    return top1_indices, final_commit_mask


def render_html_tokens(tensor_seq, prompt_len, draft_tensor=None):
    gen_seq = tensor_seq[0, prompt_len:].tolist()
    
    draft_seq = None
    if draft_tensor is not None:
        draft_seq = draft_tensor.tolist()

    html_parts = ["<div class='token-container'>"]
    
    for i, t in enumerate(gen_seq):
        if t == MASK_ID:
            if draft_seq is not None and draft_seq[i] != MASK_ID:
                token_str = TOKENIZER.decode([draft_seq[i]])
                token_str = token_str.replace(' ', ' ').replace('Ġ', '')
                if not token_str.strip(): token_str = "&nbsp;"
                html_parts.append(f"<span class='token draft-token'>{token_str}</span>")
            else:
                html_parts.append("<span class='token mask-token'></span>")
        else:
            token_str = TOKENIZER.decode([t])
            token_str = token_str.replace(' ', ' ').replace('Ġ', '')
            if not token_str.strip(): token_str = "&nbsp;"
            html_parts.append(f"<span class='token decoded-token'>{token_str}</span>")
            
    html_parts.append("</div>")
    return "".join(html_parts)


@torch.no_grad()
def generate_baseline_stream(prompt_tensor, steps=256, gen_length=256, block_length=32, temperature=0., cfg_scale=0.):
    x = torch.full((1, prompt_tensor.shape[1] + gen_length), MASK_ID, dtype=torch.long).to(DEVICE)
    x[:, :prompt_tensor.shape[1]] = prompt_tensor.clone()
    prompt_len = prompt_tensor.shape[1]
    prompt_index = (x != MASK_ID)

    yield render_html_tokens(x, prompt_len)
    
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt_len + num_block * block_length: prompt_len + (num_block + 1) * block_length:] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == MASK_ID)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = MODEL(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = MODEL(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            yield render_html_tokens(x, prompt_len)

    yield render_html_tokens(x, prompt_len)

@torch.no_grad()
def generate_dcleap_stream(prompt_tensor, gen_length=256, block_length=32, max_window_size=128, commit_thres=0.65, draft_thres=0.95, temperature=0., cfg_scale=0.):
    max_window_size = gen_length//2
    x = torch.full((1, prompt_tensor.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=DEVICE)
    x[:, :prompt_tensor.shape[1]] = prompt_tensor.clone()
    prompt_len = prompt_tensor.shape[1]
    draft_bank = torch.full((gen_length,), MASK_ID, dtype=torch.long, device=DEVICE)

    yield render_html_tokens(x, prompt_len, draft_tensor=draft_bank)

    verified_end = 0 
    while verified_end < gen_length:
        l2r_len = max_window_size
        future_len = max_window_size

        win_s = verified_end
        win_e = min(verified_end + l2r_len + future_len, gen_length)
        abs_win_s = prompt_len + win_s
        abs_win_e = prompt_len + win_e

        if abs_win_s >= abs_win_e: break

        x_for_prediction = x.clone()
        drafts = draft_bank[win_s:win_e]
        target_slice = x_for_prediction[0, abs_win_s:abs_win_e]
        mask_locs = (target_slice == MASK_ID)
        valid_drafts = (drafts != MASK_ID)
        fill_locs = mask_locs & valid_drafts
        
        if fill_locs.any():
            x_for_prediction[0, abs_win_s:abs_win_e][fill_locs] = drafts[fill_locs]

        leader_end = min(abs_win_s + max_window_size, abs_win_e)
        x_for_prediction[0, abs_win_s:leader_end] = MASK_ID
        
        if cfg_scale > 0.:
            pred_model_out = MODEL(x_for_prediction, output_hidden_states=False)
            prediction_logits = pred_model_out.logits
            pred_conditional, pred_unconditional = prediction_logits.chunk(2, dim=0)
            prediction_logits = pred_unconditional + (cfg_scale + 1) * (pred_conditional - pred_unconditional)
        else:
            prediction_logits = MODEL(x_for_prediction, output_hidden_states=False).logits

        l2r_abs_end = min(prompt_len + verified_end + l2r_len, abs_win_e)
        
        if l2r_abs_end > abs_win_s:
            region_logits = prediction_logits[0, abs_win_s:l2r_abs_end]
            region_x = x[0, abs_win_s:l2r_abs_end]

            left_boundary_known = True
            if verified_end > 0:
                left_boundary_known = (x[0, abs_win_s - 1].item() != MASK_ID)
            
            top1_tokens, commit_mask = verify_and_commit(region_logits, region_x, MASK_ID, commit_thres, left_boundary_known)

            if not commit_mask.any():
                is_mask = (region_x == MASK_ID)
                if is_mask.any():
                    first_idx = is_mask.nonzero(as_tuple=True)[0][0]
                    commit_mask[first_idx] = True

            if commit_mask.any():
                update_idx = commit_mask.nonzero(as_tuple=True)[0]
                x[0, abs_win_s + update_idx] = top1_tokens[update_idx]
            
                next_mask = (x[0, abs_win_s:l2r_abs_end] == MASK_ID).nonzero(as_tuple=True)
                if next_mask[0].numel() > 0:
                    verified_end += next_mask[0][0].item()
                else:
                    verified_end += (l2r_abs_end - abs_win_s)

        draft_logits = prediction_logits[0, abs_win_s:abs_win_e]
        r_idx, r_probs = get_top1_info(draft_logits)

        draft_candidates_mask = r_probs > draft_thres
        if draft_candidates_mask.any():
            indices = draft_candidates_mask.nonzero(as_tuple=True)[0]
            bank_indices = win_s + indices
            valid = bank_indices < gen_length
            if valid.all():
                draft_bank[bank_indices] = r_idx[indices]
            elif valid.any():
                draft_bank[bank_indices[valid]] = r_idx[indices][valid]

        yield render_html_tokens(x, prompt_len, draft_tensor=draft_bank)

    yield render_html_tokens(x, prompt_len)


def thread_runner(func, q, key, *args, **kwargs):
    start_time = time.time()
    try:
        for html_chunk in func(*args, **kwargs):
            elapsed = time.time() - start_time
            q.put((key, html_chunk, elapsed))
    except Exception as e:
        q.put((key, f"<div style='color:red;'>Error: {str(e)}</div>", 0))
    finally:
        q.put((key, "DONE", time.time() - start_time))

def race_models(prompt_text, gen_len, c_thres, d_thres):
    load_model()
    
    m = [{"role": "user", "content": prompt_text}]
    prompt_str = TOKENIZER.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = TOKENIZER(prompt_str, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)

    q = queue.Queue()

    t1 = threading.Thread(target=thread_runner, args=(generate_baseline_stream, q, 'base', input_ids), kwargs={'steps': gen_len, 'gen_length': gen_len} )
    t2 = threading.Thread(target=thread_runner, args=(generate_dcleap_stream, q, 'dcleap', input_ids), kwargs={'gen_length': gen_len, 'commit_thres': c_thres, 'draft_thres': d_thres})
    
    t1.start()
    t2.start()
    
    base_html, dcleap_html = "", ""
    base_time, dcleap_time = 0.0, 0.0
    base_done, dcleap_done = False, False
    
    while not (base_done and dcleap_done):
        try:
            key, text, elapsed = q.get(timeout=0.1)
            if text == "DONE":
                if key == 'base': 
                    base_done = True
                    base_time = elapsed
                else: 
                    dcleap_done = True
                    dcleap_time = elapsed
            else:
                if key == 'base':
                    base_html = text
                    base_time = elapsed
                else:
                    dcleap_html = text
                    dcleap_time = elapsed
                    
            b_tps = gen_len / base_time if base_time > 0 else 0
            d_tps = gen_len / dcleap_time if dcleap_time > 0 else 0
            
            yield (
                base_html, 
                f"<div class='metric-badge base-badge'>⏱️ <b>{base_time:.2f}s</b> | 🚀 <b>{b_tps:.1f} TPS</b></div>", 
                dcleap_html, 
                f"<div class='metric-badge dcleap-badge'>⏱️ <b>{dcleap_time:.2f}s</b> | 🚀 <b>{d_tps:.1f} TPS</b></div>",
                "<div class='loading-status'>⏳ Analyzing Generation Trajectories...</div>"
            )
        except queue.Empty:
            continue

    speedup = (base_time / dcleap_time) if dcleap_time > 0 else 1.0
    latency_reduction = ((base_time - dcleap_time) / base_time * 100) if base_time > 0 else 0.0
    b_tps = gen_len / base_time if base_time > 0 else 0.0
    d_tps = gen_len / dcleap_time if dcleap_time > 0 else 0.0
    
    stats_html = f"""
    <div class='results-dashboard fade-in'>
        <div class='dashboard-header'>
            <h2>✨ Generation Complete ✨</h2>
            <p>DC-Leap achieved a <span class='highlight-speedup'>{speedup:.2f}x</span> speedup over the Baseline.</p>
        </div>
        <div class='dashboard-metrics'>
            <div class='metric-card dcleap-card fade-in-up' style='animation-delay: 0.1s;'>
                <span class='metric-title'>DC-Leap</span>
                <div class='metric-value'>{dcleap_time:.2f}s</div>
                <span class='metric-sub'>{d_tps:.1f} Tokens/sec</span>
                <div class='speed-badge'>⚡ FASTER</div>
            </div>
            <div class='metric-card fade-in-up' style='animation-delay: 0.2s;'>
                <span class='metric-title'>Baseline</span>
                <div class='metric-value'>{base_time:.2f}s</div>
                <span class='metric-sub'>{b_tps:.1f} Tokens/sec</span>
            </div>
            <div class='metric-card reduction-card fade-in-up' style='animation-delay: 0.3s;'>
                <span class='metric-title'>Latency Reduction</span>
                <div class='metric-value highlight-reduction'>🔻 {latency_reduction:.1f}%</div>
                <span class='metric-sub'>Less Time Waiting</span>
            </div>
        </div>
    </div>
    """
    
    yield (
        base_html, 
        f"<div class='metric-badge base-badge'>⏱️ <b>{base_time:.2f}s</b> | 🚀 <b>{b_tps:.1f} TPS</b></div>", 
        dcleap_html, 
        f"<div class='metric-badge dcleap-badge'>⏱️ <b>{dcleap_time:.2f}s</b> | 🚀 <b>{d_tps:.1f} TPS</b></div>",
        stats_html
    )

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

/* Global Gradient Background & Typography */
body, .gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    color: #0f172a;
    overflow-x: hidden;
}

/* Floating Abstract Bubbles for Premium Feel */
.ambient-bubbles {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.ambient-bubbles .bubble {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.6;
    animation: float-bubble 20s infinite alternate ease-in-out;
}
.ambient-bubbles .b1 {
    width: 600px; height: 600px;
    background: #dbeafe; /* blue-100 */
    top: -100px; left: -100px;
    animation-duration: 25s;
}
.ambient-bubbles .b2 {
    width: 500px; height: 500px;
    background: #fce7f3; /* pink-100 */
    bottom: -150px; right: -50px;
    animation-duration: 28s; animation-delay: -5s;
}
.ambient-bubbles .b3 {
    width: 400px; height: 400px;
    background: #dcfce7; /* green-100 */
    top: 40%; left: 30%;
    animation-duration: 32s; animation-delay: -10s;
}

@keyframes float-bubble {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(50px, -50px) scale(1.1); }
}

/* Glassmorphism Main Layout */
.glass-container {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.8) !important;
    border-radius: 28px !important;
    box-shadow: 0 40px 80px -20px rgba(0, 0, 0, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.6) !important;
    padding: 48px !important;
    margin-top: 40px !important;
    margin-bottom: 40px !important;
    position: relative;
    z-index: 10;
}

/* Elegant Header Typography */
.header-box { 
    text-align: center; 
    margin-bottom: 40px; 
}
.header-box h1 { 
    font-size: 3.2rem; 
    font-weight: 900; 
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #0f172a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.header-box h3 {
    color: #64748b;
    font-weight: 500;
    font-size: 1.4rem;
    letter-spacing: 0px;
    margin-top: 0;
}
.header-box .notice-text {
    background: rgba(255,255,255,0.7);
    padding: 16px 24px;
    border-radius: 12px;
    color: #475569;
    font-size: 0.95rem;
    max-width: 800px;
    margin: 20px auto 0;
    line-height: 1.6;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 6px rgba(0,0,0,0.02);
}

/* Call to Action Button */
.cta-button {
    background: linear-gradient(135deg, #0f172a 0%, #334155 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    letter-spacing: 0.5px;
    padding: 16px 36px !important;
    box-shadow: 0 15px 30px -10px rgba(15, 23, 42, 0.4) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
}
.cta-button:hover {
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 0 20px 40px -10px rgba(15, 23, 42, 0.5) !important;
    background: linear-gradient(135deg, #1e293b 0%, #475569 100%) !important;
}

/* Token Container UI (Clean & Elegant) */
.token-wrapper h2 {
    font-size: 1.3rem; 
    font-weight: 700; 
    color: #1e293b;
    margin-bottom: 5px;
}
.token-wrapper p {
    color: #64748b; font-size: 0.95rem; font-style: normal; margin-bottom: 16px;
}

.token-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    background: rgba(255, 255, 255, 0.7);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 1);
    min-height: 320px;
    align-content: flex-start;
    font-family: 'JetBrains Mono', Consolas, monospace;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.02), 0 10px 20px -5px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
}

/* Tokens */
.token {
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.95em;
    font-weight: 500;
    text-align: center;
    position: relative;
    overflow: hidden;
    /* Gentle transition instead of popping/flashing animations to improve performance */
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}

/* Minimalist Token Colors */
/* [MASK] */
.mask-token {
    background: rgba(241, 245, 249, 0.6); color: #94a3b8; border: 1px dashed #cbd5e1; min-width: 32px;
}
/* Draft (Yellow/Gold) */
.draft-token {
    background: #fef9c3; color: #a16207; border: 1px solid #fde047; font-weight: 600;
}
/* Decoded (Green) */
.decoded-token {
    background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; font-weight: 600;
}

/* Badges */
.metric-badge {
    background: rgba(255, 255, 255, 0.9);
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.95rem;
    font-weight: 500;
    float: right;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.base-badge { color: #475569; }
.dcleap-badge { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }

/* Dashboard Cards (End Result) */
.results-dashboard {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 40px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.6);
    text-align: center;
    margin-top: 40px;
}
.highlight-speedup {
    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900; 
    font-size: 1.8em; 
    padding: 0 5px;
}
.dashboard-metrics {
    display: flex; justify-content: center; gap: 30px; margin-top: 30px; flex-wrap: wrap;
}
.metric-card {
    flex: 1; min-width: 250px;
    background: #ffffff;
    padding: 30px; border-radius: 20px; border: 1px solid #f1f5f9;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
    position: relative;
    overflow: hidden;
}
.dcleap-card {
    background: #f0fdf4; border-color: #bbf7d0; transform: scale(1.05); z-index: 2;
    box-shadow: 0 20px 40px -10px rgba(22, 163, 74, 0.15);
}
.reduction-card {
    background: #fdf4ff; border-color: #f5d0fe;
}
.highlight-reduction {
    color: #c026d3 !important; /* Fuchsia for emphasis on reduction */
}
.metric-title { font-size: 0.9em; color: #64748b; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;}
.metric-value { font-size: 2.5em; font-weight: 900; color: #0f172a; margin: 10px 0; letter-spacing: -1px;}
.metric-sub { font-size: 1em; color: #475569; font-weight: 500;}

.speed-badge {
    position: absolute; top: 12px; right: 12px;
    background: #22c55e; color: white;
    font-size: 0.75rem; font-weight: 800; padding: 4px 10px; border-radius: 100px;
    letter-spacing: 1px;
}

.loading-status { text-align: center; color: #64748b; font-weight: 500; padding: 20px; letter-spacing: 0.5px;}

/* Animations */
.fade-in { animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.fade-in-up { opacity: 0; animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Input boxes refined */
.gradio-container textarea, .gradio-container input {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.3s ease !important;
}
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"), css=custom_css) as demo:

    gr.HTML("""
    <div class="ambient-bubbles">
        <div class="bubble b1"></div>
        <div class="bubble b2"></div>
        <div class="bubble b3"></div>
    </div>
    """)

    with gr.Column(elem_classes="glass-container fade-in"):
        gr.HTML("""
        <div class="header-box">
            <h1>⚡DC-Leap: Training-Free Acceleration of dLLMs⚡</h1>
            <h3>via Draft-Guided Contiguous Leaping Decoding</h3>
            <div class="notice-text">
                <b>Notice:</b> DC-Leap is a simple yet effective method that introduces Dynamic Contiguous Verification and leverages the inherent bidirectional attention of dLLMs via Draft-guided Decoding to unlock the acceleration potential within lower-confidence regimes.
            </div>
        </div>
        """)

        with gr.Row():
            prompt_input = gr.Textbox(
                label="📝 User Prompt", 
                value="Albert has 2 apples. He buys 3 more boxes of apples. Each box contains 4 apples. He then gives half of his total apples to his friend. How many apples does Albert have left?",
                lines=3,
                elem_classes="prompt-box"
            )
            
        with gr.Accordion("⚙️ Configurations (You can change these settings)", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🐢 Baseline Model Settings")
                with gr.Column():
                    gr.Markdown("### 🚀 DC-Leap Settings")
                    dc_commit = gr.Slider(0.5, 1.0, value=0.65, step=0.01, label="Commit Threshold (τ_c)")
                    dc_draft = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Draft Threshold (τ_d)")
            gen_len = gr.Slider(64, 1024, value=256, step=64, label="Target Generation Length")

        generate_btn = gr.Button("🚀 START THE RACE (Simultaneous Inference)", size="lg", elem_classes="cta-button")

        with gr.Row():
            with gr.Column(elem_classes="token-wrapper"):
                gr.Markdown("## 🐢 Baseline Decoding\n*Tokens are decoded one by one through forward passes.*")
                base_time = gr.HTML("<div class='metric-badge base-badge'>⏱️ Ready</div>")
                base_output = gr.HTML("<div class='token-container'></div>")
                
            with gr.Column(elem_classes="token-wrapper"):
                gr.Markdown("## 🚀 DC-Leap Decoding \n*Future drafts (<span style='color: #a16207; font-weight: 700;'>yellow</span>) guide rapid generation with DCV.*")
                dcleap_time = gr.HTML("<div class='metric-badge dcleap-badge'>⏱️ Ready</div>")
                dcleap_output = gr.HTML("<div class='token-container'></div>")

        result_dashboard = gr.HTML("<div style='min-height: 50px;'></div>")

        generate_btn.click(
            fn=race_models,
            inputs=[prompt_input, gen_len, dc_commit, dc_draft],
            outputs=[base_output, base_time, dcleap_output, dcleap_time, result_dashboard]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
