import torch
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from .heatmap import render_context_heatmap

class LogitLensAnalyzer:
    def __init__(self, model, acts_cache, sae=None):
        """
        Args:
            model: HookedTransformer
            acts_cache: [Batch, Seq, Dim] or [Seq, Dim]
            sae: SAE object
        """
        self.model = model
        # Ensure acts is [Seq, Dim]
        if len(acts_cache.shape) == 3:
            self.acts = acts_cache[0].detach().cpu()
        else:
            self.acts = acts_cache.detach().cpu()
            
        self.sae_weights = None
        if sae is not None and hasattr(sae, 'W_dec'):
            self.sae_weights = sae.W_dec.detach().cpu()

    # --- New method: Get the activation of a specific feature across the sequence ---
    def get_feature_activations(self, sae_idx):
        """Get the activation value of a specific SAE feature across all tokens"""
        if sae_idx < 0 or sae_idx >= self.acts.shape[1]:
            # Return all zeros to prevent out of bounds
            return torch.zeros(self.acts.shape[0])
        # Get a specific column [Seq]
        return self.acts[:, sae_idx]

    def get_top_sae_features(self, token_idx, k=20):
        latent_vec = self.acts[token_idx]
        top_vals, top_inds = torch.topk(latent_vec, k=k)
        return top_inds.numpy(), top_vals.numpy()

    def _apply_lens(self, input_vec, apply_final_block=False):
        vec = input_vec.to(self.model.cfg.device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if apply_final_block:
                vec = self.model.blocks[-1](vec)
            ln_vec = self.model.ln_final(vec)
            logits = self.model.unembed(ln_vec)
        return logits[0, 0]

    def analyze_step(self, token_idx, sae_idx, use_final_block):
        latent_vec = self.acts[token_idx]
        
        # Context (for display)
        top_vals_ctx, top_dims_ctx = torch.topk(latent_vec, k=20)
        df_context = pd.DataFrame({
            "Dim Index": top_dims_ctx.numpy(),
            "Value": top_vals_ctx.numpy()
        })

        # Determine SAE Index
        if sae_idx == -1:
            sae_idx = torch.argmax(latent_vec).item()
        
        if self.sae_weights is None:
            return df_context, None, None, "Error: SAE Weights Not Loaded", sae_idx
        if sae_idx >= self.sae_weights.shape[0]:
            return df_context, None, None, f"Error: SAE Index {sae_idx} Out of Bounds", sae_idx
                
        target_vec = self.sae_weights[sae_idx]
        current_act_val = latent_vec[sae_idx].item()
        source_name = f"SAE Latent #{sae_idx} (Act: {current_act_val:.2f})"

        # Logits
        logits = self._apply_lens(target_vec, apply_final_block=use_final_block)
        probs = torch.softmax(logits, dim=0)

        # Top/Bottom tables
        top_vals, top_ids = torch.topk(logits, k=20)
        df_top = self._format_logit_df(top_vals, top_ids, probs)

        bot_vals, bot_ids = torch.topk(logits, k=20, largest=False)
        df_bottom = self._format_logit_df(bot_vals, bot_ids, probs)

        # Return the real sae_idx for GUI use
        return df_context, df_top, df_bottom, source_name, sae_idx

    def _format_logit_df(self, vals, ids, all_probs):
        tokens = [self.model.to_string(tid) for tid in ids]
        selected_probs = all_probs[ids]
        return pd.DataFrame({
            "Token": tokens,
            "Logit": vals.cpu().numpy(),
            "Prob": selected_probs.cpu().numpy()
        })

class LogitLensGUI:
    def __init__(self, analyzer, tokens_str_list, max_heatmap_tokens=None):
        self.analyzer = analyzer

        # Important: render_context_heatmap will escape the tokens internally.
        # If you also escape here, it will cause "double escaping" (e.g. < becomes &amp;lt;).
        import html
        self.tokens_str_raw = [t if t is not None else "" for t in tokens_str_list]

        # Verify length consistency
        acts_seq_len = self.analyzer.acts.shape[0]
        tokens_len = len(self.tokens_str_raw)
        if tokens_len != acts_seq_len:
            print(f"⚠️  Warning: tokens_str_list length ({tokens_len}) does not match the activation sequence length ({acts_seq_len})!")
            print("   Truncate or pad to match the same length.")
            # Truncate or pad to match the activation sequence length
            if tokens_len > acts_seq_len:
                self.tokens_str_raw = self.tokens_str_raw[:acts_seq_len]
                print(f"   tokens_str truncated to {acts_seq_len} elements")
            else:
                # If tokens are shorter, pad with empty strings
                self.tokens_str_raw.extend([""] * (acts_seq_len - tokens_len))
                print(f"   tokens_str padded to {acts_seq_len} elements")

        # UI displayed tokens (escape HTML special characters to prevent symbols like < from breaking the layout)
        self.tokens_str = [html.escape(t) for t in self.tokens_str_raw]

        # Maximum number of tokens to display in the heatmap; None represents the full sequence
        self.max_heatmap_tokens = max_heatmap_tokens

        self._init_widgets()
        self._layout_ui()
        # Initial trigger
        self._on_token_change({'new': 0})

    def _init_widgets(self):
        # 1. Token Select
        self.w_token = widgets.ToggleButtons(
            options=[(f"{i}: {t}", i) for i, t in enumerate(self.tokens_str)],
            value=0, description='', 
            style={'button_width': 'initial'}, 
            layout=widgets.Layout(width='100%', flex_flow='row wrap', display='flex')
        )

        # 2. Controls
        self.w_sae_manual = widgets.IntText(value=-1, description='Manual ID:', layout=widgets.Layout(width='140px'))
        self.w_sae_select = widgets.Dropdown(options=[], description='Top Active Dims:', layout=widgets.Layout(width='220px'))
        self.w_block = widgets.Checkbox(value=False, description='Apply Final Block', indent=False, layout=widgets.Layout(width='auto'))

        # --- New component: sequence heatmap row ---
        # Use HTML components to render text spans with background colors
        self.w_heatmap_row = widgets.HTML(
            value="Loading sequence overview...",
            layout=widgets.Layout(width='100%', margin='5px 0px 15px 0px', overflow='auto')
        )

        self.out = widgets.Output()

        # Events
        self.w_token.observe(self._on_token_change, names='value')
        self.w_sae_select.observe(self._on_dropdown_change, names='value')
        self.w_sae_manual.observe(self._on_render_trigger, names='value')
        self.w_block.observe(self._on_render_trigger, names='value')

    def _layout_ui(self):
        ctrl_row = widgets.HBox(
            [self.w_sae_select, self.w_sae_manual, self.w_block], 
            layout=widgets.Layout(margin='10px 0', align_items='center', grid_gap='15px')
        )
        
        self.ui = widgets.VBox([
            widgets.HTML("<h3>🔬 Modular Logit Lens Explorer</h3>"), 
            widgets.HTML("<b>Select Token Position:</b>"),
            self.w_token,
            widgets.HTML("<hr style='margin: 5px 0'>"),
            ctrl_row,
            # Put the heatmap row below the controls, above the output
            widgets.HTML("<b>Selected Feature Activation Across Sequence:</b>"),
            self.w_heatmap_row,
            self.out
        ])

    def display(self, width='100%'):
        self.ui.layout.width = width
        display(self.ui)

    # ================= Logics =================

    def _update_sae_options(self, token_idx):
        top_inds, top_vals = self.analyzer.get_top_sae_features(token_idx, k=20)
        options = [(f"#{idx:<5} (Act: {val:.1f})", int(idx)) for idx, val in zip(top_inds, top_vals)]
        
        # Temporarily unbind all related observers to avoid multiple renders
        self.w_sae_select.unobserve(self._on_dropdown_change, names='value')
        self.w_sae_manual.unobserve(self._on_render_trigger, names='value')
        
        self.w_sae_select.options = options
        if options:
            self.w_sae_select.value = options[0][1]
            self.w_sae_manual.value = options[0][1] # Synchronize manual
        
        # Rebind observers
        self.w_sae_select.observe(self._on_dropdown_change, names='value')
        self.w_sae_manual.observe(self._on_render_trigger, names='value')

    def _on_token_change(self, change):
        self._update_sae_options(change['new'])
        self._on_render_trigger(None)

    def _on_dropdown_change(self, change):
        if change['new'] is not None:
            # Temporarily unbind observers to avoid additional renders
            self.w_sae_manual.unobserve(self._on_render_trigger, names='value')
            self.w_sae_manual.value = change['new']
            self.w_sae_manual.observe(self._on_render_trigger, names='value')

    def _on_render_trigger(self, change):
        t_idx = self.w_token.value
        s_idx = self.w_sae_manual.value
        use_block = self.w_block.value
        self._render_analysis(t_idx, s_idx, use_block)

    # ================= Rendering =================

    # --- New rendering function: generate heatmap HTML ---
    def _render_heatmap_row_html(self, sae_idx, current_token_idx):
        # 1) Get the activation values for the entire sequence
        acts_seq_t = self.analyzer.get_feature_activations(sae_idx)
        try:
            acts_seq = acts_seq_t.detach().cpu().numpy()
        except AttributeError:
            acts_seq = np.asarray(acts_seq_t)

        # 2) Align lengths
        seq_len = min(len(self.tokens_str_raw), len(acts_seq))
        if seq_len <= 0:
            return "<div style='color: #999;'>No tokens/activations to render.</div>"

        total_len = seq_len
        display_start = 0
        display_end = total_len
        truncated = False

        # 3) Optional truncation (None represents the full sequence)
        if self.max_heatmap_tokens is not None and total_len > self.max_heatmap_tokens:
            half_window = self.max_heatmap_tokens // 2
            display_start = max(0, current_token_idx - half_window)
            display_end = min(total_len, display_start + self.max_heatmap_tokens)
            if display_end - display_start < self.max_heatmap_tokens:
                display_start = max(0, display_end - self.max_heatmap_tokens)
            truncated = True

        # 4) Build the tokens + values to pass to render_context_heatmap (note: pass raw tokens to avoid double escaping)
        tokens_to_render = self.tokens_str_raw[display_start:display_end]
        acts_to_render = acts_seq[display_start:display_end]
        activation_values_dict = {i: float(v) for i, v in enumerate(acts_to_render)}

        local_activation_idx = current_token_idx - display_start
        act_val = None
        if 0 <= local_activation_idx < len(acts_to_render):
            act_val = float(acts_to_render[local_activation_idx])

        heatmap_html = render_context_heatmap(
            tokens_to_render,
            activation_values_dict,
            title_template=None,
            positive_color=(255, 140, 0),
            negative_color=(255, 140, 0),
            line_height=1.8,
            activation_idx=local_activation_idx,
            act_val=act_val,
            sample_idx=None,
            token_idx=current_token_idx,
        )

        html_parts = []
        if truncated:
            html_parts.append(
                f'<div style="padding: 5px; background-color: #fff3cd; border: 1px solid #ffc107; '
                f'border-radius: 4px; margin-bottom: 5px; font-size: 12px; color: #856404;">'
                f'⚠️  Sequence较长（共 {total_len} 个 tokens），仅显示位置 {display_start} - {display_end - 1}'
                f'（当前选中位置: {current_token_idx}）</div>'
            )

        # Keep the same "fixed height + scrollable" experience as the old version
        html_parts.append('<div style="padding: 5px; border: 1px solid #eee; border-radius: 4px; max-height: 200px; overflow-y: auto;">')
        html_parts.append(heatmap_html)
        html_parts.append('</div>')
        return "".join(html_parts)

    def _render_analysis(self, t_idx, s_idx, use_block):
        # Note here we receive the real real_s_idx
        df_ctx, df_top, df_bot, src_name, real_s_idx = self.analyzer.analyze_step(t_idx, s_idx, use_block)

        # --- Core change: update the heatmap row HTML ---
        # We use the real_s_idx confirmed after analysis to render, ensuring accuracy
        self.w_heatmap_row.value = self._render_heatmap_row_html(real_s_idx, t_idx)
        
        with self.out:
            clear_output(wait=True)
            # (Top Info bar removed, because now there is a heatmap row, which is redundant, you can add it back if needed)
            
            if df_top is None:
                print(f"❌ Error: {src_name}")
                return

            out_style = widgets.Layout(width='32%')
            out_top = widgets.Output(layout=out_style)
            out_bot = widgets.Output(layout=out_style)
            out_ctx = widgets.Output(layout=out_style)

            with out_top: display(self.render_logit_frame(df_top, "🟢 Top 20 Predictions"))
            with out_bot: display(self.render_logit_frame(df_bot, "🔴 Bottom 20 (Suppressed)", is_bottom=True))
            with out_ctx: display(self.render_context_frame(df_ctx))

            display(widgets.HBox([out_top, out_bot, out_ctx], layout=widgets.Layout(width='100%', justify_content='space-between')))

    def render_logit_frame(self, df, title, is_bottom=False):
        if df is None: return None
        format_dict = {"Logit": "{:.2f}", "Prob": "{:.2%}"}
        styler = df.style.format(format_dict).set_caption(title)
        cmap = "Reds_r" if is_bottom else "Blues"
        return styler.background_gradient(subset=["Logit"], cmap=cmap)

    def render_context_frame(self, df):
        if df is None: return None
        return df.style.format({"Value": "{:.2f}"})\
                 .background_gradient(subset=["Value"], cmap="Greens")\
                 .set_caption("Current Pos Top Latents")