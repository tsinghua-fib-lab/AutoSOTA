import torch
import ipywidgets as widgets
from .heatmap import render_context_heatmap
from sae_tools.model import format_with_tokenizer
import string

class FeatureActivationViewer:
    """
    SAE feature activation viewer class, encapsulates all functionality related to feature activation
    """
    
    def __init__(self, sparse_data, data_list, tokenizer):
        """
        Initialize the feature activation viewer
        
        Args:
            sparse_data: dictionary containing 'sparse_acts' and 'seq_lens'
            data_list: list of sample metadata
            tokenizer: tokenizer object
        """
        self.sparse_data = sparse_data
        self.data_list = data_list
        self.tokenizer = tokenizer
    
    def _tokenize_text(self, text):
        """Convert text to token string list, using tokenizer"""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def find_feature_activations(self, feature_idx, threshold=0.0):
        """
        Find all activation positions for a given feature in sparse_data
        Each sample only returns one activation position (the position with the maximum activation value)
        
        Args:
            feature_idx: feature index
            threshold: activation threshold, default 0.0
            
        Returns:
            activation position list, each position contains:
            - sample_idx: sample index
            - token_idx: token position in the sample (relative position)
            - global_token_idx: global token index (index in sparse_acts)
            - activation_value: activation value
        """
        sparse_acts = self.sparse_data['sparse_acts']
        seq_lens = self.sparse_data['seq_lens']
        
        # Extract all activations for the given feature from the sparse tensor
        indices = sparse_acts.indices()  # [2, NNZ]
        values = sparse_acts.values()    # [NNZ]
        
        # Find all positions where the feature index matches and the activation value is greater than the threshold
        feature_mask = (indices[1] == feature_idx) & (values > threshold)
        global_token_indices = indices[0][feature_mask]
        activation_values = values[feature_mask]
        
        # Convert global token index to (sample index, token position in the sample)
        all_activations = []
        sample_starts = torch.cumsum(torch.cat([torch.tensor([0]), seq_lens[:-1]]), dim=0)
        
        for global_idx, act_val in zip(global_token_indices, activation_values):
            sample_idx = torch.searchsorted(sample_starts, global_idx, right=True) - 1
            token_idx = global_idx - sample_starts[sample_idx]
            all_activations.append({
                'sample_idx': sample_idx.item(),
                'token_idx': token_idx.item(),
                'global_token_idx': global_idx.item(),
                'activation_value': act_val.item()
            })
        
        # Group by sample, each sample only retains the position with the maximum activation value
        sample_activations = {}
        for act in all_activations:
            sample_idx = act['sample_idx']
            if sample_idx not in sample_activations:
                sample_activations[sample_idx] = act
            else:
                # Keep the position with the larger activation value
                if act['activation_value'] > sample_activations[sample_idx]['activation_value']:
                    sample_activations[sample_idx] = act
        
        # Sort by activation value in descending order and return
        activations = list(sample_activations.values())
        activations.sort(key=lambda x: x['activation_value'], reverse=True)
        
        return activations
    
    def get_activation_values_for_sample(self, feature_idx, sample_idx, seq_len):
        """
        Get the activation values for a given feature in all token positions for a given sample
        
        Args:
            feature_idx: feature index
            sample_idx: sample index
            seq_len: sequence length of the sample
            
        Returns:
            dict: key is token_idx (position in the sample), value is activation value (0.0 if no activation)
        """
        sparse_acts = self.sparse_data['sparse_acts']
        seq_lens = self.sparse_data['seq_lens']
        
        # Calculate the global token start position for the sample
        sample_starts = torch.cumsum(torch.cat([torch.tensor([0]), seq_lens[:-1]]), dim=0)
        if sample_idx >= len(sample_starts):
            return {i: 0.0 for i in range(seq_len)}
        
        global_start = sample_starts[sample_idx].item()
        global_end = global_start + seq_len
        
        # Initialize the activation value dictionary (default all 0.0)
        activation_dict = {i: 0.0 for i in range(seq_len)}
        
        # Extract all activations for the given feature from the sparse tensor
        indices = sparse_acts.indices()  # [2, NNZ]
        values = sparse_acts.values()    # [NNZ]
        
        # Find all positions where the feature index matches and is within the sample range
        feature_mask = indices[1] == feature_idx
        matching_indices = indices[0][feature_mask]
        matching_values = values[feature_mask]
        
        # Filter out activations within the current sample range
        sample_mask = (matching_indices >= global_start) & (matching_indices < global_end)
        sample_global_indices = matching_indices[sample_mask]
        sample_values = matching_values[sample_mask]
        
        # Convert to token index within the sample and fill the dictionary
        for global_idx, act_val in zip(sample_global_indices, sample_values):
            token_idx = (global_idx - global_start).item()
            activation_dict[token_idx] = act_val.item()
        
        return activation_dict
    
    def get_context_for_activation(self, sample_idx, token_idx, context_window=10, feature_id=None):
        """
        Get the context token list for a given activation position
        
        Args:
            sample_idx: sample index
            token_idx: token position in the sample (relative to the entire sequence)
            context_window: context window size, default 10
            feature_id: feature index (optional, if provided returns the activation value dictionary)
            
        Returns:
            tuple: (token_str_list, activation_token_idx, activation_values_dict)
            - token_str_list: token string list (tokens within the context window)
            - activation_token_idx: index of the activated token in the list
            - activation_values_dict: dictionary, key is the index of the token within the context window, value is the activation value (if feature_id is None returns an empty dictionary)
        """
        if sample_idx >= len(self.data_list):
            return [], -1, {}
        
        # Get the full text for the sample
        sample = self.data_list[sample_idx]
        prompt_text = sample.get('prompt', "")
        response_text = sample.get('response', None)
        
        full_text, full_token_list, _ = format_with_tokenizer(
            self.tokenizer,
            prompt_text,
            response_text
        )

        full_token_str_list = self._tokenize_text(full_text)
        
        # Ensure token_idx is within the valid range
        if token_idx >= len(full_token_str_list):
            return [], -1, {}
        
        # Calculate the context range
        start_idx = max(0, token_idx - context_window)
        end_idx = min(len(full_token_str_list), token_idx + context_window + 1)
        
        # Return the context and activation position (relative to the context)
        context_tokens = full_token_str_list[start_idx:end_idx]
        activation_token_idx = token_idx - start_idx
        
        # Get the activation value dictionary
        activation_values_dict = {}
        if feature_id is not None:
            seq_len = len(full_token_str_list)
            sample_activation_dict = self.get_activation_values_for_sample(feature_id, sample_idx, seq_len)
            # Convert the token index within the sample to the index within the context window
            for i in range(len(context_tokens)):
                sample_token_idx = start_idx + i
                activation_values_dict[i] = sample_activation_dict.get(sample_token_idx, 0.0)
        
        return context_tokens, activation_token_idx, activation_values_dict
    
    def get_full_sequence_for_sample(self, sample_idx, feature_id=None):
        """
        Get the full token sequence and activation values for a given sample
        
        Args:
            sample_idx: sample index
            feature_id: feature index (optional, if provided returns the activation value dictionary)
            
        Returns:
            tuple: (token_str_list, activation_values_dict)
            - token_str_list: full token string list
            - activation_values_dict: dictionary, key is the token index, value is the activation value (if feature_id is None returns an empty dictionary)
        """
        if sample_idx >= len(self.data_list):
            return [], {}
        
        # Get the full text for the sample
        sample = self.data_list[sample_idx]
        prompt_text = sample.get('prompt', '')
        response_text = sample.get('response', None)
        
        full_text, full_token_list, _ = format_with_tokenizer(
            self.tokenizer,
            prompt_text,
            response_text
        )

        full_token_str_list = self._tokenize_text(full_text)

        # Get the activation value dictionary
        activation_values_dict = {}
        if feature_id is not None:
            seq_len = len(full_token_str_list)
            activation_values_dict = self.get_activation_values_for_sample(feature_id, sample_idx, seq_len)
        
        return full_token_str_list, activation_values_dict
    
    def _is_punctuation_token(self, token_str):
        """Check if the token is a punctuation token"""
        if not token_str:
            return False
        # Check if all characters are punctuation (after stripping whitespace)
        stripped = token_str.strip()
        if not stripped:
            return False
        # Check if all characters are punctuation
        return all(c in string.punctuation or c in '，。！？；：、' for c in stripped)
    
    def _escape_markdown_special_chars(self, text):
        """
        Escape markdown special characters in text
        
        Args:
            text: Input text string
            
        Returns:
            str: Text with markdown special characters escaped
        """
        escaped = text
        escaped = escaped.replace('*', '\\*')
        escaped = escaped.replace('_', '\\_')
        escaped = escaped.replace('`', '\\`')
        escaped = escaped.replace('[', '\\[')
        escaped = escaped.replace(']', '\\]')
        escaped = escaped.replace('(', '\\(')
        escaped = escaped.replace(')', '\\)')
        escaped = escaped.replace('#', '\\#')
        escaped = escaped.replace('+', '\\+')
        escaped = escaped.replace('-', '\\-')
        escaped = escaped.replace('.', '\\.')
        escaped = escaped.replace('!', '\\!')
        return escaped
    
    def _render_context_heatmap(self, token_str_list, activation_idx, act_val, sample_idx, token_idx, activation_values_dict=None):
        """Render the context heatmap HTML (backward compatible wrapper method)"""
        if activation_values_dict is None:
            activation_values_dict = {}
            for i in range(len(token_str_list)):
                if i == activation_idx:
                    activation_values_dict[i] = act_val
                else:
                    activation_values_dict[i] = 0.0
        
        # Call the independent rendering function, using the default orange color
        return render_context_heatmap(
            token_str_list,
            activation_values_dict,
            title_template=None,  # use default title logic
            positive_color=(255, 140, 0),  # orange
            negative_color=(255, 140, 0),  # orange
            line_height=1.8,
            activation_idx=activation_idx,
            act_val=act_val,
            sample_idx=sample_idx,
            token_idx=token_idx
        )
    
    def show_feature_activations(self, feature_id, threshold=0.0, max_display=10, show_full_text=False, container_width=None):
        """
        Show the activation positions and context for a given feature
        
        Args:
            feature_id: feature index
            threshold: activation threshold, default 0.0
            max_display: maximum number of activation positions to display, default 10
            show_full_text: whether to display the full text (prompt and response), default False
            container_width: container width, can be a CSS value (e.g. "800px", "100%", "90vw"), default None (no limit)
            
        Returns:
            widgets.VBox: widget container containing all displayed content
        """
        # Find the activation positions
        activations = self.find_feature_activations(feature_id, threshold)
        
        # Statistics
        num_activations = len(activations)
        unique_samples = len(set(a['sample_idx'] for a in activations))
        if activations:
            max_act = max(a['activation_value'] for a in activations)
            avg_act = sum(a['activation_value'] for a in activations) / num_activations
        else:
            max_act = 0
            avg_act = 0
        
        # Build the statistics HTML
        stats_html = f"""
        <div style='padding: 15px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px;'>
            <h4 style='margin-top: 0;'>Feature #{feature_id} Activation Statistics</h4>
            <ul style='margin: 5px 0;'>
                <li><b>Total Activations:</b> {num_activations}</li>
                <li><b>Activated Samples:</b> {unique_samples}</li>
                <li><b>Max Activation:</b> {max_act:.4f}</li>
                <li><b>Average Activation:</b> {avg_act:.4f}</li>
            </ul>
        </div>
        """
        
        widgets_list = [
            widgets.HTML("<h3>🔍 SAE Feature Activation Viewer</h3>"),
            widgets.HTML(stats_html)
        ]
        
        if not activations:
            widgets_list.append(widgets.HTML("<p style='color: orange;'>No activations found (try lowering the threshold)</p>"))
        else:
            # Display the activation position list (first max_display positions)
            display_count = min(max_display, len(activations))
            widgets_list.append(widgets.HTML(f"<h4>Activation List (showing first {display_count} of {num_activations}):</h4>"))
            
            for i, act in enumerate(activations[:display_count]):
                sample_idx = act['sample_idx']
                token_idx = act['token_idx']
                act_val = act['activation_value']
                
                # Build the activation position information HTML
                activation_info_html = f"""
                <div style='margin: 5px 0; padding: 5px; border-radius: 5px; background-color: #fff;'>
                    <p style='margin: 2px 0;'><b>Activation #{i+1}:</b> Sample #{sample_idx}, Token #{token_idx}, Activation: {act_val:.4f}</p>
                </div>
                """
                
                widgets_list.append(widgets.HTML(activation_info_html))
                
                # Based on show_full_text, decide whether to display the full text or the context
                if show_full_text:
                    # Display the activation heatmap for the full text
                    full_tokens, full_activation_dict = self.get_full_sequence_for_sample(sample_idx, feature_id)
                    if full_tokens:
                        # Convert the activation value dictionary to the index relative to the full text
                        full_activation_values_dict = {i: full_activation_dict.get(i, 0.0) for i in range(len(full_tokens))}
                        heatmap_html = self._render_context_heatmap(
                            full_tokens,
                            token_idx,  # index of the activated token in the full text
                            act_val,
                            sample_idx,
                            token_idx,
                            activation_values_dict=full_activation_values_dict
                        )
                        widgets_list.append(widgets.HTML(heatmap_html))
                    else:
                        widgets_list.append(widgets.HTML("<p style='color: red;'>Failed to get full text</p>"))
                else:
                    # Display the context heatmap
                    context_tokens, activation_token_idx, activation_values_dict = self.get_context_for_activation(
                        sample_idx,
                        token_idx,
                        context_window=15,
                        feature_id=feature_id
                    )
                    
                    if context_tokens and activation_token_idx >= 0:
                        heatmap_html = self._render_context_heatmap(
                            context_tokens,
                            activation_token_idx,
                            act_val,
                            sample_idx,
                            token_idx,
                            activation_values_dict=activation_values_dict
                        )
                        widgets_list.append(widgets.HTML(heatmap_html))
                    else:
                        widgets_list.append(widgets.HTML("<p style='color: red;'>Failed to get context</p>"))
                
                widgets_list.append(widgets.HTML("<hr style='margin: 10px 0;'>"))
        
        # Create the VBox and set the layout (if a width is specified)
        if container_width:
            layout = widgets.Layout(width=container_width, margin='0 auto')
            return widgets.VBox(widgets_list, layout=layout)
        else:
            return widgets.VBox(widgets_list)
    
    def get_markdown_text_with_activations(self, feature_id, threshold=0.0, max_display=10, newline_marker='↵'):
        """
        Output text, wrap the activated positions with <active>xxx</active> tags and replace newlines
        
        Args:
            feature_id: feature index
            threshold: activation threshold, default 0.0 (positions with activation value > threshold will be marked)
            max_display: maximum number of activated samples to display, default 10
            newline_marker: newline replacement marker, default '↵'
            
        Returns:
            str: formatted text string
        """
        activations = self.find_feature_activations(feature_id, threshold)
        
        if not activations:
            return ""
        
        display_count = min(max_display, len(activations))
        processed_samples = set()  # Track processed samples to avoid duplicates
        
        result_parts = []
        result_parts.append(f'''
The tokens wrapped with <active> are the activation positions of SAE feature {feature_id}.
Analyze the commonalities of this feature activation positions and return in the following format:
1. Name this feature;
2. Explain this feature in one sentence;
3. Explain the commonalities of the activation data.''')
        
        for act in activations[:display_count]:
            sample_idx = act['sample_idx']
            
            # Skip if sample already processed (each sample may have multiple activations)
            if sample_idx in processed_samples:
                continue
            processed_samples.add(sample_idx)
            
            # Get full token sequence and activation values
            full_tokens, full_activation_dict = self.get_full_sequence_for_sample(sample_idx, feature_id)
            
            if not full_tokens:
                continue
            
            # Build text for this sample, merging consecutive activated tokens
            text_parts = []
            current_active_group = []  # Track consecutive activated tokens
            in_active_group = False
            
            for token_idx_in_seq, token_str in enumerate(full_tokens):
                activation_value = full_activation_dict.get(token_idx_in_seq, 0.0)
                is_activated = activation_value > threshold
                
                # Replace newlines with marker
                processed_token = token_str.replace('\n', newline_marker)
                
                # Escape markdown special characters
                escaped_token = self._escape_markdown_special_chars(processed_token)
                
                if is_activated:
                    # Add to current active group
                    current_active_group.append(escaped_token)
                    in_active_group = True
                else:
                    # If we were in an active group, close it first
                    if in_active_group:
                        # Join consecutive activated tokens and wrap with <active> tag
                        active_text = "".join(current_active_group)
                        text_parts.append(f"<active>{active_text}</active>")
                        current_active_group = []
                        in_active_group = False
                    
                    # Add non-activated token
                    text_parts.append(escaped_token)
            
            # Handle case where sequence ends with active tokens
            if in_active_group:
                active_text = "".join(current_active_group)
                text_parts.append(f"<active>{active_text}</active>")
            
            # Join all tokens and add separator
            result_parts.append("".join(text_parts))
            result_parts.append("\n\n---\n\n")
        
        return "".join(result_parts)
