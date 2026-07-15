import html
import string

def render_context_heatmap(token_str_list, activation_values_dict, title_template=None,
                            positive_color=(255, 140, 0), negative_color=(255, 140, 0),
                            line_height=1.8, activation_idx=None, act_val=None,
                            sample_idx=None, token_idx=None):
    """
    Render the context heatmap HTML (independent function, supports custom colors and titles)
    
    Args:
        token_str_list: token string list
        activation_values_dict: dictionary, key is the token index, value is the activation value/difference value
        title_template: title template function, receives (i, val) parameters, returns the title string. If None, use the default title
        positive_color: RGB color tuple for positive values, default (255, 140, 0) orange
        negative_color: RGB color tuple for negative values, default (255, 140, 0) orange
        line_height: line height, default 1.8
        activation_idx: activation position index (for default title, optional)
        act_val: activation value (for default title, optional)
        sample_idx: sample index (for default title, optional)
        token_idx: token index (for default title, optional)
        
    Returns:
        str: HTML string
    """
    def _is_punctuation_token(token_str):
        """Check if the token is a punctuation token"""
        if not token_str:
            return False
        stripped = token_str.strip()
        if not stripped:
            return False
        return all(c in string.punctuation or c in ',.!?;:,\'\"' for c in stripped)
    
    html_parts = [f'<div style="padding: 0px; border-radius: 3px; background-color: #fafafa; line-height: {line_height}; word-break: normal; font-size: 0; overflow-wrap: break-word; word-wrap: break-word; max-width: 100%; box-sizing: border-box;">']
    
    # Calculate the normalized values (for color transparency)
    act_values_list = list(activation_values_dict.values())
    abs_values = [abs(v) for v in act_values_list]
    max_abs_value = max(abs_values) if abs_values else 1.0
    
    for i, token_str in enumerate(token_str_list):
        val = activation_values_dict.get(i, 0.0)
        
        # Check if the token is a punctuation token
        is_punctuation = _is_punctuation_token(token_str)
        
        # Set the color based on the value
        if abs(val) > 1e-6 and max_abs_value > 1e-6:
            normalized_abs_val = abs(val) / max_abs_value
            alpha = 0.3 + (normalized_abs_val * 0.6)  # alpha range: 0.3-0.9
            
            if val > 0:
                # Positive value: use positive_color
                bg_color = f"rgba({positive_color[0]}, {positive_color[1]}, {positive_color[2]}, {alpha:.2f})"
            else:
                # Negative value: use negative_color
                bg_color = f"rgba({negative_color[0]}, {negative_color[1]}, {negative_color[2]}, {alpha:.2f})"
            
            text_color = "#fff" if normalized_abs_val > 0.5 else "#000"
        else:
            # Token with value 0: light gray background
            bg_color = "#f0f0f0"
            text_color = "#000"
        
        # Generate the title
        if title_template is not None:
            title = title_template(i, val)
        else:
            # Default title
            if i == activation_idx and act_val is not None:
                title = f"Activated Token (Act: {act_val:.4f}, Sample: {sample_idx}, Pos: {token_idx})" if sample_idx is not None else f"Token {i} (Act: {val:.4f})"
            elif abs(val) > 1e-6:
                title = f"Token {i} (Act: {val:.4f})"
            else:
                title = f"Token {i}"
        
        # Special handling for newline characters
        if '\n' in token_str:
            parts = token_str.split('\n')
            for part_idx, part in enumerate(parts):
                if part:
                    escaped_part = html.escape(part)
                    span_html = f'<span style="background-color: {bg_color}; color: {text_color}; padding: 0px; border-radius: 5px; font-family: monospace; font-size: 12px; line-height: {line_height}; white-space: normal; text-align: left; display: inline; letter-spacing: -0.5px;" title="{title}">{escaped_part}</span>'
                    html_parts.append(span_html)
                
                if part_idx < len(parts) - 1:
                    newline_placeholder = f'<span style="background-color: {bg_color}; color: {text_color}; padding: 0px; border-radius: 5px; font-family: monospace; font-size: 12px; line-height: {line_height}; white-space: normal; text-align: left; display: inline; letter-spacing: -0.5px;" title="{title}">↵</span>'
                    html_parts.append(newline_placeholder)
                    html_parts.append('<br>')
        else:
            # No newline characters, normal processing
            escaped_token = html.escape(token_str)
            # If the token is a punctuation token, add a zero-width non-breaking space in front
            if is_punctuation and i > 0:
                escaped_token = '\u2060' + escaped_token
            span_html = f'<span style="background-color: {bg_color}; color: {text_color}; padding: 0px; border-radius: 5px; font-family: monospace; font-size: 12px; line-height: {line_height}; white-space: normal; text-align: left; display: inline; letter-spacing: -0.5px;" title="{title}">{escaped_token}</span>'
            html_parts.append(span_html)
    
    html_parts.append('</div>')
    return "".join(html_parts)