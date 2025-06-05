# attention_visualizer.py
"""
Attention visualization tools for Chameleon model.

This module provides functions to visualize attention patterns between 
different token types: image tokens, text tokens, and output tokens.

Note that you need to install the chameleon-7b model first and make some changes to the original file,
since it doesn't support output_attetions=True.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import json

def identify_token_boundaries(tokens: List[int], vocab, prompt_length: int) -> Dict[str, Dict[str, int]]:
    boundaries = {
        "text": {"start": None, "end": None},
        "image": {"start": None, "end": None},
        "output": {"start": prompt_length, "end": len(tokens)},
        "prompt": {"start": 0, "end": prompt_length}
    }
    
    in_image = False
    
    for i, token_id in enumerate(tokens):
        if token_id == vocab.begin_image:
            if boundaries["image"]["start"] is None:
                boundaries["image"]["start"] = i
            in_image = True
        elif token_id == vocab.end_image:
            boundaries["image"]["end"] = i + 1
            in_image = False
    
    # Text tokens are in prompt but not in image
    if boundaries["image"]["start"] is not None:
        if boundaries["image"]["start"] > 1:
            boundaries["text"]["start"] = 0
            boundaries["text"]["end"] = boundaries["image"]["start"]
        elif boundaries["image"]["end"] < prompt_length:
            boundaries["text"]["start"] = boundaries["image"]["end"]
            boundaries["text"]["end"] = prompt_length
    else:
        # No image tokens, all prompt tokens are text
        boundaries["text"]["start"] = 0
        boundaries["text"]["end"] = prompt_length
    
    return boundaries

def map_token_types(tokens: List[int], vocab, prompt_length: int) -> List[str]:
    token_types = []
    in_image = False
    
    for i, token_id in enumerate(tokens):
        if i >= prompt_length:
            token_types.append("output")
        elif token_id == vocab.begin_image:
            token_types.append("image_boundary")
            in_image = True
        elif token_id == vocab.end_image:
            token_types.append("image_boundary")
            in_image = False
        elif in_image:
            token_types.append("image")
        elif token_id in [vocab.bos_id, vocab.eos_id, vocab.eot_id]:
            token_types.append("special")
        else:
            token_types.append("text")
    
    return token_types

def visualize_attention(
    layer_idx: int,
    attention_weights: torch.Tensor,
    token_types: List[str],
    output_dir: str,
    pool_size: int = 1,
    title: Optional[str] = None
) -> Tuple[np.ndarray, Dict]:
    # Average across attention heads
    avg_attention = torch.mean(attention_weights, dim=0).float().cpu().numpy()
    
    # Apply pooling if needed
    if pool_size > 1:
        pooled_attn = np.zeros((avg_attention.shape[0] // pool_size, 
                                avg_attention.shape[1] // pool_size))
        for i in range(0, avg_attention.shape[0], pool_size):
            for j in range(0, avg_attention.shape[1], pool_size):
                if i + pool_size <= avg_attention.shape[0] and j + pool_size <= avg_attention.shape[1]:
                    pooled_attn[i // pool_size, j // pool_size] = np.mean(
                        avg_attention[i:i+pool_size, j:j+pool_size]
                    )
        avg_attention = pooled_attn
        # Also pool token types
        pooled_types = [token_types[i] for i in range(0, len(token_types), pool_size) 
                      if i + pool_size <= len(token_types)]
        token_types = pooled_types
    
    # Create figure
    plt.figure(figsize=(12, 10), dpi=300)
    
    # Create mask for different token type regions
    type_to_idx = {t: i for i, t in enumerate(['text', 'image', 'output', 'special', 'image_boundary'])}
    token_type_indices = np.array([type_to_idx.get(t, 0) for t in token_types])
    
    # Create color boundaries
    boundaries = {}
    for token_type in ['text', 'image', 'output', 'image_boundary']:
        indices = [i for i, t in enumerate(token_types) if t == token_type]
        if indices:
            boundaries[token_type] = (min(indices), max(indices))
    
    # Plot attention heatmap with log normalization
    vmin = max(0.0001, np.min(avg_attention[avg_attention > 0]))
    vmax = np.max(avg_attention)
    log_norm = LogNorm(vmin=vmin, vmax=vmax)
    
    ax = sns.heatmap(avg_attention, cmap="viridis", norm=log_norm,
                     cbar_kws={'label': 'Attention score'})
    
    # Add boundary lines and labels for token types
    colors = {'text': 'blue', 'image': 'red', 'output': 'green', 'image_boundary': 'purple'}
    
    for token_type, (start, end) in boundaries.items():
        if token_type != 'image_boundary':  # Don't add separate lines for boundaries
            color = colors[token_type]
            plt.axhline(y=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axhline(y=end+1, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axvline(x=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axvline(x=end+1, color=color, linestyle='-', linewidth=2, alpha=0.7)
            
            # Add labels in corners of regions
            plt.text(start + (end-start)//2, start - 0.5, token_type, 
                     fontsize=12, color='white', weight='bold',
                     bbox=dict(facecolor=color, alpha=0.8))
    
    # Set title and labels
    title = title or f"Layer {layer_idx+1} Attention"
    plt.title(title, fontsize=16)
    plt.xlabel("Key position", fontsize=14)
    plt.ylabel("Query position", fontsize=14)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"layer_{layer_idx}_attention.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Compute cross-attention metrics
    metrics = compute_attention_metrics(avg_attention, token_types)
    
    # Find top attention pairs
    top_pairs = find_top_attention_pairs(avg_attention, token_types, k=10)
    
    return avg_attention, {"metrics": metrics, "top_pairs": top_pairs}

def compute_attention_metrics(attention_matrix: np.ndarray, token_types: List[str]) -> Dict:
    # Group indices by token type
    type_indices = {}
    for t in set(token_types):
        type_indices[t] = [i for i, token_type in enumerate(token_types) if token_type == t]
    
    # Compute cross-attention metrics
    metrics = {}
    for src_type, src_indices in type_indices.items():
        if not src_indices:
            continue
            
        for tgt_type, tgt_indices in type_indices.items():
            if not tgt_indices:
                continue
                
            # Extract attention from this type to that type
            type_attention = attention_matrix[np.ix_(src_indices, tgt_indices)]
            metrics[f"{src_type}_to_{tgt_type}"] = {
                "mean": float(np.mean(type_attention)),
                "max": float(np.max(type_attention)),
                "min": float(np.min(type_attention)),
                "std": float(np.std(type_attention))
            }
    
    return metrics

def find_top_attention_pairs(attention_matrix: np.ndarray, token_types: List[str], k: int = 10) -> List[Dict]:
    flat_indices = np.argsort(attention_matrix.flatten())[-k:]
    
    # Convert flat indices to 2D indices
    top_pairs = []
    for flat_idx in flat_indices:
        row = flat_idx // attention_matrix.shape[1]
        col = flat_idx % attention_matrix.shape[1]
        
        if row < len(token_types) and col < len(token_types):
            top_pairs.append({
                "query_idx": int(row),
                "key_idx": int(col),
                "query_type": token_types[row],
                "key_type": token_types[col],
                "attention_score": float(attention_matrix[row, col])
            })
    
    return sorted(top_pairs, key=lambda x: x["attention_score"], reverse=True)

def analyze_all_layers(
    all_attention_weights: List[torch.Tensor],
    tokens: List[int],
    vocab,
    prompt_length: int,
    output_dir: str = "attention_analysis",
    pool_size: int = 1
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    
    # Map tokens to types
    token_types = map_token_types(tokens, vocab, prompt_length)
    
    # Analyze each layer
    results = {}
    for layer_idx, layer_attention in enumerate(all_attention_weights):
        print(f"Analyzing layer {layer_idx+1}/{len(all_attention_weights)}...")
        
        # Visualize and analyze
        _, layer_results = visualize_attention(
            layer_idx=layer_idx,
            attention_weights=layer_attention,
            token_types=token_types,
            output_dir=output_dir,
            pool_size=pool_size,
            title=f"Layer {layer_idx+1} Attention"
        )
        
        results[f"layer_{layer_idx}"] = layer_results
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, "attention_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary visualization showing key metrics across layers
    plot_cross_attention_trends(results, output_dir)
    
    return results

def plot_cross_attention_trends(results: Dict, output_dir: str):
    patterns = [
        "text_to_image", 
        "image_to_text",
        "output_to_text", 
        "output_to_image"
    ]
    
    # Collect metrics across layers
    layer_indices = []
    metrics_by_pattern = {pattern: [] for pattern in patterns}
    
    for layer_key, layer_data in results.items():
        if not layer_key.startswith("layer_"):
            continue
            
        layer_idx = int(layer_key.split("_")[1])
        layer_indices.append(layer_idx)
        
        for pattern in patterns:
            # Find any keys that contain this pattern
            for key, value in layer_data["metrics"].items():
                if pattern in key:
                    metrics_by_pattern[pattern].append(value["mean"])
                    break
            else:
                # Pattern not found, add 0
                metrics_by_pattern[pattern].append(0)
    
    # Create trend plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple']
    for i, (pattern, values) in enumerate(metrics_by_pattern.items()):
        if values:  # Only plot if we have data
            plt.plot(layer_indices, values, marker='o', linestyle='-', 
                     color=colors[i % len(colors)], label=pattern)
    
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Mean Attention Score", fontsize=14)
    plt.title("Cross-Modal Attention Trends Across Layers", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "cross_attention_trends.png"), bbox_inches='tight')
    plt.close()


def visualize_attention_summary(
    attention_data: List[Dict], 
    output_dir: str, 
    title: str = "Attention Summary"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key patterns to track
    patterns = ["text_to_image", "image_to_text", "output_to_text", "output_to_image"]
    
    # Prepare data for plotting
    layer_indices = []
    pattern_values = {pattern: [] for pattern in patterns}
    
    for layer_data in attention_data:
        layer_idx = layer_data["layer"]
        layer_indices.append(layer_idx)
        
        # Process top patterns
        for pattern in pattern_values:
            pattern_found = False
            for p in layer_data["top_patterns"]:
                pattern_key = f"{p['from']}_to_{p['to']}"
                if pattern in pattern_key:
                    pattern_values[pattern].append(p["value"])
                    pattern_found = True
                    break
            
            if not pattern_found:
                pattern_values[pattern].append(0.0)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'purple']
    for i, (pattern, values) in enumerate(pattern_values.items()):
        if any(values):  # Only plot if we have data
            plt.plot(layer_indices, values, marker='o', linestyle='-', 
                     color=colors[i % len(colors)], label=pattern)
    
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Attention Strength", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "attention_summary.png"), bbox_inches='tight')
    plt.close()
    
    # Save data as JSON
    with open(os.path.join(output_dir, "attention_summary.json"), "w") as f:
        json.dump({
            "layers": layer_indices,
            "patterns": pattern_values
        }, f, indent=2)

def create_attention_heatmap(
    attention_data: List[Dict],
    output_dir: str,
    sequence_length: int,
    prompt_length: int,
    token_type_map: Optional[List[str]] = None
):
    os.makedirs(output_dir, exist_ok=True)
    
    # If no token type map is provided, create a simple one
    if token_type_map is None:
        token_type_map = ["prompt" if i < prompt_length else "output" 
                         for i in range(sequence_length)]
    
    # Create heatmaps for each layer
    for layer_data in attention_data:
        layer_idx = layer_data["layer"]
        
        # Create a synthetic attention matrix for visualization
        # This is a simplified representation based on the attention patterns
        matrix_size = min(sequence_length, 100)  # Limit size for visualization
        ratio = sequence_length / matrix_size if sequence_length > matrix_size else 1
        
        # Create a base low-attention matrix
        attn_matrix = np.ones((matrix_size, matrix_size)) * 0.01
        
        # Fill in higher attention values based on patterns
        for pattern in layer_data["top_patterns"]:
            from_type = pattern["from"]
            to_type = pattern["to"]
            value = pattern["value"]
            
            # Find indices for this pattern
            from_indices = [i // ratio for i, t in enumerate(token_type_map) if t == from_type and i // ratio < matrix_size]
            to_indices = [i // ratio for i, t in enumerate(token_type_map) if t == to_type and i // ratio < matrix_size]
            
            # Set attention values
            for i in from_indices:
                for j in to_indices:
                    attn_matrix[int(i), int(j)] = value
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Use log normalization for better visualization
        log_norm = LogNorm(vmin=0.001, vmax=max(0.1, attn_matrix.max()))
        
        # Create custom colormap
        colors = plt.cm.viridis(np.linspace(0, 1, 256))
        custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors)
        
        # Plot heatmap
        ax = sns.heatmap(attn_matrix, cmap=custom_cmap, norm=log_norm,
                        cbar_kws={'label': 'Attention score'})
        
        # Add boundary lines for prompt/output
        prompt_boundary = min(prompt_length / ratio, matrix_size)
        if prompt_boundary < matrix_size:
            plt.axhline(y=prompt_boundary, color='white', linestyle='--', alpha=0.7)
            plt.axvline(x=prompt_boundary, color='white', linestyle='--', alpha=0.7)
            
            # Add labels
            plt.text(prompt_boundary/2, -0.5, "Prompt", ha='center', fontsize=12)
            plt.text(prompt_boundary + (matrix_size-prompt_boundary)/2, -0.5, "Output", ha='center', fontsize=12)
        
        plt.title(f"Layer {layer_idx+1} Attention", fontsize=16)
        plt.xlabel("Key position", fontsize=14)
        plt.ylabel("Query position", fontsize=14)
        
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_heatmap.png"), bbox_inches='tight')
        plt.close()

def visualize_token_type_attention(attention_data: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique token types
    token_types = set()
    for layer_data in attention_data:
        for pattern in layer_data["top_patterns"]:
            token_types.add(pattern["from"])
            token_types.add(pattern["to"])
    
    # Remove 'special' tokens for clarity
    if 'special' in token_types:
        token_types.remove('special')
    
    token_types = sorted(list(token_types))
    
    # Create matrices for each layer
    num_layers = len(attention_data)
    type_attention_matrices = []
    
    for layer_data in attention_data:
        # Initialize matrix with zeros
        matrix = np.zeros((len(token_types), len(token_types)))
        
        # Fill matrix with attention values
        for pattern in layer_data["top_patterns"]:
            from_type = pattern["from"]
            to_type = pattern["to"]
            
            if from_type in token_types and to_type in token_types:
                from_idx = token_types.index(from_type)
                to_idx = token_types.index(to_type)
                matrix[from_idx, to_idx] = pattern["value"]
        
        type_attention_matrices.append(matrix)
    
    # Create plots for each layer
    for layer_idx, matrix in enumerate(type_attention_matrices):
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        ax = sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".3f",
                        xticklabels=token_types, yticklabels=token_types)
        
        plt.title(f"Layer {layer_idx+1} Token Type Attention", fontsize=16)
        plt.xlabel("Attending To", fontsize=14)
        plt.ylabel("Attending From", fontsize=14)
        
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_type_attention.png"), bbox_inches='tight')
        plt.close()
    
    # Create average across all layers
    if type_attention_matrices:
        avg_matrix = np.mean(type_attention_matrices, axis=0)
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(avg_matrix, cmap="YlOrRd", annot=True, fmt=".3f",
                        xticklabels=token_types, yticklabels=token_types)
        
        plt.title("Average Token Type Attention Across All Layers", fontsize=16)
        plt.xlabel("Attending To", fontsize=14)
        plt.ylabel("Attending From", fontsize=14)
        
        plt.savefig(os.path.join(output_dir, "average_type_attention.png"), bbox_inches='tight')
        plt.close()

def process_attention_data(
    attention_data: List[Dict],
    full_sequence: List[int],
    prompt_length: int,
    output_dir: str = "attention_analysis"
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing attention data for {len(attention_data)} layers...")
    print(f"Sequence length: {len(full_sequence)}, Prompt length: {prompt_length}")
    
    # Save raw attention data
    with open(os.path.join(output_dir, "attention_data.json"), "w") as f:
        # Convert to JSON-serializable format
        serializable_data = json.dumps(attention_data, default=lambda o: o if not isinstance(o, np.ndarray) else o.tolist())
        f.write(serializable_data)
    
    # Create overall summary visualization
    visualize_attention_summary(attention_data, output_dir)
    
    # Create heatmap visualizations
    create_attention_heatmap(attention_data, output_dir, len(full_sequence), prompt_length)
    
    # Create token type attention visualizations
    visualize_token_type_attention(attention_data, output_dir)
    
    # Display key insights
    print("\nKey attention insights:")
    
    important_patterns = ["text_to_image", "image_to_text", "output_to_text", "output_to_image"]
    
    for layer_idx, layer_data in enumerate(attention_data):
        print(f"\nLayer {layer_idx+1}:")
        
        # Show top patterns
        print("  Top attention patterns:")
        for pattern in layer_data["top_patterns"][:3]:
            print(f"    {pattern['from']} → {pattern['to']}: {pattern['value']:.6f}")
    
    print(f"\nVisualization results saved to {output_dir}")
    return output_dir

def visualize_combined_attention(
    layer_attention_weights: torch.Tensor,  # [32, 1, 32, 1045, 4096] - from first forward pass
    output_attention_weights: torch.Tensor,  # [576, 1, 32, 1, 4096] - from autoregressive output tokens
    tokens: List[int],
    vocab,
    prompt_length: int,
    output_dir: str = "combined_attention_analysis",
    pool_size: int = 1
) -> Dict:

    os.makedirs(output_dir, exist_ok=True)
    
    # Map tokens to types
    token_types = map_token_types(tokens, vocab, prompt_length)
    
    # 获取层数和输出token数量
    n_layers = layer_attention_weights.shape[0]
    n_output_tokens = output_attention_weights.shape[0] // n_layers
    
    # 计算完整的attention形状
    input_seq_len = layer_attention_weights.shape[3]
    full_seq_len = input_seq_len + n_output_tokens
    ctx_size = layer_attention_weights.shape[4]
    
    print(f"Processing {n_layers} layers with {n_output_tokens} output tokens")
    print(f"Input shape: {input_seq_len}, Full sequence length: {full_seq_len}, Context size: {ctx_size}")
    
    # Identify token boundaries
    boundaries = identify_token_boundaries(tokens, vocab, prompt_length)
    
    # Prepare results dictionary
    results = {}
    
    # Process each layer
    for layer_idx in range(n_layers):
        print(f"Processing layer {layer_idx+1}/{n_layers}...")
        
        # Extract layer attention from forward pass
        layer_attn = layer_attention_weights[layer_idx, 0]  # [n_heads, seq_len, seq_len]
        
        # Extract output attention for this layer
        output_attn_indices = [layer_idx + i * n_layers for i in range(n_output_tokens)]
        output_attn = output_attention_weights[output_attn_indices, 0]  # [n_output_tokens, n_heads, 1, seq_len]
        
        # Average the attention across heads
        avg_layer_attn = torch.mean(layer_attn, dim=0).float()  # [seq_len, seq_len]
        avg_output_attn = torch.mean(output_attn, dim=1).squeeze(1).float()  # [n_output_tokens, seq_len]
        
        # 创建完整的注意力矩阵
        full_attn = torch.zeros((full_seq_len, ctx_size), device=avg_layer_attn.device, dtype=avg_layer_attn.dtype)
        
        # 填充自注意力部分
        full_attn[:input_seq_len, :ctx_size] = avg_layer_attn
        
        # 填充输出token的注意力部分（只保留指向前面token的注意力）
        for i, output_row in enumerate(avg_output_attn):
            output_pos = input_seq_len + i
            full_attn[output_pos, :ctx_size] = output_row
        
        # 完整注意力矩阵存储到GPU
        avg_layer_attn = full_attn
        
        # Apply pooling if needed
        if pool_size > 1:
            # Pool the layer attention
            pooled_layer_attn = torch.nn.functional.avg_pool2d(
                avg_layer_attn.unsqueeze(0).unsqueeze(0), 
                pool_size, 
                stride=pool_size
            ).squeeze(0).squeeze(0)
            
            # Pool the output attention
            pooled_output_attn = torch.nn.functional.avg_pool2d(
                avg_output_attn.unsqueeze(1),
                (1, pool_size),
                stride=(1, pool_size)
            ).squeeze(1)
            
            avg_layer_attn = pooled_layer_attn
            avg_output_attn = pooled_output_attn
        
        # Create visualizations
        # 1. Layer self-attention heatmap
        plt.figure(figsize=(12, 10), dpi=300)
        
        # Log normalization for better visualization
        vmin = max(0.0001, torch.min(avg_layer_attn[avg_layer_attn > 0]).item())
        vmax = torch.max(avg_layer_attn).item()
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        ax = sns.heatmap(avg_layer_attn.cpu().numpy(), cmap="viridis", norm=log_norm,
                     cbar_kws={'label': 'Attention score'})
        
        # Add boundary lines for different token types
        colors = {'text': 'blue', 'image': 'red', 'output': 'green', 'image_boundary': 'purple'}
        
        for token_type, boundary in boundaries.items():
            if token_type not in ['prompt', 'special'] and boundary['start'] is not None and boundary['end'] is not None:
                start, end = boundary['start'], boundary['end']
                if pool_size > 1:
                    start = start // pool_size
                    end = end // pool_size
                
                color = colors.get(token_type, 'gray')
                plt.axhline(y=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
                plt.axhline(y=end, color=color, linestyle='-', linewidth=2, alpha=0.7)
                plt.axvline(x=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
                plt.axvline(x=end, color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add labels
                plt.text(start + (end-start)//2, start - 0.5, token_type, 
                        fontsize=12, color='white', weight='bold',
                        bbox=dict(facecolor=color, alpha=0.8))
        
        plt.title(f"Layer {layer_idx+1} Self-Attention", fontsize=16)
        plt.xlabel("Key position", fontsize=14)
        plt.ylabel("Query position", fontsize=14)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_self_attention.png"), bbox_inches='tight')
        plt.close()
        
        # 2. Output tokens attention to all input tokens
        plt.figure(figsize=(12, 10), dpi=300)
        
        # Log normalization for better visualization
        vmin = max(0.0001, torch.min(avg_output_attn[avg_output_attn > 0]).item()) if torch.any(avg_output_attn > 0) else 0.0001
        vmax = torch.max(avg_output_attn).item()
        log_norm = LogNorm(vmin=vmin, vmax=vmax)
        
        ax = sns.heatmap(avg_output_attn.cpu().numpy(), cmap="viridis", norm=log_norm,
                     cbar_kws={'label': 'Attention score'})
        
        # Add vertical lines for token type boundaries
        for token_type, boundary in boundaries.items():
            if token_type not in ['prompt', 'output', 'special'] and boundary['start'] is not None and boundary['end'] is not None:
                start, end = boundary['start'], boundary['end']
                if pool_size > 1:
                    start = start // pool_size
                    end = end // pool_size
                
                color = colors.get(token_type, 'gray')
                plt.axvline(x=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
                plt.axvline(x=end, color=color, linestyle='-', linewidth=2, alpha=0.7)
                
                # Add labels
                plt.text(start + (end-start)//2, -0.5, token_type, 
                        fontsize=12, color='white', weight='bold',
                        bbox=dict(facecolor=color, alpha=0.8))
        
        plt.title(f"Layer {layer_idx+1} Output-to-Input Attention", fontsize=16)
        plt.xlabel("Input token position", fontsize=14)
        plt.ylabel("Output token index", fontsize=14)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"layer_{layer_idx}_output_attention.png"), bbox_inches='tight')
        plt.close()
        
        # 3. Analysis of cross-attention patterns
        # Compute average attention scores between different token types
        metrics = {}
        
        # For self-attention
        text_range = (boundaries['text']['start'], boundaries['text']['end']) if boundaries['text']['start'] is not None else (0, 0)
        image_range = (boundaries['image']['start'], boundaries['image']['end']) if boundaries['image']['start'] is not None else (0, 0)
        output_range = (boundaries['output']['start'], boundaries['output']['end'])
        
        # Define token ranges
        ranges = {
            'text': text_range,
            'image': image_range,
            'output': output_range
        }
        
        # Only include ranges that exist
        ranges = {k: v for k, v in ranges.items() if v[0] is not None and v[0] != v[1]}
        
        # Compute attention metrics between different token types
        for src_type, (src_start, src_end) in ranges.items():
            for tgt_type, (tgt_start, tgt_end) in ranges.items():
                if src_start >= src_end or tgt_start >= tgt_end:
                    continue
                
                # Adjust for pooling
                if pool_size > 1:
                    src_start = src_start // pool_size
                    src_end = src_end // pool_size
                    tgt_start = tgt_start // pool_size
                    tgt_end = tgt_end // pool_size
                
                # Extract the relevant attention region
                attention_region = avg_layer_attn[src_start:src_end, tgt_start:tgt_end]
                
                # Calculate metrics
                if attention_region.numel() > 0:
                    metrics[f"{src_type}_to_{tgt_type}"] = {
                        "mean": float(torch.mean(attention_region).item()),
                        "max": float(torch.max(attention_region).item()),
                        "min": float(torch.min(attention_region).item()),
                        "std": float(torch.std(attention_region).item())
                    }
        
        # For output-to-input attention
        for tgt_type, (tgt_start, tgt_end) in ranges.items():
            if tgt_type != 'output' and tgt_start < tgt_end:
                # Adjust for pooling
                if pool_size > 1:
                    tgt_start = tgt_start // pool_size
                    tgt_end = tgt_end // pool_size
                
                # Extract attention from output tokens to this token type
                attention_region = avg_output_attn[:, tgt_start:tgt_end]
                
                if attention_region.numel() > 0:
                    metrics[f"output_to_{tgt_type}"] = {
                        "mean": float(torch.mean(attention_region).item()),
                        "max": float(torch.max(attention_region).item()),
                        "min": float(torch.min(attention_region).item()),
                        "std": float(torch.std(attention_region).item())
                    }
        
        # Store results for this layer
        results[f"layer_{layer_idx}"] = {
            "metrics": metrics
        }
        
        create_combined_attention_plot(
            layer_idx=layer_idx,
            avg_layer_attn=avg_layer_attn,
            avg_output_attn=avg_output_attn,
            boundaries=boundaries,
            pool_size=pool_size,
            output_dir=output_dir
        )
    
    # Create cross-attention trend plots
    plot_cross_attention_trends(results, output_dir)
    
    # Save all metrics to a JSON file
    metrics_path = os.path.join(output_dir, "combined_attention_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def create_combined_attention_plot(
    layer_idx: int,
    avg_layer_attn: torch.Tensor,  # 完整的自注意力矩阵，已合并输出token的注意力
    avg_output_attn: torch.Tensor,  # output-attention [n_output_tokens, seq_len]，仅用于参考
    boundaries: Dict[str, Dict[str, int]],
    pool_size: int,
    output_dir: str
):
    plt.figure(figsize=(14, 12), dpi=300)
    
    prompt_length = boundaries['prompt']['end']
    if pool_size > 1:
        prompt_length = prompt_length // pool_size
    
    # 直接使用已合并的注意力矩阵
    combined_attn = avg_layer_attn.cpu().numpy()
    
    # 设置对数归一化以增强可视化效果
    vmin = max(0.0001, np.min(combined_attn[combined_attn > 0]))
    vmax = np.max(combined_attn)
    log_norm = LogNorm(vmin=vmin, vmax=vmax)
    
    ax = sns.heatmap(combined_attn, cmap="viridis", norm=log_norm,
                 cbar_kws={'label': 'Attention score'})
    
    colors = {'text': 'blue', 'image': 'red', 'output': 'green', 'image_boundary': 'purple'}
    
    for token_type, boundary in boundaries.items():
        if token_type not in ['prompt', 'special'] and boundary['start'] is not None and boundary['end'] is not None:
            start, end = boundary['start'], boundary['end']
            if pool_size > 1:
                start = start // pool_size
                end = end // pool_size
            
            color = colors.get(token_type, 'gray')
            plt.axhline(y=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axhline(y=end, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axvline(x=start, color=color, linestyle='-', linewidth=2, alpha=0.7)
            plt.axvline(x=end, color=color, linestyle='-', linewidth=2, alpha=0.7)
            
            plt.text(start + (end-start)//2, start - 0.5, token_type, 
                    fontsize=12, color='white', weight='bold',
                    bbox=dict(facecolor=color, alpha=0.8))
    
    # 添加输入/输出分界线
    input_seq_len = boundaries['prompt']['end']
    if pool_size > 1:
        input_seq_len = input_seq_len // pool_size
        
    plt.axhline(y=input_seq_len, color='white', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(x=input_seq_len, color='white', linestyle='--', linewidth=2, alpha=0.8)
    
    # 修改标题和标签
    plt.title(f"Layer {layer_idx+1} Combined Attention", fontsize=16)
    plt.xlabel("Key position", fontsize=14)
    plt.ylabel("Query position", fontsize=14)
    
    # 添加输出→输入区域标签
    plt.text(input_seq_len // 2, input_seq_len + (combined_attn.shape[0] - input_seq_len) // 2, 
             "Output → Input\nAttention", 
             fontsize=12, color='white', ha='center', va='center', weight='bold',
             bbox=dict(facecolor='green', alpha=0.8))
    
    # 添加"已生成输出"区域标签
    plt.text(input_seq_len + (combined_attn.shape[1] - input_seq_len) // 2, input_seq_len // 2,
             "Generated\nOutput", 
             fontsize=12, color='white', ha='center', va='center', weight='bold',
             bbox=dict(facecolor='purple', alpha=0.8))
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"layer_{layer_idx}_combined_attention.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def run_combined_analysis(
    model_output_attention: torch.Tensor,  # From first forward pass
    autoregressive_attention: torch.Tensor,  # From autoregressive generation
    tokens: List[int],
    vocab,
    prompt_length: int,
    output_dir: str = "combined_attention_analysis",
    pool_size: int = 20  # Match example.py default
):
    print(f"Starting combined attention analysis...")
    print(f"Model attention shape: {model_output_attention.shape}")
    print(f"Autoregressive attention shape: {autoregressive_attention.shape}")
    print(f"Sequence length: {len(tokens)}")
    print(f"Prompt length: {prompt_length}")
    
    # Run analysis
    results = visualize_combined_attention(
        layer_attention_weights=model_output_attention,
        output_attention_weights=autoregressive_attention,
        tokens=tokens,
        vocab=vocab,
        prompt_length=prompt_length,
        output_dir=output_dir,
        pool_size=pool_size
    )
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return results

