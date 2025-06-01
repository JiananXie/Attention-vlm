import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

# Dictionary mapping vision encoder size to models
dict_encoder = {
    0.1: ["TinyGPT-V", "llava-fastvithd_0.5b_stage3"],
    0.3: ["mobilevlm_v2_7b","llava-phi2", "MiniGemini", "MoE-LLaVA","llava-v1.5-vicuna-7b","llava-v1.6-mistral-7b","llava-v1.6-vicuna-7b"],
    0.4: [
        "TinyLLaVA",
        "Deepseek-VL-7B",
        "bunny-phi2-siglip",
        "bunny-phi1.5-eva",
        "bunny-phi1.5-siglip",
        "bunny-phi2-eva",
        "bunny-stablelm2-eva",
        "bunny-stablelm2-siglip",
        "idefics2",
    ],
    0.6: ["qwen2-VL-2B-Instruct", "qwen2-VL-7B-Instruct", "qwen2.5-VL-3B-Instruct", "qwen2.5-VL-7B-Instruct"],
    0.7: ["cobra"]
}

# Dictionary mapping LLM type to models
dict_llm = {
    "phi2": [
        "llava-phi2",
        "TinyLLaVA",
        "MoE-LLaVA",
        "TinyGPT-V",
        "bunny-phi2-siglip",
        "bunny-phi2-eva",
    ],
    "phi1.5": [
        "bunny-phi1.5-eva",
        "bunny-phi1.5-siglip",
    ],
    "stablelm2": [
        "bunny-stablelm2-eva",
        "bunny-stablelm2-siglip",
    ],
    "qwen2": [
        "llava-fastvithd_0.5b_stage3",
        "qwen2-VL-2B-Instruct",
        "qwen2-VL-7B-Instruct",
    ],
    "qwen2.5": [
        "qwen2.5-VL-3B-Instruct",
        "qwen2.5-VL-7B-Instruct",
    ],
    "mistral":[
        "idefics2",
        "llava-v1.6-mistral-7b",
    ],
    "vicuna":[
        "mobilevlm_v2_7b",
        "llava-v1.5-vicuna-7b",
        "llava-v1.6-vicuna-7b",
    ]
}

def load_attention_data(csv_file):
    """
    Load attention data from CSV file
    
    Args:
        csv_file: Path to the CSV file containing attention data
        
    Returns:
        DataFrame with attention data
    """
    df = pd.read_csv(csv_file)
    # Filter out models without pie charts
    df = df[df['has_pie'] == True]
    return df

def get_models_with_data(df, model_list):
    """
    Filter model list to only include models that have attention data
    
    Args:
        df: DataFrame with attention data
        model_list: List of model names to check
        
    Returns:
        List of models that have data in the DataFrame
    """
    available_models = df['model'].unique()
    return [model for model in model_list if model in available_models]

def add_model_labels_to_plot(ax, x_data, y_data, model_name, color, offset_idx=None):
    """
    Add a small model name label to a line at an appropriate position
    
    Args:
        ax: Matplotlib axis
        x_data: X-coordinates of the line
        y_data: Y-coordinates of the line
        model_name: Name of the model to display
        color: Color of the text
        offset_idx: Optional index to place label (if None, will choose position)
    """
    # Choose a position at about 70% along the line if not specified
    if offset_idx is None:
        idx = int(len(x_data) * 0.7)
    else:
        idx = offset_idx
    
    # Make sure idx is within bounds
    idx = min(idx, len(x_data) - 1)
    
    # Get coordinates for the label
    x = x_data[idx]
    y = y_data[idx]
    
    # Add a small offset to prevent label from overlapping the line
    y_offset = 0.02 * (max(y_data) - min(y_data))
    
    # Add the label with a small font size
    ax.annotate(
        model_name, 
        (x, y + y_offset),
        fontsize=7, 
        color=color, 
        ha='center', 
        va='bottom',
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7)
    )

def plot_by_encoder_size(df):
    """
    Plot attention values by vision encoder size with separate plots for each token type
    to ensure all models are displayed properly.
    
    Args:
        df: DataFrame with attention data
    """
    # Define token types and their visualization styles
    token_types = ['img_attention', 'inst_attention', 'sys_attention', 'out_attention']
    token_names = ['Image', 'Instruction', 'System', 'Output']
    
    # Get all available models from the dataframe
    all_available_models = sorted(df['model'].unique())
    print(f"Found {len(all_available_models)} models with data: {', '.join(all_available_models)}")
    
    # Map each model to its encoder size
    model_to_encoder = {}
    for encoder_size, models in dict_encoder.items():
        for model in models:
            model_to_encoder[model] = encoder_size
    
    # Group models by type to assign consistent colors
    model_groups = {
        'bunny': [m for m in all_available_models if 'bunny' in m],
        'llava': [m for m in all_available_models if 'llava' in m],
        'qwen': [m for m in all_available_models if 'qwen' in m],
        'tiny': [m for m in all_available_models if 'Tiny' in m],
        'other': [m for m in all_available_models if not any(x in m for x in ['bunny', 'llava', 'qwen', 'Tiny'])]
    }
    
    # Create color mapping for model groups
    group_colors = {
        'bunny': plt.cm.Reds(np.linspace(0.3, 0.9, len(model_groups['bunny']) or 1)),
        'llava': plt.cm.Blues(np.linspace(0.3, 0.9, len(model_groups['llava']) or 1)),
        'qwen': plt.cm.Greens(np.linspace(0.3, 0.9, len(model_groups['qwen']) or 1)),
        'tiny': plt.cm.Purples(np.linspace(0.3, 0.9, len(model_groups['tiny']) or 1)),
        'other': plt.cm.Greys(np.linspace(0.3, 0.9, len(model_groups['other']) or 1))
    }
    
    # Assign colors to each model
    model_colors = {}
    for group, models in model_groups.items():
        for i, model in enumerate(models):
            model_colors[model] = group_colors[group][min(i, len(group_colors[group])-1)]
    
    # Define a wide variety of line styles to distinguish models
    base_linestyles = ['-', '--', '-.', ':']
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    # Create extended linestyles with varying dash patterns
    extended_linestyles = base_linestyles.copy()
    extended_linestyles.extend([(0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1))])
    
    # For each token type, create a separate plot
    for token_idx, (token_type, token_name) in enumerate(zip(token_types, token_names)):
        # Use log scale for Image token type
        use_log_scale = token_type == 'img_attention'
        
        # Set up figure with a modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # For Image attention, create two plots - one with log scale and one with regular scale
        plot_configs = [(False, '')] if not use_log_scale else [(True, '_log'), (False, '_linear')]
        
        for log_scale, suffix in plot_configs:
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Keep track of legend entries
            legend_handles = []
            
            # Track y-values for jittering
            y_value_sets = {}
            
            # For each available model
            for model_idx, model in enumerate(all_available_models):
                # Get data for this model
                model_df = df[df['model'] == model]
                
                # Skip if no data
                if model_df.empty:
                    continue
                    
                # Get encoder size for this model (if known)
                encoder_size = model_to_encoder.get(model, 0.0)  # Default to 0.0 if unknown
                
                # Choose marker based on model index
                marker = markers[model_idx % len(markers)]
                
                # Choose color based on model group
                color = model_colors.get(model, (0.5, 0.5, 0.5, 1.0))  # Default gray if not found
                
                # Choose line style based on combination of encoder size and model index
                linestyle_idx = (hash(model) % len(extended_linestyles))
                linestyle = extended_linestyles[linestyle_idx]
                
                # Extract layer values for this token type
                max_layer = model_df['layer'].max()
                layer_values = []
                
                for layer in range(1, max_layer + 1):
                    layer_df = model_df[model_df['layer'] == layer]
                    if not layer_df.empty and token_type in layer_df.columns and not pd.isna(layer_df[token_type].iloc[0]):
                        layer_values.append(layer_df[token_type].iloc[0])
                    else:
                        layer_values.append(np.nan)
                
                # Skip if no valid data for this token type
                if not any(not pd.isna(x) for x in layer_values):
                    continue
                
                # Get non-NaN indices for jittering calculation
                valid_indices = [i for i, x in enumerate(layer_values) if not pd.isna(x)]
                
                # For Image token type with small values, add small offsets to prevent complete overlap
                if token_type == 'img_attention':
                    # Group values by their magnitude for jittering
                    for i in valid_indices:
                        val = layer_values[i]
                        # Create buckets for similar values
                        bucket = round(val * 100) / 100
                        if bucket not in y_value_sets:
                            y_value_sets[bucket] = []
                        y_value_sets[bucket].append((model, i))
                
                # Plot the data
                x_values = list(range(1, len(layer_values) + 1))
                
                # Apply small jitter for image token type with close values
                if token_type == 'img_attention':
                    jittered_values = layer_values.copy()
                    for i in valid_indices:
                        val = layer_values[i]
                        bucket = round(val * 100) / 100
                        # Find position of this model in the bucket
                        bucket_models = y_value_sets[bucket]
                        position = next((idx for idx, (m, _) in enumerate(bucket_models) if m == model), 0)
                        # Add proportional jitter based on position
                        jitter = 0.002 * (position + 1) / (len(bucket_models) + 1)
                        jittered_values[i] = val + jitter
                    
                    # Use jittered values
                    plot_values = jittered_values
                else:
                    plot_values = layer_values
                
                # Plot the line with markers
                line = ax.plot(
                    x_values, 
                    plot_values,
                    label=f"{model} ({encoder_size}B)",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    alpha=0.9,
                    marker=marker,
                    markersize=5,
                    markevery=max(1, len(plot_values)//8),
                    markeredgecolor='black',
                    markeredgewidth=0.5
                )
                
                # Add to legend handles
                legend_handles.append(line[0])
                
                # Add model label directly on the line at a staggered position
                stagger_factor = model_idx % 5  # Stagger labels to avoid overlap
                label_position = int(len(x_values) * (0.5 + 0.08 * stagger_factor))
                label_position = min(label_position, len(x_values)-1)
                
                if label_position >= 0:
                    # Determine text color based on background
                    text_color = 'black' if np.mean(color[:3]) > 0.5 else 'white'
                    
                    # Add bbox for better visibility
                    ax.annotate(
                        model, 
                        (x_values[label_position], plot_values[label_position]),
                        fontsize=7, 
                        color=text_color, 
                        ha='center', 
                        va='bottom',
                        bbox=dict(
                            boxstyle="round,pad=0.1", 
                            fc=color, 
                            ec="black", 
                            alpha=0.9,
                            linewidth=0.5
                        ),
                        zorder=100  # Ensure labels are on top
                    )
            
            # Skip this plot if nothing was plotted
            if not legend_handles:
                plt.close(fig)
                continue
                
            # Enhance plot aesthetics
            ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{token_name} Attention Proportion', fontsize=14, fontweight='bold')
            
            scale_type = "Log Scale" if log_scale else "Linear Scale"
            ax.set_title(f'{token_name} Attention Distribution by Model ({scale_type})', fontsize=16, fontweight='bold')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            
            # Set y-axis scale based on token type and configuration
            if log_scale:
                ax.set_yscale('log')
                # Ensure the bottom limit is not zero for log scale
                ax.set_ylim(0.001, min(1.0, ax.get_ylim()[1] * 1.1))
            else:
                min_y = 0
                if token_type == 'img_attention':
                    # For image token with linear scale, zoom in to see details
                    max_visible = max([max([v for v in plot_values if not pd.isna(v)]) 
                                     for plot_values in [layer_values] if any(not pd.isna(v) for v in plot_values)])
                    # Add some headroom
                    ax.set_ylim(min_y, min(0.15, max_visible * 1.5))
                else:
                    # For other token types, use dynamic scaling
                    ax.set_ylim(min_y, min(1.0, ax.get_ylim()[1] * 1.1))
                    
            # Add minor gridlines for better readability
            ax.grid(True, which='major', linestyle='-', alpha=0.6)
            ax.grid(True, which='minor', linestyle=':', alpha=0.3)
            
            # Add a light gray background for better contrast
            ax.set_facecolor('#f8f8f8')
            
            # Customize the tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Add the legend
            # Sort by model group, then by encoder size
            by_label = dict(zip([h.get_label() for h in legend_handles], legend_handles))
            
            # Helper function to get model group
            def get_model_group(label):
                model = label.split(' (')[0]
                for group, models in model_groups.items():
                    if model in models:
                        return group
                return 'other'
            
            # Sort labels by group, then by encoder size, then by model name
            sorted_labels = sorted(by_label.keys(), 
                                  key=lambda x: (
                                      get_model_group(x),
                                      float(x.split('(')[1].replace('B)', '')), 
                                      x.split(' (')[0]
                                  ))
            
            legend_handles_sorted = [by_label[label] for label in sorted_labels]
            
            # Add the legend with sorted labels in multiple columns for better readability
            legend = ax.legend(
                handles=legend_handles_sorted,
                labels=sorted_labels,
                fontsize=7,
                frameon=True,
                facecolor='white',
                framealpha=0.9,
                edgecolor='gray',
                loc='upper center',  # Place at top
                bbox_to_anchor=(0.5, -0.05),  # Position below the plot
                ncol=min(4, len(legend_handles_sorted)),  # Multiple columns 
                title='Model (Encoder Size)',
                title_fontsize=9
            )
            
            # Add a light border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1.0)
            
            # Save figure with higher resolution
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)  # Make room for the legend
            plt.savefig(os.path.join('images', f'{token_name.lower()}_by_model{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def plot_by_llm_type(df):
    """
    Plot attention values for each LLM type, with a separate plot for each LLM type
    showing individual lines for each model within that LLM type
    
    Args:
        df: DataFrame with attention data
    """
    # Define token types and their visualization styles
    token_types = ['img_attention', 'inst_attention', 'sys_attention', 'out_attention']
    token_names = ['Image', 'Instruction', 'System', 'Output']
    token_colors = ['green', 'blue', 'red', 'purple']
    token_markers = ['o', '^', 's', 'D']  # circle, triangle, square, diamond
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for llm_type, models in dict_llm.items():
        # Get models that have data
        valid_models = get_models_with_data(df, models)
        
        if not valid_models:
            continue
            
        # Create a figure for this LLM type
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define line styles for different models
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
        
        # Create a color palette for different models
        model_cmap = plt.cm.tab10(np.linspace(0, 1, len(valid_models)))
        model_colors = {model: model_cmap[i] for i, model in enumerate(valid_models)}
        
        # Store legend handles
        legend_handles = []
        
        # For each token type
        for token_idx, (token_type, token_name, token_color, token_marker) in enumerate(
            zip(token_types, token_names, token_colors, token_markers)):
            
            # For each model in this LLM type
            for model_idx, model in enumerate(valid_models):
                model_df = df[df['model'] == model]
                
                if not model_df.empty:
                    # Get data for this model
                    max_layer = model_df['layer'].max()
                    layer_values = []
                    
                    for layer in range(1, max_layer + 1):
                        layer_df = model_df[model_df['layer'] == layer]
                        if (not layer_df.empty and token_type in layer_df.columns 
                            and not layer_df[token_type].isna().all()):
                            layer_values.append(layer_df[token_type].iloc[0])
                        else:
                            layer_values.append(np.nan)
                    
                    if layer_values:
                        # Define x values for the plot
                        x_values = list(range(1, len(layer_values) + 1))
                        
                        # Choose a line style that depends on both token type and model
                        linestyle = linestyles[model_idx % len(linestyles)]
                        
                        # Plot individual line for this model and token type
                        line = ax.plot(
                            x_values, 
                            layer_values,
                            label=f"{token_name} ({model})",
                            color=token_color,
                            linestyle=linestyle,
                            linewidth=2.0,
                            alpha=0.8,
                            marker=token_marker,
                            markersize=6,
                            markevery=max(1, len(layer_values)//10),
                            markeredgecolor='black',
                            markeredgewidth=0.8
                        )
                        
                        # Add the line to legend handles
                        legend_handles.append(line[0])
                        
                        # Add model label on the line
                        label_position = int(len(x_values) * (0.3 + 0.1 * token_idx + 0.05 * model_idx))
                        if label_position < len(x_values):
                            add_model_labels_to_plot(
                                ax, x_values, layer_values, model, 
                                token_color, offset_idx=label_position
                            )
        
        # Create a sorted legend for better readability
        by_label = dict(zip([h.get_label() for h in legend_handles], legend_handles))
        
        # Sort by token type first, then by model name
        token_names_ordered = ['Image', 'Instruction', 'System', 'Output']
        sorted_labels = []
        for token in token_names_ordered:
            token_labels = [label for label in by_label.keys() if label.startswith(token)]
            token_labels.sort()  # Sort by model name
            sorted_labels.extend(token_labels)
        
        # Add the legend with sorted labels
        legend_handles_sorted = [by_label[label] for label in sorted_labels]
        
        # Enhance the plot aesthetics
        ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
        ax.set_ylabel('Attention Proportion', fontsize=14, fontweight='bold')
        ax.set_title(f'Attention Distribution for {llm_type.upper()} Models', fontsize=16, fontweight='bold')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # Set y-axis limits to ensure all data is visible
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.1))
        
        # Improve grid appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a light gray background for better contrast
        ax.set_facecolor('#f8f8f8')
        
        # Customize the tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add an enhanced legend with adjusted position
        ax.legend(
            handles=legend_handles_sorted,
            labels=sorted_labels,
            fontsize=8,  # Smaller font due to more entries
            frameon=True,
            facecolor='white',
            framealpha=0.9,
            edgecolor='gray',
            loc='center right',
            ncol=2,  # Use two columns for better space usage
            title='Token Type (Model)',
            title_fontsize=10
        )
        
        # Add a light border around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1.0)
        
        # Save figure with higher resolution
        plt.tight_layout()
        plt.savefig(os.path.join('images', f'attention_by_{llm_type}_llm.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compare_same_llm_different_encoder(df):
    """
    Compare models with the same LLM but different vision encoder sizes
    
    Args:
        df: DataFrame with attention data
    """
    # Define token types and their visualization styles
    token_types = ['img_attention', 'inst_attention', 'sys_attention', 'out_attention']
    token_names = ['Image', 'Instruction', 'System', 'Output']
    token_colors = ['green', 'blue', 'red', 'purple']
    token_markers = ['o', '^', 's', 'D']  # circle, triangle, square, diamond
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Find pairs of models with same LLM but different encoder sizes
    comparisons = []
    
    # Create a mapping of models to their encoder sizes
    model_to_encoder = {}
    for encoder_size, models in dict_encoder.items():
        for model in models:
            model_to_encoder[model] = encoder_size
    
    # For each LLM type
    for llm_type, models in dict_llm.items():
        # Get valid models for this LLM
        valid_models = get_models_with_data(df, models)
        
        # If we have at least 2 models with the same LLM, we can compare
        if len(valid_models) >= 2:
            # Group models by encoder size
            encoder_models = {}
            for model in valid_models:
                if model in model_to_encoder:
                    encoder_size = model_to_encoder[model]
                    if encoder_size not in encoder_models:
                        encoder_models[encoder_size] = []
                    encoder_models[encoder_size].append(model)
            
            # If we have models with different encoder sizes
            if len(encoder_models) >= 2:
                comparisons.append((llm_type, encoder_models))
    
    # Plot comparisons for each token type
    for token_type, token_name, token_color, token_marker in zip(token_types, token_names, token_colors, token_markers):
        for llm_type, encoder_models in comparisons:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            encoder_sizes = sorted(encoder_models.keys())
            # Create a colormap for encoder sizes
            encoder_cmap = plt.cm.viridis(np.linspace(0, 0.9, len(encoder_sizes)))
            
            # Define line styles for different encoder sizes
            linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
            
            for i, encoder_size in enumerate(encoder_sizes):
                models = encoder_models[encoder_size]
                # Calculate average token values across models with this encoder size
                encoder_data = []
                model_names = []
                
                for model in models:
                    model_df = df[df['model'] == model]
                    if not model_df.empty:
                        # Create list of layer values, filling missing layers with NaN
                        max_layer = model_df['layer'].max()
                        layer_values = []
                        
                        for layer in range(1, max_layer + 1):
                            layer_df = model_df[model_df['layer'] == layer]
                            if (not layer_df.empty and token_type in layer_df.columns 
                                and not layer_df[token_type].isna().all()):
                                layer_values.append(layer_df[token_type].iloc[0])
                            else:
                                layer_values.append(np.nan)
                        
                        encoder_data.append(layer_values)
                        model_names.append(model)
                
                if encoder_data:
                    # Find the maximum length across all model data
                    max_len = max(len(data) for data in encoder_data)
                    
                    # Pad shorter arrays with NaN
                    encoder_data = [data + [np.nan] * (max_len - len(data)) for data in encoder_data]
                    
                    # Calculate mean, ignoring NaN values
                    mean_values = np.nanmean(encoder_data, axis=0)
                    
                    # Define x values for the plot
                    x_values = list(range(1, len(mean_values) + 1))
                    
                    # Choose line style based on encoder size index
                    linestyle = linestyles[i % len(linestyles)]
                    
                    # Plot the line with markers
                    line = ax.plot(
                        x_values, 
                        mean_values,
                        label=f"{encoder_size}B",
                        color=encoder_cmap[i],
                        linestyle=linestyle,
                        linewidth=2.5,
                        alpha=0.8,
                        marker=token_marker,
                        markersize=8,
                        markevery=max(1, len(mean_values)//10),
                        markeredgecolor='black',
                        markeredgewidth=1.0
                    )
                    
                    # Add model labels if there are 1-3 models
                    if 1 <= len(model_names) <= 3:
                        # Use different positions for different encoder sizes to avoid overlap
                        position_offset = i * 3  # Offset by encoder size index
                        position = int(len(x_values) * 0.5 + position_offset)
                        
                        # Add small label with model names
                        model_label = ", ".join(model_names)
                        add_model_labels_to_plot(
                            ax, x_values, mean_values, model_label, encoder_cmap[i], 
                            offset_idx=min(position, len(x_values)-1)
                        )
            
            # Enhance the plot aesthetics
            ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'Attention Proportion', fontsize=14, fontweight='bold')
            ax.set_title(f'{token_name} Attention for {llm_type.upper()} with Different Vision Encoders', 
                        fontsize=16, fontweight='bold')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # Set y-axis limits to ensure all data is visible
            ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.1))
            
            # Improve grid appearance
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add a light gray background for better contrast
            ax.set_facecolor('#f8f8f8')
            
            # Customize the tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Add an enhanced legend with adjusted position
            ax.legend(
                title='Vision Encoder Size',
                fontsize=12,
                frameon=True,
                facecolor='white',
                framealpha=0.9,
                edgecolor='gray', 
                title_fontsize=14,
                loc='center right'  # Move from 'best' to 'center right'
            )
            
            # Add a light border around the plot
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
                spine.set_linewidth(1.0)
            
            # Save figure with higher resolution
            plt.tight_layout()
            plt.savefig(os.path.join('images', f'{token_name.lower()}_{llm_type}_by_encoder.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def main():
    # Load attention data
    csv_file = 'attention_analysis.csv'
    
    try:
        attention_df = load_attention_data(csv_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure to run collect_attention.py first to generate the CSV file.")
        return
    
    # Ensure the analysis and images directories exist
    os.makedirs('analysis', exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Plot attention by vision encoder size
    print("Plotting attention by vision encoder size...")
    plot_by_encoder_size(attention_df)
    
    # Plot attention by LLM type
    print("Plotting attention by LLM type...")
    plot_by_llm_type(attention_df)
    
    # Compare models with same LLM but different encoder sizes
    print("Comparing models with same LLM but different encoder sizes...")
    compare_same_llm_different_encoder(attention_df)
    
    print("Plotting complete. Images saved to the 'images' directory.")

if __name__ == "__main__":
    main() 