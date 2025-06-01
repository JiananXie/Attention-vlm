import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import seaborn as sns
from matplotlib.colors import LogNorm
import argparse
from pathlib import Path
import os
import json
from tqdm import tqdm

# 导入Cobra相关模块
from cobra import load

class SSMStateTracker:
    """跟踪Mamba SSM内部状态的类"""
    
    def __init__(self):
        self.state_matrices = {}
        self.handles = []
        self.input_maps = {}
        self.output_maps = {}
        
    def register_hooks(self, model):
        """向Mamba模型的各层注册钩子"""
        # 寻找SSM的核心计算模块
        for name, module in model.llm_backbone.model.named_modules():
            if "ssm" in name.lower() and hasattr(module, "forward"):
                # 为SSM模块注册钩子
                self.handles.append(
                    module.register_forward_hook(self._get_ssm_hook(name))
                )
                
                # 尝试找到适用于selective_scan_fn的模块
                if hasattr(module, "D"):
                    self.handles.append(
                        module.register_forward_pre_hook(self._get_state_pre_hook(name))
                    )
        
        return self
    
    def _get_ssm_hook(self, name):
        """创建SSM模块的钩子函数"""
        def hook(module, inputs, outputs):
            self.output_maps[name] = outputs
            if len(inputs) > 0:
                self.input_maps[name] = inputs[0]
        return hook
    
    def _get_state_pre_hook(self, name):
        """创建用于捕获状态矩阵的钩子"""
        def hook(module, inputs):
            # 存储层名称和模块，以便后续分析
            if not hasattr(self, "ssm_modules"):
                self.ssm_modules = {}
            self.ssm_modules[name] = module
        return hook
    
    def analyze_states(self):
        """分析并构建状态交互矩阵"""
        if not hasattr(self, "ssm_modules"):
            return {}
            
        # 对每个SSM模块生成状态矩阵
        for name, module in self.ssm_modules.items():
            if name not in self.input_maps:
                continue
                
            input_x = self.input_maps[name]
            
            # 从SSM模块获取参数
            if hasattr(module, "A_log") and hasattr(module, "D"):
                bsz, seqlen, dim = input_x.shape
                dstate = module.d_state
                
                # 获取SSM参数
                A = -torch.exp(module.A_log.float())  # (dim, dstate)
                D = module.D.float()  # (dim,)
                
                # 构建状态交互矩阵 - 模拟状态如何随时间演变
                state_matrix = torch.zeros((seqlen, seqlen), device=input_x.device)
                
                # 对于每个位置，计算其对后续位置的影响
                for i in range(seqlen):
                    for j in range(i, seqlen):
                        # 计算状态传递影响 (简化版)
                        delta = j - i
                        influence = torch.exp(A * delta).mean()
                        state_matrix[i, j] = influence
                        # 使矩阵对称
                        state_matrix[j, i] = influence
                
                # 将多头状态合并
                self.state_matrices[name] = state_matrix.cpu()
        
        return self.state_matrices
    
    def remove_hooks(self):
        """移除所有注册的钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def visualize_attention(matrix, output_path="state_map.png", title="Layer State"):
    """可视化状态交互矩阵，模仿注意力图"""
    # 确保输入是PyTorch张量
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.float().cpu().numpy()
    
    # 对矩阵进行归一化处理
    if matrix.shape[0] > 100:
        # 使用平均池化减少尺寸
        pooled_size = min(100, matrix.shape[0])
        step = matrix.shape[0] // pooled_size
        pooled_matrix = np.zeros((pooled_size, pooled_size))
        
        for i in range(pooled_size):
            for j in range(pooled_size):
                i_start, j_start = i*step, j*step
                i_end, j_end = min(matrix.shape[0], (i+1)*step), min(matrix.shape[1], (j+1)*step)
                pooled_matrix[i, j] = matrix[i_start:i_end, j_start:j_end].mean()
        
        matrix = pooled_matrix
    
    # 创建绘图
    plt.figure(figsize=(5, 5), dpi=400)
    
    # 使用LogNorm进行归一化
    min_val = max(0.0001, matrix.min())
    log_norm = LogNorm(vmin=min_val, vmax=matrix.max())
    
    # 创建热图
    ax = sns.heatmap(matrix, 
                     cmap="viridis", 
                     norm=log_norm)
    
    # 设置刻度标签
    x_ticks = [str(i*20) for i in range(0, matrix.shape[0], 20)]
    y_ticks = [str(i*20) for i in range(0, matrix.shape[0], 20)]
    tick_positions = [i for i in range(0, matrix.shape[0], 20)]
    
    if len(tick_positions) > 0:
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(x_ticks, fontsize=3, rotation=90)
        ax.set_yticklabels(y_ticks, fontsize=3, rotation=0)
    
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # 找出矩阵中每行的前10个最高值
    top_interactions = []
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx]
        # 获取前10个最大值的索引和值
        top_indices = np.argsort(row)[-10:][::-1]
        top_values = row[top_indices]
        top_interactions.append(list(zip(top_indices.tolist(), top_values.tolist())))
    
    return top_interactions, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', type=str, default="cobra+3b", help='Cobra模型ID')
    parser.add_argument('--image-path', type=str, default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png", help='图像路径')
    parser.add_argument('--prompt', type=str, default="What is going on in this image", help='提示文本')
    parser.add_argument('--output-path', type=str, default="output", help='输出路径')
    parser.add_argument('--hf-token', type=str, default="", help='HuggingFace token')
    args = parser.parse_args()
    
    # 设置设备和精度
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 加载模型
    print(f"Loading model {args.model_id}...")
    vlm = load(args.model_id, hf_token=args.hf_token)
    vlm.to(device, dtype=dtype)
    
    # 加载图像
    print(f"Loading image from {args.image_path}...")
    if args.image_path.startswith('http'):
        image = Image.open(requests.get(args.image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(args.image_path).convert("RGB")
    
    # 构建提示
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=args.prompt)
    prompt_text = prompt_builder.get_prompt()
    
    # 创建状态跟踪器
    tracker = SSMStateTracker().register_hooks(vlm)
    
    # 创建输出目录
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    attn_maps_dir = output_path / "attn_maps"
    attn_maps_dir.mkdir(exist_ok=True)
    
    # 生成文本并捕获内部状态
    print("Generating text and capturing internal states...")
    with torch.inference_mode():
        generated_text = vlm.generate(
            image,
            prompt_text,
            use_cache=True,
            do_sample=True,
            temperature=0.4,
            max_new_tokens=512
        )
    
    # 分析状态并构建状态交互矩阵
    print("Analyzing state matrices...")
    state_matrices = tracker.analyze_states()
    
    # 可视化状态交互矩阵
    print("Visualizing state interaction matrices...")
    results = {}
    
    for i, (name, matrix) in enumerate(tqdm(state_matrices.items())):
        layer_idx = name.split(".")[-3]  # 提取层索引
        attn_map_path = attn_maps_dir / f"atten_map_{layer_idx}.png"
        
        # 可视化状态矩阵
        top_interactions, _ = visualize_attention(
            matrix,
            output_path=str(attn_map_path),
            title=f"Layer {layer_idx}"
        )
        
        # 存储结果
        results[name] = {
            "layer_idx": layer_idx,
            "shape": list(matrix.shape),
            "top_interactions": top_interactions[:10]  # 只保存前10行的交互信息
        }
    
    # 如果没有成功构建状态矩阵，创建一个合成的示例
    if len(state_matrices) == 0:
        print("Warning: No state matrices were generated. Creating synthetic examples...")
        num_layers = 24  # 假设模型有24层
        
        for i in range(num_layers):
            # 创建一个随机的状态矩阵
            seq_len = 200
            synthetic_matrix = np.zeros((seq_len, seq_len))
            
            # 填充主对角线附近的元素，模拟局部注意力
            for j in range(seq_len):
                for k in range(seq_len):
                    dist = abs(j - k)
                    synthetic_matrix[j, k] = np.exp(-dist / 20) * (1 + 0.2 * np.random.randn())
            
            # 添加一些长距离依赖
            for _ in range(20):
                j, k = np.random.randint(0, seq_len, 2)
                synthetic_matrix[j, k] = 0.8 + 0.2 * np.random.random()
                synthetic_matrix[k, j] = synthetic_matrix[j, k]
            
            attn_map_path = attn_maps_dir / f"atten_map_{i}.png"
            
            # 可视化合成矩阵
            top_interactions, _ = visualize_attention(
                synthetic_matrix,
                output_path=str(attn_map_path),
                title=f"Layer {i} (Synthetic)"
            )
            
            # 存储结果
            results[f"synthetic_layer_{i}"] = {
                "layer_idx": i,
                "shape": [seq_len, seq_len],
                "synthetic": True,
                "top_interactions": top_interactions[:10]
            }
    
    # 保存生成的文本和可视化信息
    with open(output_path / "output.json", "w") as f:
        json.dump({
            "prompt": args.prompt,
            "image": args.image_path,
            "output": generated_text,
            "state_matrix_results": results
        }, f, indent=4)
    
    # 移除钩子
    tracker.remove_hooks()
    
    print(f"Visualization completed. Results saved to {args.output_path}")
    print(f"Attention map-like visualizations are available in {attn_maps_dir}")


if __name__ == "__main__":
    main() 