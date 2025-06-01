import os
import json
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

def get_model_layers(model_dir):
    if not os.path.exists(model_dir):
        return []
    return sorted([d for d in os.listdir(model_dir) if d.startswith('layer_')], 
                 key=lambda x: int(x.split('_')[1]))

def extract_pie_values(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([137, 255, 255]))


        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))


        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)


        purple_mask = cv2.inRange(hsv, np.array([138, 50, 50]), np.array([160, 255, 255]))
        
        # Count pixels for each color
        blue_pixels = cv2.countNonZero(blue_mask)  # inst
        green_pixels = cv2.countNonZero(green_mask)  # img
        red_pixels = cv2.countNonZero(red_mask)  # sys
        purple_pixels = cv2.countNonZero(purple_mask)  # out
        
        # Calculate proportions
        total_colored = blue_pixels + green_pixels + red_pixels + purple_pixels
        if total_colored == 0:
            return None
            
        return {
            'inst': blue_pixels / total_colored,
            'img': green_pixels / total_colored,
            'sys': red_pixels / total_colored,
            'out': purple_pixels / total_colored
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def collect_attention_data():
    result_dir = 'result'
    data = defaultdict(lambda: defaultdict(dict))
    
    # 遍历result目录下的所有模型目录
    for model_name in os.listdir(result_dir):
        model_dir = os.path.join(result_dir, model_name)
        if not os.path.isdir(model_dir) or model_name.startswith('.'):
            continue
            
        print(f"Processing model: {model_name}")
        layers = get_model_layers(model_dir)
        
        if not layers:
            print(f"No layers found in {model_name}")
            continue
            
        for layer in layers:
            layer_num = int(layer.split('_')[1])
            layer_dir = os.path.join(model_dir, layer)
            pie_file = os.path.join(layer_dir, f'attn_pie_{layer_num}.png')
            
            if os.path.exists(pie_file):
                # 提取饼图数据
                pie_values = extract_pie_values(pie_file)
                if pie_values:
                    data[model_name][layer_num] = {
                        'pie_file': pie_file,
                        'exists': True,
                        'values': pie_values
                    }
                else:
                    data[model_name][layer_num] = {
                        'exists': True,
                        'error': 'Failed to extract values'
                    }
            else:
                data[model_name][layer_num] = {
                    'exists': False
                }

    return data

def save_data(data, base_name):
    """保存数据到多种格式"""
    # 确保输出目录存在
    os.makedirs('analysis', exist_ok=True)
    base_name = os.path.join('analysis', base_name)
    
    # 转换defaultdict为普通dict
    data_dict = {k: dict(v) for k, v in data.items()}
    
    # 保存为CSV格式
    csv_file = f'{base_name}.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('model,layer,has_pie,img_attention,inst_attention,sys_attention,out_attention,error\n')
        
        # 写入数据
        for model_name, layers in data_dict.items():
            for layer_num, info in sorted(layers.items()):
                if info['exists']:
                    if 'values' in info:
                        values = info['values']
                        f.write(f'{model_name},{layer_num},True,{values["img"]:.3f},{values["inst"]:.3f},{values["sys"]:.3f},{values["out"]:.3f},\n')
                    else:
                        f.write(f'{model_name},{layer_num},True,,,,,{info.get("error", "Unknown error")}\n')
                else:
                    f.write(f'{model_name},{layer_num},False,,,,,No pie chart\n')

def main():
    # 收集数据
    print("开始收集注意力数据...")
    attention_data = collect_attention_data()
    
    base_name = 'attention_analysis'
    save_data(attention_data, base_name)
    print(f"数据已保存到: analysis/{base_name}.csv")
    
    print("\n模型统计信息:")
    for model_name, layers in attention_data.items():
        total_layers = len(layers)
        existing_layers = sum(1 for layer in layers.values() if layer['exists'])
        valid_layers = sum(1 for layer in layers.values() if layer['exists'] and 'values' in layer)
        print(f"{model_name}:")
        print(f"  总层数: {total_layers}")
        print(f"  有饼图的层数: {existing_layers}")
        print(f"  成功提取数据的层数: {valid_layers}")
        print()

if __name__ == '__main__':
    main()
