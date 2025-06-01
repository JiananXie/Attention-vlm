#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse
from utils import visualize
import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="cuda")


    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cuda"))

    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    print(f"image_tensor: {image_tensor.shape}")
    print(f"image size: {image.size}")
    img_idx = torch.where(input_ids.squeeze(0) == IMAGE_TOKEN_INDEX)[0].item()
    # img_end = torch.where(input_ids.squeeze(0) == IMAGE_TOKEN_INDEX)[-1].item()
    len_dict = {}
    len_dict['sys'] = img_idx
    print(input_ids)
    # Run inference
    with torch.inference_mode():
        # 在generate之前添加检查
        print("=== 数值稳定性检查 ===")
        
        # 检查输入
        print(f"Input IDs stats: min={input_ids.min()}, max={input_ids.max()}")
        print(f"Input contains special tokens: {(input_ids < 0).any()}")
        
        # 检查图像tensor
        print(f"Image tensor: min={image_tensor.min():.4f}, max={image_tensor.max():.4f}")
        print(f"Image has NaN: {torch.isnan(image_tensor).any()}")
        print(f"Image has Inf: {torch.isinf(image_tensor).any()}")
        
        # 检查模型参数
        has_nan_params = any(torch.isnan(p).any() for p in model.parameters())
        has_inf_params = any(torch.isinf(p).any() for p in model.parameters())
        print(f"Model has NaN params: {has_nan_params}")
        print(f"Model has Inf params: {has_inf_params}")
        
        if has_nan_params or has_inf_params:
            raise ValueError("Model parameters contain invalid values!")

        output = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True,)
    output_ids = output.sequences
    attentions = output.attentions  # (generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])

    seq_len = attentions[0][0].shape[3] + len(attentions)
    img_len = seq_len - len_dict['sys'] - (input_ids.shape[1] - img_idx - 1) - len(attentions)
    len_dict['img'] = img_len
    len_dict['inst'] = input_ids.shape[1] - img_idx -1 
    len_dict['out'] = len(attentions)
    print(f"total seq length: {seq_len}")
    print(f"input length: {attentions[0][0].shape[3]}")
    print(len_dict)
    print(attentions[-1][0].shape)
    #attention map
    interval = 20
    if model.config.image_aspect_ratio == 'anyres':
        interval = 40
    visualize(model_name, attentions, seq_len, len_dict, model.config.num_hidden_layers, interval)



    outputs = tokenizer.batch_decode(output_ids)[0].strip()
    print(f"Generation: {outputs}")
    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None, help="location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    predict(args)
