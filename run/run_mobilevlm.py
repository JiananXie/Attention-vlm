import sys
import os
import argparse
import warnings
import torch
from PIL import Image
from utils import visualize
# 添加 MobileVLM 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MobileVLM'))
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from scripts.inference import inference_once

def inference_once(args):
    disable_torch_init()
    model_name = args.model_path.split('/')[-1]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)

    images = [Image.open(args.image_file).convert("RGB")]
    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # Input
    input_ids = (tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda())
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    img_idx = torch.where(input_ids.squeeze(0) == IMAGE_TOKEN_INDEX)[0].item()
    len_dict = {}
    len_dict['sys'] = img_idx
    print(input_ids)
    # Inference
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            output_attentions=True,
            return_dict_in_generate=True
        )
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
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    print(f"🚀 {model_name}: {outputs}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", action='store_true', default=False)
    parser.add_argument("--load_4bit", action='store_true', default=False)
    args = parser.parse_args()

    inference_once(args)