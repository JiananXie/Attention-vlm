import argparse
import torch

from llava_phi.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_phi.conversation import conv_templates, SeparatorStyle
from llava_phi.model.builder import load_pretrained_model
from llava_phi.utils import disable_torch_init
from llava_phi.mm_utils import tokenizer_image_token, get_model_name_from_path
from utils import visualize
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'phi' in model_name.lower():
        conv_mode = "phi-2_v0"
    else:
        conv_mode = "default"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    img_indices= (input_ids == IMAGE_TOKEN_INDEX).nonzero()
    img_start = img_indices[0][1].item()
    len_dict = {'sys': img_start}
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,  # End of sequence token
            pad_token_id=tokenizer.eos_token_id,  # Pad token
            use_cache=True,
            output_attentions=True, 
            return_dict_in_generate=True)

    output_ids = output.sequences
    attentions = output.attentions #(generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])

    seq_len = attentions[0][0].shape[3] + len(attentions)
    img_len = seq_len - len_dict['sys'] - (input_ids.shape[1] - img_start - 1) - len(attentions)
    len_dict['img'] = img_len
    len_dict['inst'] = input_ids.shape[1] - img_start -1 
    len_dict['out'] = len(attentions)
    print(f"total seq length: {seq_len}")
    print(f"input length: {attentions[0][0].shape[3]}")
    print(len_dict)
    print(attentions[-1][0].shape)

    visualize(model_name, attentions, seq_len, len_dict, model.config.num_hidden_layers)

    generated_texts = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nGeneration: {generated_texts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/songx_lab/cse12110714/MLLM/VLM/models/llava-phi2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
