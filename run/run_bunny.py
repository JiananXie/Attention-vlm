import argparse
import torch
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from utils import visualize
from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates, SeparatorStyle
from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
    KeywordsStoppingCriteria


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           args.model_type, args.load_8bit,
                                                                           args.load_4bit, device=args.device)

    conv_mode = "bunny"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode,
                                                                                                              args.conv_mode,
                                                                                                              args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    roles = conv.roles

    image = load_image(args.image_file)
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=model.dtype) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=model.dtype)

    inp = args.query
    
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model.device)
    img_indices= (input_ids == IMAGE_TOKEN_INDEX).nonzero()
    img_start = img_indices[0][1].item()
    # img_end = img_indices[-1][1].item()
    # print(input_ids) 

    len_dict = {'sys': img_start}

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # print(f"{roles[1]}: ", end="")
    
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            repetition_penalty=args.repetition_penalty,
            stopping_criteria=[stopping_criteria],
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
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
