import requests
import matplotlib.pyplot as plt
import argparse
import warnings
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
import numpy as np   
import math
import os
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from utils import visualize

def main(args):
    DEVICE = "cuda:0"

    # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
    image = load_image(args.image_file)

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
         torch_dtype=torch.float16,    
    ).to(DEVICE)

    # Create inputs
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text",
                "text": "A chat between a curious human and an artificial intelligence assistant. "
               "The assistant gives helpful, detailed, and polite answers to the human's questions."}
            ]
        },
        {   
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.query},
            ]
        }      
    ]
    # processor.image_processor.do_image_splitting = False
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    img_indices= (inputs['input_ids'] == model.config.image_token_id).nonzero()
    img_start = img_indices[0][1].item()
    img_end = img_indices[-1][1].item()
    print(inputs['input_ids']) #[1, 371]
    len_dict = {'sys': img_start - 1, 'img': img_end - img_start + 3, 'inst': inputs['input_ids'].size(1) - img_end - 1}
    print(len_dict)
    print(processor.tokenizer.batch_decode(inputs['input_ids']))
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate
    output = model.generate(**inputs, use_cache=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens, output_attentions=True, return_dict_in_generate=True)
    output_ids = output.sequences
    attentions = output.attentions #(generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])

    print(output_ids.size())
    generated_texts = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(len(attentions))
    print(len(attentions[0]))
    print(attentions[0][0].size())
    print(attentions[-1][0].size())
    seq_len = output_ids.size(1)
    len_dict['out'] = seq_len -  inputs['input_ids'].size(1)

    visualize('idefics2',attentions, seq_len, len_dict, model.config.num_hidden_layers)


    print(f"\nGeneration: {generated_texts}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)