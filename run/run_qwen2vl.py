from PIL import Image
import requests
import torch
from typing import Dict
import argparse
import warnings
from modelscope import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from utils import visualize

def main(args):
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)


    model_name = args.model_path.split('/')[-1]

    # Image
    image = Image.open(args.image_file)

    conversation = [
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


    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    img_start = (inputs['input_ids'] == model.config.vision_start_token_id).nonzero()[0][1].item()
    img_end = (inputs['input_ids'] == model.config.vision_end_token_id).nonzero()[0][1].item()
    # inst_end = (inputs['input_ids'] == 151645).nonzero()[-1][1].item()
    sys_start = (inputs['input_ids'] == 151644).nonzero()[0][1].item()
    sys_end = (inputs['input_ids'] == 151645).nonzero()[0][1].item()
    # assit_start = (inputs['input_ids'] == 151644).nonzero()[-1][1].item()
    inputs = inputs.to("cuda")

    len_dict = {'sys': sys_end - sys_start + 1, 'img': img_end - img_start + 1, 'inst': inputs['input_ids'].size(1) - img_end - 1}
    # print(processor.tokenizer.batch_decode(inputs['input_ids']))

    # Inference: Generation of the output
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, output_attentions=True, return_dict_in_generate=True)
    
    output_ids = output.sequences
    attentions = output.attentions #(generate_length, num_layers, [batch_size, num_heads, seq_length, seq_length])

    print(output_ids.size())
    generated_texts = processor.tokenizer.decode(output_ids[0, inputs['input_ids'].size(1):], skip_special_tokens=True).strip()
    print(len(attentions))
    print(len(attentions[0]))
    print(attentions[0][0].size())
    print(attentions[-1][0].size())
    seq_len = output_ids.size(1)
    len_dict['out'] = seq_len -  inputs['input_ids'].size(1)
    print(len_dict)
    visualize(model_name,attentions, seq_len, len_dict, model.config.num_hidden_layers)


    print(f"\nGeneration: {generated_texts}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)