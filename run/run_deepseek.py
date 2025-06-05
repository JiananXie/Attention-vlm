# run deepseek-vl model, see https://github.com/deepseek-ai/DeepSeek-VL

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

from .utils import visualize

def get_segment_lengths(tokenizer, prompt_text, num_image_tokens=576):
    # BOS token
    bos_token = tokenizer.bos_token or "<s>" # Fallback if not defined
    len_sys = len(tokenizer.encode(bos_token, add_special_tokens=False))

    # Image tokens
    len_img = num_image_tokens

    # Instruction segment (e.g., "User: <prompt_text>\n")
    user_prefix = "User: "
    user_suffix = "\n" # Common separator
    inst_content = user_prefix + prompt_text + user_suffix
    len_inst = len(tokenizer.encode(inst_content, add_special_tokens=False))

    # Output/Assistant prefix segment (e.g., "Assistant:\n")
    assistant_prefix = "Assistant:"
    assistant_suffix = "\n" # Common separator
    out_content = assistant_prefix + assistant_suffix
    len_out = len(tokenizer.encode(out_content, add_special_tokens=False))
    
    len_dict = {
        'sys': len_sys,
        'img': len_img,
        'inst': len_inst,
        'out': len_out
    }
    return len_dict

def run_deepseek_attention_and_visualize(image_path: str, prompt: str, model_id: str = "deepseek-ai/deepseek-vl-7b-base", output_dir_name: str = "deepseek_vl_7b_base_viz"):
    """
    Runs the DeepSeek-VL model, extracts LLM attention, and visualizes it using utils.visualize.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}. Using a dummy image.")
        image = Image.new("RGB", (224, 224), (0, 0, 0)) # Dummy image

    if not hasattr(model, 'vl_chat_processor'):
        print("Error: Model does not have 'vl_chat_processor'. Ensure correct DeepSeek-VL model.")
        try:
            from deepseek_vl.models import VLChatProcessor
            vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
            print("Loaded VLChatProcessor from deepseek_vl library.")
        except ImportError:
            print("Failed to import VLChatProcessor. Ensure DeepSeek-VL repo is setup.")
            return
        except Exception as e:
            print(f"Error loading VLChatProcessor separately: {e}")
            return
    else:
        vl_chat_processor = model.vl_chat_processor
        if vl_chat_processor.tokenizer != tokenizer:
             print("Warning: model.vl_chat_processor.tokenizer differs. Using AutoTokenizer instance.")

    messages = [
        {"role": "User", "content": f"<image_placeholder>{prompt}"},
        {"role": "Assistant", "content": ""}
    ]
    
    inputs_for_forward_pass = vl_chat_processor(
        messages=messages,
        images=[image],
        return_tensors="pt",
        tokenizer=tokenizer
    ).to(device)

    input_ids = inputs_for_forward_pass.input_ids
    actual_sequence_length = input_ids.shape[1]
    
    print(f"Actual input sequence length: {actual_sequence_length}")

    print("\nRunning a direct forward pass for attention extraction...")
    with torch.no_grad():
        model_outputs = model(
            **inputs_for_forward_pass,
            output_attentions=True,
            return_dict=True
        )
    
    llm_attentions = model_outputs.attentions

    if not llm_attentions:
        print("No LLM attentions found. Cannot visualize.")
        return

    num_llm_layers = len(llm_attentions)
    print(f"Extracted {num_llm_layers} LLM attention layers.")
    if num_llm_layers > 0 and llm_attentions[0] is not None:
        print(f"Shape of one LLM attention layer tensor: {llm_attentions[0].shape}")
    else:
        print("LLM attentions tuple is empty or contains None.")
        return

    viz_model_name = output_dir_name
    attentions_for_visualize = [llm_attentions]
    seq_len_for_visualize = actual_sequence_length
    num_hidden_layers_for_visualize = num_llm_layers
    
    final_len_dict = {}
    final_len_dict['sys'] = 1 # BOS token
    num_image_tokens = 576
    final_len_dict['img'] = num_image_tokens
    
    current_sum_sys_img = final_len_dict['sys'] + final_len_dict['img']
    
    assistant_prompt_str = "Assistant:" 
    assistant_prompt_ids = tokenizer.encode(assistant_prompt_str, add_special_tokens=False)

    found_assistant_at = -1
    input_ids_list = input_ids[0].tolist()
    # Search for assistant_prompt_ids after a plausible length for BOS + image_tokens + some user prompt
    # Start search after BOS + image_tokens - a small buffer for short user prompts
    search_start_index = max(0, current_sum_sys_img - len(assistant_prompt_ids) - 10) 
    if len(input_ids_list) > len(assistant_prompt_ids):
        for i in range(search_start_index, len(input_ids_list) - len(assistant_prompt_ids) + 1):
            if input_ids_list[i:i+len(assistant_prompt_ids)] == assistant_prompt_ids:
                # Ensure this is not within the image tokens or too early
                if i >= current_sum_sys_img - 5: # Allow assistant prompt to start right after image tokens for very short user prompts
                    found_assistant_at = i
                    break
    
    if found_assistant_at != -1:
        len_inst_actual = found_assistant_at - current_sum_sys_img
        final_len_dict['inst'] = max(0, len_inst_actual) # Ensure non-negative
        
        current_sum_sys_img_inst = current_sum_sys_img + final_len_dict['inst']
        len_out_actual = (actual_sequence_length - 1) - current_sum_sys_img_inst
        final_len_dict['out'] = max(0, len_out_actual) # Ensure non-negative
    else:
        print("Warning: Could not robustly determine 'inst' and 'out' lengths. Using heuristic.")
        remaining_for_inst_out = (actual_sequence_length - 1) - current_sum_sys_img
        if remaining_for_inst_out > 0:
            final_len_dict['inst'] = max(1, int(remaining_for_inst_out * 0.8))
            final_len_dict['out'] = remaining_for_inst_out - final_len_dict['inst']
            if final_len_dict['out'] < 0: final_len_dict['out'] = 0
            if final_len_dict['inst'] + final_len_dict['out'] != remaining_for_inst_out:
                 final_len_dict['inst'] = remaining_for_inst_out - final_len_dict['out']
        else:
            final_len_dict['inst'] = 0
            final_len_dict['out'] = 0
    
    # Final check on sum for S-1 tokens
    if sum(final_len_dict.values()) != actual_sequence_length -1 :
         final_len_dict['out'] = (actual_sequence_length -1) - (final_len_dict['sys'] + final_len_dict['img'] + final_len_dict['inst'])
         final_len_dict['out'] = max(0, final_len_dict['out'])

    print(f"Final len_dict for visualization: {final_len_dict}")
    print(f"Sum of final_len_dict: {sum(final_len_dict.values())}, Target for S-1 tokens: {actual_sequence_length - 1}")

    if sum(final_len_dict.values()) != actual_sequence_length - 1:
         print(f"Warning: Sum of len_dict ({sum(final_len_dict.values())}) does not match S-1 ({actual_sequence_length-1}). Visualization may be incorrect.")

    for k_ld, v_ld in final_len_dict.items():
        if v_ld < 0:
            print(f"Warning: len_dict has negative value for {k_ld}: {v_ld}. Setting to 0.")
            final_len_dict[k_ld] = 0

    print(f"\nCalling visualize function from utils.py...")
    visualize(
        model_name=viz_model_name,
        attentions=attentions_for_visualize,
        seq_len=seq_len_for_visualize,
        len_dict=final_len_dict,
        num_hidden_layers=num_hidden_layers_for_visualize,
        interval=1
    )
    print("Visualization process finished.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attention_vlm_root = os.path.dirname(script_dir)
    workspace_root_guess = os.path.dirname(attention_vlm_root)
    
    default_image_name = "view.jpg"
    current_image_path = os.path.join(os.getcwd(), default_image_name)

    if not os.path.exists(current_image_path):
        current_image_path = os.path.join(workspace_root_guess, default_image_name)
        if not os.path.exists(current_image_path):
            print(f"'{default_image_name}' not found in CWD or guessed workspace root ({workspace_root_guess}).")
            dummy_img_path = os.path.join(script_dir, "dummy_deepseek_image.jpg")
            try:
                dummy_img = Image.new("RGB", (224, 224), "blue")
                dummy_img.save(dummy_img_path)
                current_image_path = dummy_img_path
                print(f"Using dummy image: '{current_image_path}'")
            except Exception as e:
                print(f"Could not create dummy image: {e}. Exiting.")
                exit(1)
        else:
            print(f"Using image: '{current_image_path}' from guessed workspace root.")
    else:
        print(f"Using image: '{current_image_path}' from CWD.")

    sample_prompt = "What are the things I should be cautious about when I visit this place? Are there any dangerous areas or activities I should avoid? Or any other important information I should know?"
    output_directory_name_for_viz = "deepseek_run_output"

    print(f"Running DeepSeek-VL attention extraction and visualization for image: '{current_image_path}' and prompt: '{sample_prompt}'")
    run_deepseek_attention_and_visualize(
        image_path=current_image_path,
        prompt=sample_prompt,
        output_dir_name=output_directory_name_for_viz
    )

    print("\nScript finished.")

