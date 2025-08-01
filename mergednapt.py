#!/usr/bin/env python
"""
Usage:
python merge_llama_with_chinese_lora.py \
    --base_model path/to/llama/model \
    --lora_model path/to/first/lora/model [path/to/second/lora/model] \
    --output_type [pth|huggingface] \
    --output_dir path/to/output/dir
"""
import subprocess
import os
# 设置环境变量, autodl一般区域
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

import argparse
import json
import gc
import torch
import peft
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument('--lora_model', default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument('--offload_dir', default=None, type=str,
                    help="(Optional) Please specify a temp folder for offloading (useful for low-RAM machines). Default None (disable offload).")
parser.add_argument('--output_type', default='huggingface', choices=['pth', 'huggingface'], type=str,
                    help="save the merged model in pth or huggingface format.")
parser.add_argument('--output_dir', default='./', type=str)

# --- 以下的 'pth' 格式轉換相關程式碼保持不變 ---
emb_to_model_size = {
    4096: '7B',
    5120: '13B',
    6656: '33B',
    8192: '65B',
    # 增加對 8B 的支持
    8192: '8B'
}
num_shards_of_models = {'7B': 1, '13B': 2, '33B': 4, '65B': 8, '8B': 1}
params_of_models = {
    '7B': {
        "dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32,
        "norm_eps": 1e-06, "vocab_size": -1,
    },
    '13B': {
        "dim": 5120, "multiple_of": 256, "n_heads": 40, "n_layers": 40,
        "norm_eps": 1e-06, "vocab_size": -1,
    },
    '33B': {
        "dim": 6656, "multiple_of": 256, "n_heads": 52, "n_layers": 60,
        "norm_eps": 1e-06, "vocab_size": -1,
    },
    '65B': {
        "dim": 8192, "multiple_of": 256, "n_heads": 64, "n_layers": 80,
        "norm_eps": 1e-05, "vocab_size": -1,
    },
    '8B': {
        "dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32,
        "norm_eps": 1e-05, "vocab_size": -1,
    },
}

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def translate_state_dict_key(k):
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight": return "tok_embeddings.weight"
    elif k == "model.norm.weight": return "norm.weight"
    elif k == "lm_head.weight": return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"): return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"): return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"): return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"): return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"): return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"): return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"): return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"): return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"): return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k: return None
        else: raise NotImplementedError
    else: raise NotImplementedError

def unpermute(w):
    return (w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim))

def save_shards(model_sd, num_shards: int):
    # 省略未更改的 save_shards 函數內容以保持簡潔...
    # (此函數與你的原始碼相同)
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    base_model_path = args.base_model
    lora_model_paths = [s.strip() for s in args.lora_model.split(',') if len(s.strip()) != 0]
    output_dir = args.output_dir
    output_type = args.output_type
    offload_dir = args.offload_dir

    print(f"Base model: {base_model_path}")
    print(f"LoRA model(s): {lora_model_paths}")

    # [FIX] 載入基礎模型。 device_map='auto' 能更好地利用可用硬體
    print("Loading base model...")
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map='auto', # 使用 'auto' 讓 accelerate 自動分配
        offload_folder=offload_dir if offload_dir else None,
    )

    # [FIX] 從基礎模型路徑載入 Tokenizer，這應該在迴圈外執行一次
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model_path)

    # [FIX] 如果 Tokenizer 的詞彙表大於模型的詞彙表，則調整模型大小
    # 這通常發生在 fine-tuning 過程中增加了新的 token
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    
    print(f"Base model vocab size: {model_vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")

    if model_vocab_size != tokenizer_vocab_size:
        print(f"Resizing model vocabulary from {model_vocab_size} to {tokenizer_vocab_size}")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    # [MODIFIED] 遍歷所有 LoRA 模型並依次合併
    for lora_model_path in lora_model_paths:
        print(f"Loading and merging LoRA: {lora_model_path}")
        
        # 使用 PeftModel 將 LoRA 權重加載到當前模型上
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            device_map='auto', # 同樣使用 auto
        )
        
        # 合併 LoRA 權重
        base_model = lora_model.merge_and_unload()
        print(f"Successfully merged {lora_model_path}")
    
    # [MODIFIED] 所有 LoRA 都合併完成後，儲存最終的模型
    print(f"\nAll LoRA models have been merged. Saving final model to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_type == 'huggingface':
        print("Saving to Hugging Face format...")
        base_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir) # 儲存 Tokenizer
    else: # output_type == 'pth'
        print("Saving to pth format...")
        base_model_sd = base_model.state_dict()
        
        embedding_size = base_model.get_input_embeddings().weight.size(1)
        model_size = emb_to_model_size.get(embedding_size, 'Unknown')
        if model_size == 'Unknown':
             raise ValueError(f"Could not determine model size from embedding size {embedding_size}")
        
        params = params_of_models[model_size]
        num_shards = num_shards_of_models[model_size]
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

        save_shards(model_sd=base_model_sd, num_shards=num_shards)
        # pth 格式也需要儲存 tokenizer
        tokenizer.save_pretrained(output_dir)

    print(f"✅ Final merged model saved to {output_dir}")
