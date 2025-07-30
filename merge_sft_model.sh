#!/bin/sh
python mergednapt.py \
    --base_model dnahlm-llama-8b-sft-v0 \
    --lora_model PromoterPredictTask_Final-RK4/sft_lora_model \
    --output_type huggingface \
    --output_dir LlaMaDNAPromoter_0605
