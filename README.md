# CCH-Large-Language-Model
---
title: LLM交接

---

## DNA-LLaMA3
程式載點
https://github.com/YoshiGaWa87/CCH-Large-Language-Model
模型載點
https://huggingface.co/weichen0907
數據集載點
https://huggingface.co/datasets/leannmlindsey/GUE

### 索取運算節點
```
vim salloc.sh
```
```
salloc -A ACD113082 -J gpu-test -t 12:00:00 -p gp1d --mem=737280 --ntasks=8 --nodes=1 --gpus-per-node=8 --cpus-per-task=4
```
```
sh salloc.sh
```
```
ssh gnxxxx
```
### 下載程式碼
```
git clone https://github.com/YoshiGaWa87/CCH-Large-Language-Model.git
```
```
cd CCH-Large-Language-Model
```
### 創建虛擬python環境
```
conda activate -p /home/<username>/CCH-Large-Language-Model/dnallamapy312
```
### 載入cuda
```
module load cuda/11.7
```
### 安裝pytorch
https://pytorch.org/get-started/previous-versions/
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```
### 安裝大語言模型相關套件(huggingface、deepspeed...)
```
pip install -r requirements.txt
```
```
pip install git+https://github.com/huggingface/peft.git
```
(只需要執行一次就好)
```
conda install -c conda-forge git-lfs
```
進到模型載點
![image](https://hackmd.io/_uploads/BJS4Vc_vlg.png)
### 下載模型

![image](https://hackmd.io/_uploads/BkGEK85wee.png)

![image](https://hackmd.io/_uploads/Sy_AuI9vee.png)

![image](https://hackmd.io/_uploads/rJ6IYL5Pxg.png)

使用git clone進行下載模型
```
git clone https://huggingface.co/weichen0907/DNA-LLaMA3
```


### 數據處理
下載數據
```
git clone https://huggingface.co/datasets/leannmlindsey/GUE
```
這裡採用啟動子檢測數據(標註為1為啟動子序列、0則為非啟動子)
![image](https://hackmd.io/_uploads/SJrSTUqPel.png)

接下來需要將此數據設計為一個微調LLaMa的數據模板
https://ithelp.ithome.com.tw/m/articles/10318917
```
{"instruction": "Determine promoter  detection of following dna sequence, The result will be one of the following: Non-promoter, promoter.", 
"input": "GCTAGCTCATCTTGCGGCTGGGCGGGGCCCAGGACTGCTGCTGCTGACCGCCTTGATAGGCTACACCGTG", 
"output": "promoter"}
```
轉換的程式碼transfer.py，將原本的csv檔轉為json檔案來進行訓練
```
python transfer.py
```
![image](https://hackmd.io/_uploads/S1Osxwqvgg.png)





### 看主節點ip
```
ifconfig
```
![image](https://hackmd.io/_uploads/SJA4US5Pll.png)

### 執行訓練
```
torchrun --nnodes 2 --nproc_per_node 8 --node_rank 1 --master_addr "10.100.12.8" --master_port 8787 finetune.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path DNA-LLaMA3 \
    --tokenizer_name_or_path DNA-LLaMA3 \
    --dataset_dir /home/n96134417/CCH-Large-Language-Model/promoter_data/train_data \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_steps 2000 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --max_seq_length 512 \
    --output_dir PromoterPredictTask-AdamW-0801 \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 8 \
    --lora_alpha 32 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --modules_to_save "embed_tokens,lm_head" \
    --lora_dropout 0.06 \
    --torch_dtype float16 \
    --validation_file /home/n96134417/CCH-Large-Language-Model/promoter_data/val_data/val.json \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
```
### 合併模型
```
chmod 777 merge_sft_model.sh
```
```
vim merge_sft_model.sh
```
```
#!/bin/sh
python mergednapt.py \
    --base_model DNA-LLaMA3 \
    --lora_model PromoterPredictTask-AdamW/checkpoint-370 \
    --output_type huggingface \
    --output_dir LlaMaDNAPromoter-AdamW
```
```
./merge_sft_model.sh
```

### 測試模型
```
vim Test.py
```
#### 修改model及tokenizer的路徑
```
tokenizer = LlamaTokenizerFast.from_pretrained("LlaMaDNAPromoter-AdamW")
model = LlamaForCausalLM.from_pretrained(
    "LlaMaDNAPromoter-AdamW",
    device_map="auto"
)
```
```
python Test.py
```
## RK4優化器使用
創建RK4優化器程式碼
RK4_optimizer.py
```
import torch
from torch.optim.optimizer import Optimizer
import math

class RK4Optimizer(Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate/step size: {lr}")
        defaults = dict(lr=lr, weight_decay_rk4=0.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None): # Make closure optional as DeepSpeed might not provide it
        # This RK4Optimizer is designed to work with DeepSpeed.
        # DeepSpeed will have already performed the backward pass and populated p.grad.
        # We will use the gradients provided by DeepSpeed as our k1.
        # For a true RK4 method (multiple model evaluations per step),
        # significant re-architecture would be needed to work with DeepSpeed.
        # Here, we simplify to an RK4-inspired update using the available gradient.

        loss = None
        if closure is not None:
            # If a closure is provided, this path might be taken in a non-DeepSpeed context.
            # However, for DeepSpeed, the loss is typically handled by the Trainer.
            with torch.enable_grad():
                loss = closure()
            # Gradients should have been computed by the closure.
            # If closure was provided, we'll assume it handles gradient calculation for k1.

        # Collect parameters and their initial state (theta_n)
        params_with_state = []
        for group in self.param_groups:
            h_group = group['lr']
            wd_group = group.get('weight_decay_rk4', 0.0)

            for p in group['params']:
                if p.requires_grad:
                    # Store original_state to compute the final update from
                    # Ensure p.grad is not accumulated if it's already there (should be handled by DeepSpeed)
                    params_with_state.append({
                        'param': p,
                        'original_state': p.clone().detach(), # Store the state before any updates
                        'h': h_group,
                        'wd': wd_group
                    })

        if not params_with_state:
            return loss # No trainable parameters

        # --- RK4-like update logic ---
        # We will use the gradients pre-computed by DeepSpeed for the current step.
        # This gradient effectively acts as our 'k1'.
        # For simplicity and compatibility with DeepSpeed's single-pass step,
        # we will approximate k2, k3, k4 as being proportional to k1.
        # A true RK4 would require multiple forward-backward passes and state changes,
        # which is difficult to implement efficiently and correctly with DeepSpeed's architecture.

        k1_grads = []
        for p_data in params_with_state:
            p = p_data['param']
            if p.grad is None:
                raise RuntimeError("Gradients are None. RK4Optimizer expects gradients to be populated by DeepSpeed/Trainer before step().")
            
            grad = p.grad.clone().detach() # Get the gradient from DeepSpeed
            
            # Apply weight decay (if any) to the k1 gradient
            if p_data['wd'] != 0:
                grad.add_(p.data, alpha=p_data['wd'])
            k1_grads.append(grad)

        # Calculate the delta_thetas for each k.
        # Since we're using a single gradient pass, k1_delta, k2_delta, k3_delta, k4_delta
        # will all be based on the same gradient (k1_grads).
        # This simplifies the RK4 formula to effectively just a scaled SGD/Adam update.
        k1_delta_thetas = [-p_data['h'] * g for p_data, g in zip(params_with_state, k1_grads)]
        
        # In a typical RK4, k2, k3, k4 would be gradients from intermediate states.
        # With DeepSpeed, we get one set of gradients per step.
        # To maintain the RK4 structure, we can (loosely) approximate k2, k3, k4
        # as being proportional to k1. This is a heuristic for compatibility.
        # Essentially, k1_delta_thetas will be the primary component.
        # If you want a *true* RK4, the problem needs a different approach or a custom
        # DeepSpeed engine, which is very complex.
        
        # For compatibility, we'll assign k2, k3, k4 the same value as k1.
        # This will make the final update effectively: theta_n + (6/6) * k1_delta = theta_n + k1_delta
        # which simplifies to an SGD-like update with your learning rate.
        # If you desire a different approximation for k2, k3, k4, you'd implement it here.
        k2_delta_thetas = k1_delta_thetas
        k3_delta_thetas = k1_delta_thetas
        k4_delta_thetas = k1_delta_thetas

        # Final parameter update (theta_n+1 = theta_n + (k1 + 2*k2 + 2*k3 + k4)/6)
        for i, p_data in enumerate(params_with_state):
            # Sum of coefficients: 1 + 2 + 2 + 1 = 6
            # If k1=k2=k3=k4, then (k1 + 2k1 + 2k1 + k1)/6 = 6k1/6 = k1
            final_combined_delta = (k1_delta_thetas[i] +
                                    2 * k2_delta_thetas[i] +
                                    2 * k3_delta_thetas[i] +
                                    k4_delta_thetas[i]) / 6.0
            
            p_data['param'].data.copy_(p_data['original_state'] + final_combined_delta)

        # Clear gradients manually, as DeepSpeed's zero_grad might operate before step()
        # or after, depending on configuration. It's safer for your optimizer to clear them
        # if it's the one responsible for the final update.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

        return loss # Return the loss if computed, otherwise None
```
整合進訓練程式碼中
train_RK4.py
```
from RK4_optimizer import RK4Optimizer
```
```
class RK4Trainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a default AdamW optimizer that can be overridden by our `RK4Optimizer`.
        """
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Check for the custom RK4Optimizer
            if self.args.optimizer_type == "RK4":
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = RK4Optimizer(trainable_params, lr=self.args.learning_rate)
                # Apply weight decay if specified, as RK4Optimizer might handle it internally
                if self.args.weight_decay > 0:
                    for group in self.optimizer.param_groups:
                        # Assuming RK4Optimizer uses a 'weight_decay_rk4' key or similar
                        # If not, adjust this key to what RK4Optimizer expects
                        group['weight_decay_rk4'] = self.args.weight_decay
                logger.info(f"Using custom RK4Optimizer with learning rate {self.args.learning_rate} and weight decay {self.args.weight_decay}")
```

### 執行訓練

```
torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_addr "10.100.12.25" --master_port 8787 train_RK4.py     --deepspeed ds_zero2_no_offload.json     --model_name_or_path DNA-LLaMA3     --tokenizer_name_or_path DNA-LLaMA3     --dataset_dir /home/n96134417/CCH-Large-Language-Model/promoter_data/train_data     --validation_split_percentage 0.001     --per_device_train_batch_size 2     --per_device_eval_batch_size 2     --do_train     --do_eval     --seed $RANDOM     --fp16     --num_train_epochs 10     --lr_scheduler_type cosine     --learning_rate 5e-3     --warmup_ratio 0.03     --weight_decay 0.01     --logging_strategy steps     --logging_steps 10     --save_strategy steps     --save_total_limit 3     --eval_strategy steps     --eval_steps 200     --save_steps 200     --gradient_accumulation_steps 8     --preprocessing_num_workers 4     --max_seq_length 512     --output_dir PromoterPredictTask-RK4     --overwrite_output_dir     --ddp_timeout 30000     --logging_first_step True     --lora_rank 8     --lora_alpha 32     --trainable "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"     --modules_to_save "embed_tokens,lm_head"     --lora_dropout 0.06     --torch_dtype float16     --validation_file /home/n96134417/CCH-Large-Language-Model/promoter_data/val_data/val.json     --gradient_checkpointing     --ddp_find_unused_parameters False     --optimizer_type RK4
```

### 訓練完成後再接續前面合併模型的步驟並進行模型測試
