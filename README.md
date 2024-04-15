

### Files for synthetic-data generation

```synthgenerate.py``` hosts all the functions required to generate a new-batch of data.  
```common_helpers.py``` contains the helper functions that include preprocessing, validation and post-processing steps for generating data.   
```finetune.py``` hosts the code to run finetuning on the curated dataset.  


To run the finetuning script:
```
base_model: defog/sqlcoder-7b-2
data_path: ./abs.json
output_dir: ./lora-defog
batch_size: 16
micro_batch_size: 4
num_epochs: 10
learning_rate: 0.0003
cutoff_len: 256
val_set_size: 10
lora_r: 128
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head']
train_on_inputs: True
add_eos_token: False
group_by_length: False
wandb_project: yashxdefog
wandb_run_name: sqlcoder-lora-defog-1
wandb_watch: 
wandb_log_model: 
resume_from_checkpoint: False
prompt template: alpaca
```
