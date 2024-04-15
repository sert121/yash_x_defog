

### Files for synthetic-data generation

```synthgenerate.py``` hosts all the functions required to generate a new-batch of data.  
```common_helpers.py``` contains the helper functions that include preprocessing, validation and post-processing steps for generating data.   
```finetune.py``` hosts the code to run finetuning on the curated dataset.  


To run the finetuning script, run the following command with custom args(if required). The script uses Fire, so it takes automatically converts the arguments we pass in.   

Example usage:  
```
python3 finetuning.py --data_path 'combined_data_v2.json' --base_model 'defog/sqlcoder-7b-2' --wandb_run_name: sqlcoder-lora-defog-2 --wandb_project yashxdefog
```

