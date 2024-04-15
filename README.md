

### Files for synthetic-data generation

```synthgenerate.py``` hosts all the functions required to generate a new-batch of data.  
```common_helpers.py``` contains the helper functions that include preprocessing, validation and post-processing steps for generating data.   
```finetune.py``` hosts the code to run finetuning on the curated dataset.  

### Synthetic Data generation
The synthgenerate file generates k-shot examples and combines then in the form of input-output pairs in a file like (```combined_data.json```) to be ready to be processed for finetuning purposes. To keep things consistent we use a alpaca-dataset format that includes instruction, output and 
Example usage:  
```
python3 synthgenerate.py
```
Few args that can be used to customize the generation process. Rest included in the file. 
Note: for simplicity the ```qa_collection``` folder is read recursively when ```combine_json_files``` is called by the main function in synthgenerate. 
Can be added as a separate flag when preprocessing data.

```
'--samples'       | Number of samples 
'--method'        | Method of Generation:  'k_shot_generation' or 'follow_up_generation'
'--output_path',  | Output path for storing the combined responses : './qa_collection/combined_v3.json' # the finetuning script reads the data from this dir
'--input_examples'| Golden queries or reference set : 'macmillan_golden_queries.csv' 
'--metadata'      | Metadata (table) used to generate more accuracte sql: 'macmillan_md.csv'
```

### Finetuning
To run the finetuning script, run the following command with custom args(if required). The script uses Fire, so it takes automatically converts the arguments we pass in.   

Example usage:  
```
python3 finetuning.py --data_path 'combined_data_v2.json' --base_model 'defog/sqlcoder-7b-2' --wandb_run_name: sqlcoder-lora-defog-2 --wandb_project yashxdefog
```

