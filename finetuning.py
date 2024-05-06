import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import configparser
from tqdm import tqdm 
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import sqlparse
import openai

import json 
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from utils.prompter import Prompter
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
BASE_MODEL = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def validate_sql_using_gpt(data_point, table_metadata_ddl):
    '''
    Evaluating the generated SQL query using GPT-4
    '''
    system_prompt = '''You are an expert and thoughtful SQL analyst that works for a company called Defog AI. Your role is to validate SQL queries given to you. 
  
Always check the following when evaluating a SQL query:
1. Only use the table names and column names that are in the metadata schema. Do NOT use any other tables names or column names.
2. Do NOT create a JOIN statement or query multiple tables if the question can be answered using only one table.
3. When writing SELECT statements, always add the table alias as a prefix to the column name. For example, this SQL query is not valid: `SELECT a FROM table1 JOIN table2 ON table1.a = table2.a`. Instead, this query is correct: `SELECT table1.a FROM table1 JOIN table2 ON table1.a = table2.a`
4. SELECT statements should include all columns that are in the ORDER BY statements. For example, if the ORDER BY statement is `ORDER BY column_name`, then the SELECT statement should include `column_name`
5. Make sure that the GROUP BY statements do NOT contain an alias, and only contain original column names that exist in the schema.
6. If creating GROUP BY statements, always include columns with `id` in the column name in the SELECT and GROUP BY statements to ensure uniqueness.
7. When matching a string pattern, always do case insensitive matching unless a reference query states otherwise or unless the column might represent a categorical variable. You can chain multiple patterns using the OR operator. (e.g. LOWER(column_name) LIKE "%stringtomatch1%" OR LOWER(column_name) ILIKE "%stringtomatch2%")
8. When a user asks for data by month, they are typically asking for data by both the month and year
9. If the question cannot be answered given the database schema, always generate a query that says `SELECT 'Sorry, I could not answer that. Could you please rephrase your question?' AS answer;`. Do not give a closest approximation to the user's question. Do not use proxies for unavailable information.

You need to respond with your evaluation on the confidence of correctness on a scale of 1-5, where 5 is very confident.
'''


    system_prompt_thinking = '''You are an expert and thoughtful SQL analyst that works for a company called Defog AI. Your role is to validate SQL queries given to you. 
  
Always check the following when evaluating a SQL query:
1. Only use the table names and column names that are in the metadata schema. Do NOT use any other tables names or column names.
2. Do NOT create a JOIN statement or query multiple tables if the question can be answered using only one table.
3. When writing SELECT statements, always add the table alias as a prefix to the column name. For example, this SQL query is not valid: `SELECT a FROM table1 JOIN table2 ON table1.a = table2.a`. Instead, this query is correct: `SELECT table1.a FROM table1 JOIN table2 ON table1.a = table2.a`
4. SELECT statements should include all columns that are in the ORDER BY statements. For example, if the ORDER BY statement is `ORDER BY column_name`, then the SELECT statement should include `column_name`
5. Make sure that the GROUP BY statements do NOT contain an alias, and only contain original column names that exist in the schema.
6. If creating GROUP BY statements, always include columns with `id` in the column name in the SELECT and GROUP BY statements to ensure uniqueness.
7. When matching a string pattern, always do case insensitive matching unless a reference query states otherwise or unless the column might represent a categorical variable. You can chain multiple patterns using the OR operator. (e.g. LOWER(column_name) LIKE "%stringtomatch1%" OR LOWER(column_name) ILIKE "%stringtomatch2%")
8. When a user asks for data by month, they are typically asking for data by both the month and year
9. If the question cannot be answered given the database schema, always generate a query that says `SELECT 'Sorry, I could not answer that. Could you please rephrase your question?' AS answer;`. Do not give a closest approximation to the user's question. Do not use proxies for unavailable information.

You need to think and respond with your evaluation, expressing if the query is poor, good, or excellent. 
'''
    user_prompt = '''The SQL query to evaluate is: ```{sql_query}```. The database schema is represented in the following CSV string:
```{table_metadata_ddl}```.  You need to respond with your evaluation of the query and express the confidence of correctness. Always provide rating in the form Rating:<insert rating>'''

    client = openai.Client()

    chat_completion = client.chat.completions.create(
        messages=[
 {
            'role': 'system',
            'content': system_prompt_thinking
        },
            {
                "role": "user",
                "content": user_prompt, 
            }
        ],
        model="gpt-4",
    )

    label = chat_completion.choices[0].message.content
    rating = label.split("Rating:")[1]
    return label, rating

def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['Settings']

def query_gpt_model(question,table_metadata_ddl):
    '''
    Comparing base responses from sql coder and finetuned model to GPT responses
    '''
    system_prompt = '''You are an expert and thoughtful SQL analyst that works for a company called Defog AI. Your role is to convert user questions into valid {db_type} SQL queries, given a database schema. Recall that the current date in YYYY-MM-DD format is {date_today}.
  
Always follow these instructions for generating the SQL query:
1. Only use the table names and column names that are in the metadata schema. Do NOT use any other tables names or column names.
2. Do NOT create a JOIN statement or query multiple tables if the question can be answered using only one table.
3. When writing SELECT statements, always add the table alias as a prefix to the column name. For example, this SQL query is not valid: `SELECT a FROM table1 JOIN table2 ON table1.a = table2.a`. Instead, this query is correct: `SELECT table1.a FROM table1 JOIN table2 ON table1.a = table2.a`
4. SELECT statements should include all columns that are in the ORDER BY statements. For example, if the ORDER BY statement is `ORDER BY column_name`, then the SELECT statement should include `column_name`
5. Make sure that the GROUP BY statements do NOT contain an alias, and only contain original column names that exist in the schema.
6. If creating GROUP BY statements, always include columns with `id` in the column name in the SELECT and GROUP BY statements to ensure uniqueness.
7. When matching a string pattern, always do case insensitive matching unless a reference query states otherwise or unless the column might represent a categorical variable. You can chain multiple patterns using the OR operator. (e.g. LOWER(column_name) LIKE "%stringtomatch1%" OR LOWER(column_name) ILIKE "%stringtomatch2%")
8. When a user asks for data by month, they are typically asking for data by both the month and year
9. If the question cannot be answered given the database schema, always generate a query that says `SELECT 'Sorry, I could not answer that. Could you please rephrase your question?' AS answer;`. Do not give a closest approximation to the user's question. Do not use proxies for unavailable information.
'''
    user_prompt = '''The user's question is `{user_question}`. The database schema is represented in the following CSV string:
```{table_metadata_ddl}```
Give your response as just a markdown string with just the SQL query, and nothing else. Remember that the user question is {user_question}'''

    user_prompt = user_prompt.format(user_question=question, table_metadata_ddl=table_metadata_ddl)

    client = openai.Client()

    chat_completion = client.chat.completions.create(
        messages=[
 {
            'role': 'system',
            'content': system_prompt
        },
            {
                "role": "user",
                "content": user_prompt, # 
            }
        ],
        model="gpt-4",
    )

    label = chat_completion.choices[0].message.content
    try:
        sqlparse.format(label.split("[SQL]")[-1], reindent=True)
    except:
        return label

def generate_query(data_point,model):
    template = "### Task\nGenerate a SQL query to answer [QUESTION]{instruction}[/QUESTION]\n\n### Instructions\n- If you cannot answer the question with the available database schema, return 'I do not know'\n\n### Database Schema\nThe query will run on a database with the following schema:\n{table_metadata_string}\n\n### Answer\nGiven the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]\n[SQL]\n"
    instruction = data_point["instruction"]
    input = data_point["input"]
    table_metadata_string = data_point["table_metadata_string"]

    res = template.format(instruction= instruction,
                          question = instruction,
                          input=input,
                          table_metadata_string=table_metadata_string)

    inputs = tokenizer(res, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)
    return res,outputs[0].split("[SQL]")[-1]



def load_baseline_coder():
    model_name = "defog/sqlcoder-7b-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # if you have atleast 16GB of GPU memory, run load the model in float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )

    return model

def train(
    base_model: str = "defog/sqlcoder-7b-2",  
    data_path: str = "./combined_data_v4.json",
    output_dir: str = "./lora-defog",
    # training hyperparams
    batch_size = 16,
    eval_accumulation_steps = 8,
    save_steps = 100,
    eval_steps = 100,
    save_total_limit = 3,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 50,
    # lora hyperparams
    lora_r: int = 128,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05, 
    lora_target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "yashxdefog",
    wandb_run_name: str = f"sqlcoder-lora-defog-5   ",
    wandb_watch: str = "all",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "defog_base",  # The prompt template to use, will default to alpaca.
    prompt_template_description: str = "defog_base",
    use_config_file: bool = False,
    config_file: str = "config.ini",
    evaluate_generations: bool = False,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Defog-SQL model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"


    if use_config_file:
        print("Reading config file..")
        config = read_config(config_file)
        base_model = config.get("base_model", base_model)
        data_path = config.get("data_path", data_path)
        output_dir = config.get("output_dir", output_dir)
        batch_size = config.getint("batch_size", batch_size)
        micro_batch_size = config.getint("micro_batch_size", micro_batch_size)
        num_epochs = config.getint("num_epochs", num_epochs)
        learning_rate = config.getfloat("learning_rate", learning_rate)
        cutoff_len = config.getint("cutoff_len", cutoff_len)
        val_set_size = config.getint("val_set_size", val_set_size)
        lora_r = config.getint("lora_r", lora_r)
        save_steps = config.getint("save_steps", save_steps)
        eval_steps = config.getint("eval_steps", eval_steps)
        eval_accumulation_steps = config.getint("eval_accumulation_steps", eval_accumulation_steps)
        lora_alpha = config.getint("lora_alpha", lora_alpha)
        lora_dropout = config.getfloat("lora_dropout", lora_dropout)
        train_on_inputs = config.getboolean("train_on_inputs", train_on_inputs)
        add_eos_token = config.getboolean("add_eos_token", add_eos_token)
        group_by_length = config.getboolean("group_by_length", group_by_length)
        wandb_project = config.get("wandb_project", wandb_project)
        wandb_run_name = config.get("wandb_run_name", wandb_run_name)
        wandb_watch = config.get("wandb_watch", wandb_watch)
        wandb_log_model = config.get("wandb_log_model", wandb_log_model)
        resume_from_checkpoint = config.get("resume_from_checkpoint", resume_from_checkpoint)
        prompt_template_name = config.get("prompt_template_name", prompt_template_name)
        prompt_template_description = config.get("prompt_template_description", prompt_template_description)

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )

    tokenizer.padding_side = "left"  # Allow batched inference
    def tokenize(prompt, add_eos_token=True):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
            data_point["table_metadata_string"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ] 
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data_raw = train_val["test"]
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    model.push_to_hub(repo_id='yashxdefog-sqlcoder',token=HF_TOKEN)

    BASELINE = load_baseline_coder()

    print("Comparing the baseline and finetuned model ... saving responses")

    if evaluate_generations:
        for data_point in tqdm(val_data_raw):
            instruction = data_point["instruction"]
            _,response_base = generate_query(data_point, BASELINE)
            _,response_finetuned = generate_query(data_point, model)
            response_gpt = query_gpt_model(data_point["instruction"],data_point["table_metadata_string"])
            evalgpt, rating =  validate_sql_using_gpt(response_finetuned,data_point["table_metadata_string"])

            result = {
                "instruction": instruction,
                "response_baseline": response_base,
                "response_finetuned": response_finetuned,
                "response_gpt":response_gpt,
                "evaluation_gpt" : evalgpt,
                "rating_gpt": rating
            }

            with open("comparing_base_and_finetuned.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
        

if __name__ == "__main__":
    fire.Fire(train)
