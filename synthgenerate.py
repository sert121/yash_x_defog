from datetime import datetime
from openai import OpenAI
import os
import pandas as pd
import json
from io import StringIO
from dotenv import load_dotenv
import jsonlines
from common_helpers import open_csv_as_string, preprocess_output, combine_json_files, dump_json
import argparse    

import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import sqlglot
import random

load_dotenv()

import openai
import jsonlines

api_key = os.getenv("OPENAI_API_KEY")
PROMPT_VERSION = "v1" # change to use a different version
openai.api_key = api_key

# Check if global_promptdict exists as a JSON file
if os.path.exists('global_promptdict.json'):
    with open('global_promptdict.json', 'r') as file:
        global_promptdict = json.load(file)
else:
    global_promptdict = {
        "version":"v1",
    "sys":'''You are an expert and thoughtful SQL analyst and understand data very well. 
    You have a two-step role: You need to come up with potential user questions that could be asked about the data. 
    Now, based on the questions you have come up with, you need to write SQL queries that would help answer those questions.

    Always follow these instructions for generating the user-question:
    1. Make sure that the questions are relevant to the data and are not ambiguous.
    2. Make sure that the questions are human-like.
    3. Make sure the question is not too confusing.

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

    As a reference, to generate a valid user question and corresponding SQL you can use the following examples:
    {examples}''',
    "user": '''You need to generate 10 instances of (user_question, sql_code). The database schema is represented in the following CSV string:
    ```{table_metadata_ddl}```
    Follow the same format as the examples to produce the user question and SQL code. Separate each genrated pair with a separator #.''',
    }
    
    self_ask_prompts = {"version":"self_ask_v1",
    "sys":'''You are an expert and thoughtful SQL analyst and understand data very well. 
    You have a two-step role: You need to come up with potential user questions that could be asked about the data. 
    Now, based on the questions you have come up with, you need to write SQL queries that would help answer those questions.

    Always follow these instructions for generating the user-question:
    1. Make sure that the questions are relevant to the data and are not ambiguous.
    2. Make sure that the questions are human-like.
    3. Make sure the question is not too confusing.

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

    As a reference, to generate a valid user question and corresponding SQL you can check the query and answer asked by the user:
    {examples}''',
    "user":'''You need to follow up with another question and corresponding sql code in the form of  (user_question, sql_code). The database schema is represented in the following CSV string:
    ```{table_metadata_ddl}```. Only generate one follow up question and SQL code.
    Follow the same format as the previous conversation user_question and sql to produce the user question and SQL code. Separate each genrated pair with a separator #.''',}



client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key, 

)


def answer_query_gpt(system_prompt='', user_prompt=''):
    """Answers a query based on the prompt
    Args:
        sentence (str) : Prompt to answer
    Returns:
        label (str) : Answer to the query
    """
	
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
    return label


def execute_sql_query(db_file, query):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    try:
        # Execute the SQL query
        cursor.execute(query)

        # If it's a SELECT query, fetch and return the results
        if query.strip().upper().startswith('SELECT'):
            results = cursor.fetchall()
            return results
        else:
            # For non-SELECT queries, commit changes and return row count
            conn.commit()
            return cursor.rowcount
    except Exception as e:
        # Print any error that occurs
        print(f"An error occurred: {e}")
        return None
    finally:
        # Close the database connection
        conn.close()



def filter_csv(df):
    print(df.columns)
    df.dropna(subset=['column_description'], inplace=True)
    rows_to_drop = df[df['column_description'].str.contains("DO NOT")]
    df.drop(rows_to_drop.index, inplace=True)
    return df

def synth_gen(examples='',metadata_string = '',method = 'k_shot_generation', flag = '', k_shot=5):
    qa_pairs = ''
    if method == 'k_shot_generation':            
        # sort by length of sql
        example_set = []

        for index, row in examples.iterrows():
            example_set.append((row['question'],row['sql']))
        
        example_strings = []
        # random choice
        subset = random.sample(example_set, k_shot)
        for question, sql in subset:
            temp = f"Question: {question} \nSQL: ```{sql}```"
            example_strings.append(temp)

        example_strings = "\n\n".join(example_strings) # join the examples via new line

        system_prompt = global_promptdict['sys']
        user_prompt = global_promptdict['user']

        sys = system_prompt.format(examples=example_strings)
        user =  user_prompt.format(table_metadata_ddl=metadata_string,user_question_1 = '{user_question_1}', sql_code_1 = '{sql_code_1}', user_question_2 = '{user_question_2}', sql_code_2 = '{sql_code_2}')


        qa_pairs = answer_query_gpt(system_prompt=sys, user_prompt=user)
        # save a in text file
        d = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open(f"{args.output_txt_dir}/output_{flag}_{d}.txt", "w") as f:
            f.write(qa_pairs)

        preprocess_output(qa_pairs, system_prompt, user_prompt, flag)

    if method == 'follow_up_generation':
        # follow up generation
        
        example_set = []

        for index, row in examples.iterrows():
            example_set.append((row['question'],row['sql']))
        
        example_strings = []
        # random choice
        subset = random.sample(example_set, 3)
        for question, sql in subset:
            temp = f"Question: {question} \nSQL: ```{sql}```"
            example_strings.append(temp)

        example_strings = "\n\n".join(example_strings) # join the examples via new line

        system_prompt = self_ask_prompts['sys']
        user_prompt = self_ask_prompts['user']

        sys = system_prompt.format(examples=example_strings)
        user =  user_prompt.format(table_metadata_ddl=metadata_string,user_question_1 = '{user_question_1}', sql_code_1 = '{sql_code_1}', user_question_2 = '{user_question_2}', sql_code_2 = '{sql_code_2}')

        qa_pairs = answer_query_gpt(system_prompt=sys, user_prompt=user)
        # save a in text file
        d = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open(f"output_{flag}_{d}.txt", "w") as f:
            f.write(qa_pairs)

        preprocess_output(qa_pairs, system_prompt, user_prompt, flag)


    return qa_pairs



def csv_preprocess(csv_file, db_file, table_name, turn_to_sql=False):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file)
    df = filter_csv(df)
    return df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Synthetic Data Generation Script")

    # Adding the arguments
    parser.add_argument('--samples', type=int, default=250, help='Number of samples')
    parser.add_argument('--method', default='k_shot_generation', help='k_shot_generation or follow_up_generation')
    parser.add_argument('--output_path', type=str, default='./qa_collection/combined_v3.json', help='data output path')
    parser.add_argument('--k_shot_flag',default=5, help='k-shot flag')
    parser.add_argument('--input_examples', type=str, default='macmillan_golden_queries.csv', help='input examples')
    parser.add_argument('--metadata', type=str, default='macmillan_md.csv', help='metadata file')
    parser.add_argument('--metadata_db', type=str, default='macmillan_md.db', help='temporary metadata db file')
    parser.add_argument('--prompt_version', type=str, default='v1', help='prompt version')
    parser.add_argument('--output_txt_dir', type=str, default='./gpt_raw_responses/', help='txt files to collect gpt responses')
    
    # Parse the arguments
    args = parser.parse_args()


    examples = pd.read_csv('macmillan_golden_queries.csv') # read the golden queries/ few shot examples
    examples = pd.read_csv(args.input_examples) # read the golden queries/ few shot examples
    # preprocess the golden queries 
    metadata_df = csv_preprocess('macmillan_md.csv', 'macmillan_md.db', 'macmillan_md', turn_to_sql=False)
    metadata_df = csv_preprocess(args.metadata, args.metadata_db, 'macmillan_md', turn_to_sql=False)

    metadata_df.to_csv("macmillan_filtered_md.csv", index=False)

    # read the metadata as a string
    metadata_df_string = open_csv_as_string('macmillan_filtered_md.csv')
    
    # generate synthetic data [main]
    if args.method == 'k_shot_generation':  

        c = args.samples // 10
        k = args.k_shot_flag
        for flag in range(1, c): # num of samples : 25 x 10 = 250
            synth_gen(examples=examples, metadata_string=metadata_df_string,k_shot=k,flag=str(flag)+f"_{k}shot_")
    if args.method == 'follow_up_generation':
        # follow up generation
        for flag in range(1, args.samples): 
            synth_gen(examples=examples, metadata_string=metadata_df_string,method='follow_up_generation',flag=str(flag)+f"_followup_")
 
    # combine the generated samples
    combined_data = combine_json_files()
    dump_json([d[0] for d in combined_data], args.output_path)
