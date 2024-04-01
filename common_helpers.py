import json
import jsonlines
import csv
import sqlglot
import re
from datetime import datetime

def open_csv_as_string(filename):
    lines = []
    with open(filename, 'r') as file:
        # Read and print each line
        for line in file:
            lines.append(line.strip())
    # join
    return '\n'.join(lines)
    
def csv_to_string_alt(file_path):
    """
    Reads a CSV file and converts it into a string.
    
    :param file_path: Path to the CSV file
    :return: A string representation of the CSV data
    """
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        csv_string = ''
        for row in reader:
            csv_string += ','.join(row) + '\n'
    
    return csv_string

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data.append(line)
    return data


def write_jsonl(file_path, data):
    with jsonlines.open(file_path, 'w') as writer:
        for d in data:
            writer.write(d)

def read_txt(file_path='output_2024-03-31-19-11-13.txt'):
    with open(file_path, 'r') as f:
        s = f.read()
    
    preprocess_output(s)

def validate_sql_generation(sentence):
    print("Validating SQL: ", sentence)
    try:
        sqlglot.transpile(sentence)
        
    except Exception as e:
        print(e)
        print(f"Error in parsing SQL: {sentence}")
        return 0
    return 1
def preprocess_output(output, system_prompt, user_prompt):
    """
    Preprocess the output from the GPT model by removing any leading/trailing whitespaces and newlines.
    
    :param output: The output from the GPT model
    :return: The preprocessed output

    """
    # pattern = r"(?:User Question|Question):\s*(.*?)\n (?:SQL|SQL Code):\s*```(?:sql)?\s*(.*?)\s*```"
    pattern = r"(?:User Question|Question):\s*(.*?)\n(?:SQL:|SQL Code:|SQL code:)\s*```(?:sql)?\s*(.*?)\s*```"

    match = re.findall(pattern, output, re.DOTALL)
    qa_pairs = []

    lines = output.split('#')
    count_valid = 0
    for question, sql_query in match:
        # print(question)
        # print(sql_query)
        if validate_sql_generation(str(sql_query)) == True:
            count_valid += 1
        val = validate_sql_generation(sql_query)
        qa_pair = [{'question': question, 'sql': sql_query, 'valid': val }]
        qa_pairs.append(qa_pair)

    print(" --- ")
    print("Percentage of valid SQL queries: ")
    print(count_valid/len(match)*100)
    print(" --- ")

    qa_pairs = [{'system_prompt': system_prompt, 'user_prompt': user_prompt}] + qa_pairs

    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    write_jsonl(f'qa_pairs_{time_stamp}.jsonl', qa_pairs)

    return qa_pairs


if __name__ == '__main__':
    read_txt()
    # csv_to_string_alt('data.csv')
    # read_jsonl('prompt_dict.jsonl')
    # write_jsonl('prompt_dict.jsonl', [{'sys': 'sys', '