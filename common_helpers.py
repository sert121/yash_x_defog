import json
import jsonlines
import csv
import sqlglot
import re
import datetime
import pandas as pd
import sqlite3, random
import string


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

def preprocess_output(output, system_prompt, user_prompt, flag=''):
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
    write_jsonl(f'qa_pairs_{time_stamp}_{flag}.jsonl', qa_pairs)

    return qa_pairs


def dtype_mapping(data_type):
    return {
        'character varying': 'TEXT',
        'integer': 'INTEGER',
        'date': 'DATE',
        'numeric': 'NUMERIC',
        'bigint': 'BIGINT',
        'boolean': 'BOOLEAN',
        'real': 'FLOAT',
    }.get(data_type) 


def create_table_from_df(df):
    sql_statements = []
    for _, row in df.iterrows():
        table_name = row['table_name']
        column_name = row['column_name']
        data_type = dtype_mapping(row['data_type'])
        column_description = row['column_description']
        
        column_statement = f"{column_name} {data_type}"

        # Check for possible values in the column description
        if ": " in column_description:
            possible_values_part = column_description.split(": ")[1]
            # Handle cases where possible values are listed and replace '-' with '_'
            possible_values = [value.strip().replace('-', '_') for value in possible_values_part.split(", ")]
            possible_values_formatted = ', '.join(f"'{value}'" for value in possible_values)
            column_statement += f" CHECK ({column_name} IN ({possible_values_formatted}))"
        
        sql_statements.append(column_statement)
    
    create_table_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(sql_statements) + "\n);"
    print(create_table_sql)
    return create_table_sql

def run_sql_query(sql_query):
    # Connect to the SQLite in-memory database
    conn = sqlite3.connect(':memory:')
    # conn = sqlite3.connect('test.db')

    # Create a cursor object using the cursor() method
    cur = conn.cursor() 
    cur.execute(sql_query)

    # Commit the changes to the database
    conn.commit()
    cur.execute("SELECT * FROM temp_achive_lp_sapling_combined_activations_ai;")

    return cur

# Define function to generate random string for TEXT fields
def random_string(prefix='', length=10, suffix=''):
    letters = string.ascii_letters
    return f"{prefix}{''.join(random.choice(letters) for _ in range(length))}{suffix}"

# Function to generate a random date between two dates
def random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + datetime.timedelta(days=random_number_of_days)

# Function to generate random data for the given table structure
def generate_random_data_for_table(table_name, num_records):
    random_data = []
    start_date = datetime.date(2015, 1, 1)
    end_date = datetime.date(2023, 12, 31)
    for _ in range(num_records):
        record = {
            'unique_id': random_string(length=8),
            'copyright_year': random.randint(1990, 2023),
            'year': random.randint(1990, 2023),
            'month': random.randint(1, 12),
            'purchased_from_studentstore': random.choice(['No', 'Yes_Purchase']),
            'inclusive_access': random.choice(['No', 'Yes']),
            'has_used_access_code': random.choice(['No', 'Yes']),
            'student_subscription_status': random.choice(['Active Access', 'Expired Access']),
            'product_status': random.choice(['REB', 'NAB', 'OP']),
            'account_manager_territory': random_string(),
            'acquisition_source': random_string(),
            'division': random_string(),
            'territory': random_string(),
            'region': random.choice(['Northeast Region', 'Southern Region', 'Rocky Mountain Region', 'Central Region']),
            'product_publisher': random_string(),
            'sfdc_state': random_string(),
            'discipline': random_string(),
            'category': random_string(),
            'product_sub_discipline': random_string(),
            'region_territory': random_string(),
            'instructor_email': random_string(suffix='@example.com', length=5),
            'instructor_name': random_string(length=15),
            'profit_center_desc': random_string(),
            'hs_col': random_string(),
            'author_display_name': random_string(),
            'project_title': random_string(),
            'platform': random_string(),
            'school': random_string(),
            'sfdc_country': random_string(),
            'sfdc_city': random_string(),
            'sfdc_account_name': random_string(),
            'paid_unpaid': random.choice(['Paid', 'Unpaid']),
            'course_name': random_string(),
            'course': random_string(),
            'course_id': random_string(length=8),
            'product_isbn': random_string(length=13),
            'isbn': random_string(length=13),
            'subscription_created_date': random_date(start_date, end_date),
            'subs_start_date': random_date(start_date, end_date),
            'subs_end_date': random_date(start_date, end_date),
            'return_request_date': random_date(start_date, end_date),
            'student_subscription_expiration_date': random_date(start_date, end_date),
            'conversion_date': random_date(start_date, end_date),
            'course_end_date': random_date(start_date, end_date),
            'course_start_date': random_date(start_date, end_date),
            'course_created_date': random_date(start_date, end_date),
            'access_duration': round(random.uniform(0.0, 100.0), 2),
            'access_duration_derived': random.randint(1, 1000),
            'us_net_price': round(random.uniform(0.0, 1000.0), 2),
            'us_list_price': round(random.uniform(0.0, 1000.0), 2),
            'us_consumer_price': round(random.uniform(0.0, 1000.0), 2),
            'can_net_price': round(random.uniform(0.0, 1000.0), 2),
            'can_list_price': round(random.uniform(0.0, 1000.0), 2),
            'can_consumer_price': round(random.uniform(0.0, 1000.0), 2),
        }
        random_data.append(record)
    return random_data



def insert_random_data(cur, table_name, records):
    for i in range(len(records)):
        cur.execute(f"""
            INSERT INTO {table_name} (
                unique_id,
                copyright_year,
                year,
                month,
                purchased_from_studentstore,
                inclusive_access,
                has_used_access_code,
                student_subscription_status,
                product_status,
                account_manager_territory,
                acquisition_source,
                division,
                territory,
                region,
                product_publisher,
                sfdc_state,
                discipline,
                category,
                product_sub_discipline,
                region_territory,
                instructor_email,
                instructor_name,
                profit_center_desc,
                hs_col,
                author_display_name,
                project_title,
                platform,
                school,
                sfdc_country,
                sfdc_city,
                sfdc_account_name,
                paid_unpaid,
                course_name,
                course,
                course_id,
                product_isbn,
                isbn,
                subscription_created_date,
                subs_start_date,
                subs_end_date,
                return_request_date,
                student_subscription_expiration_date,
                conversion_date,
                course_end_date,
                course_start_date,
                course_created_date,
                access_duration,
                access_duration_derived,
                us_net_price,
                us_list_price,
                us_consumer_price,
                can_net_price,
                can_list_price,
                can_consumer_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            records[i]['unique_id'],
            records[i]['copyright_year'],
            records[i]['year'],
            records[i]['month'],
            records[i]['purchased_from_studentstore'],
            records[i]['inclusive_access'],
            records[i]['has_used_access_code'],
            records[i]['student_subscription_status'],
            records[i]['product_status'],
            records[i]['account_manager_territory'],
            records[i]['acquisition_source'],
            records[i]['division'],
            records[i]['territory'],
            records[i]['region'],
            records[i]['product_publisher'],
            records[i]['sfdc_state'],
            records[i]['discipline'],
            records[i]['category'],
            records[i]['product_sub_discipline'],
            records[i]['region_territory'],
            records[i]['instructor_email'],
            records[i]['instructor_name'],
            records[i]['profit_center_desc'],
            records[i]['hs_col'],
            records[i]['author_display_name'],
            records[i]['project_title'],
            records[i]['platform'],
            records[i]['school'],
            records[i]['sfdc_country'],
            records[i]['sfdc_city'],
            records[i]['sfdc_account_name'],
            records[i]['paid_unpaid'],
            records[i]['course_name'],
            records[i]['course'],
            records[i]['course_id'],
            records[i]['product_isbn'],
            records[i]['isbn'],
            records[i]['subscription_created_date'],
            records[i]['subs_start_date'],
            records[i]['subs_end_date'],
            records[i]['return_request_date'],
            records[i]['student_subscription_expiration_date'],
            records[i]['conversion_date'],
            records[i]['course_end_date'],
            records[i]['course_start_date'],
            records[i]['course_created_date'],
            records[i]['access_duration'],
            records[i]['access_duration_derived'],
            records[i]['us_net_price'],
            records[i]['us_list_price'],
            records[i]['us_consumer_price'],
            records[i]['can_net_price'],
            records[i]['can_list_price'],
            records[i]['can_consumer_price']
        ))

# Usage:
# Assuming `cur` is a cursor object from the sqlite3 or psycopg2 library
# and you have already created the table in your database


# Now let's create a function to insert this data into our SQLite database
def insert_data_into_table(table_name='', data=''):
    # Generate random data for the table

    filtered_df = pd.read_csv('macmillan_filtered_md.csv')
    statement = create_table_from_df(filtered_df)
    cur = run_sql_query(statement)
    table_name = 'temp_achive_lp_sapling_combined_activations_ai'

    num_records = 10  # Number of records to generate
    random_data = generate_random_data_for_table(table_name, num_records)
    
    print(len(random_data))
    print("random data is generated")
    insert_random_data(cur, 'temp_achive_lp_sapling_combined_activations_ai', random_data)
    cur.execute("SELECT * FROM temp_achive_lp_sapling_combined_activations_ai;")
    # print(len(cur.fetchall()))

    return cur

def validate_against_testdb(cur,queries):
    total_count = len(queries)
    valid_count = 0
    insert_data_into_table()
    for query in queries:
        try:
            cur.execute(query)
            print(cur.fetchall())
            valid_count += 1
        except Exception as e:
            print(e)
            print(f"Error in parsing SQL: {query}")
    
    proportion = (valid_count / total_count)*100
    return proportion
        
if __name__ == '__main__':
    # read_txt()
    # csv_to_string_alt('data.csv')
    # read_jsonl('prompt_dict.jsonl')

    # filtered_df = pd.read_csv('macmillan_filtered_md.csv')
    # statement = create_table_from_df(filtered_df)
    # run_sql_query(statement)
    # insert_data_into_table()
    # validate_against_testdb()
