import pandas as pd
import numpy as np
import re


def load_data(path):
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            raise ValueError("Format file tidak didukung (harus .csv atau .xlsx)")
        return df
    except Exception as e:
        print(f"Error saat membaca file: {e}")
        return None

def standardize_column_names(df):
    rename_mapping = {
        'DB Type': 'db_type',
        'Database Server Name': 'database_server_name',
        'Source_System-DatabaseName': 'source_system_database_name',
        'Schema_Name': 'schema_name',
        'Source_Table': 'source_table',
        'PK_Column_Name': 'pk_column_name',
        'PK_Data_Type': 'pk_data_type',
        'Watermark_Column_Name': 'watermark_column_name',
        'Watermark_Data_Type': 'watermark_data_type',
        'is_watermark_null?': 'is_watermark_null',
        'Table_Size': 'table_size',
        'Type of Table': 'type_of_table',
        'Extraction_Mode': 'extraction_mode'
    }

    df = df.rename(columns=rename_mapping)
    return df


def replace_like_null(df):
    return df.apply(lambda col: col.map(
        lambda x: None if (isinstance(x, str) and 'NULL' in x.upper()) else x
    ))


def clean_watermark_column_name(df, col='watermark_column_name'):
    if col in df.columns:
        def clean_val(x):
            if pd.isnull(x):
                return x
            x_str = str(x).strip()
            main_name = re.match(r'^[^\(]+', x_str)
            main_name = main_name.group(0).strip() if main_name else ''        
            inside_paren = re.findall(r'\((.*?)\)', x_str)
            inside_names = [s.split()[0] for s in inside_paren]
            all_names = [main_name] + inside_names if inside_names else [main_name]
            return ', '.join(all_names)
        
        df[col] = df[col].apply(clean_val)
    return df


def clean_special_chars(df):
    def clean_val(x):
        if isinstance(x, str):
            x = x.strip()
            x = re.sub(r'[^\w\s,]', '', x)
            x = re.sub(r',\s*$', '', x)
        return x
    return df.apply(lambda col: col.map(clean_val))


def clean_pk_column(df, col='pk_column_name'):
    if col in df.columns:
        df[col] = df[col].apply(lambda x: ','.join([line.strip() for line in str(x).splitlines()]) if pd.notnull(x) else x)
    return df


def clean_table_size(df, col='table_size'):
    if col in df.columns:
        def to_int(x):
            if pd.isnull(x):
                return x
            x_str = str(x)
            x_str = x_str.replace(',', '').replace('.', '')
            try:
                return int(x_str)
            except ValueError:
                return None  
        df[col] = df[col].apply(to_int)
    return df


def validate_data_types(df):
    for i, col in enumerate(df.columns):
        if i < 10:  
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, str) else None)
        elif i == 10:  
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif i in [11, 12]:  
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, str) else None)
    return df


def stdr_is_watermark_null(df, col='is_watermark_null'):
    if col in df.columns:
        df[col] = df[col].map({'Y': 'Yes', 'N': 'No'}).fillna(df[col])
    return df


def clean_watermark_data_type(df, col='watermark_data_type'):
    if col in df.columns:
        def map_type(x):
            if pd.isnull(x):
                return x
            x_lower = str(x).lower()
            if 'date' in x_lower or 'timestamp' in x_lower:
                return 'datetime'
            elif 'int' in x_lower:
                return 'integer'
            return x
        df[col] = df[col].apply(map_type)
    return df


def clean_pk_data_type(df, col='pk_data_type'):
    if col in df.columns:
        def map_pk(x):
            if pd.isnull(x):
                return x
            x_lower = str(x).lower()
            if ',' in x_lower:
                return 'composite key (varchar)'
            elif 'serial' in x_lower:
                return 'serial'
            elif 'int' in x_lower:
                return 'integer'
            return x
        df[col] = df[col].apply(map_pk)
    return df

def determine_extraction_type(row):
    pk_col = row.get('pk_column_name')
    pk_dtype = row.get('pk_data_type')
    wmk_col = row.get('watermark_column_name')
    wmk_dtype = row.get('watermark_data_type')
    is_wmk_null = row.get('is_watermark_null')
    
    if pd.notnull(pk_col) and pd.notnull(wmk_col):
        if is_wmk_null == 'No':
            return "incremental with watermark datetime"
        elif is_wmk_null == 'Yes': 
            return "customquery_find other watermark column"
    
    if pd.isnull(pk_col) and pd.notnull(wmk_col):
        if is_wmk_null == 'No':
            return "customquery_composite key"
        elif is_wmk_null == 'Yes':
            return "customquery_composite key and other watermark column"
        
    if pd.isnull(wmk_col) and pk_dtype == 'integer':
        return "incremental with watermark primarykey integer"
    
    if pd.isnull(wmk_col) and (pk_dtype != 'integer' or pd.isnull(pk_dtype)):
        return "possible full load"
    
    if pd.isnull(pk_col) and pd.isnull(wmk_col):
        return "possible full load"
    
    return "other" 


def ingestion_pipeline(file_path):
    df = load_data(file_path)
    df = standardize_column_names(df)
    df = replace_like_null(df)
    df = clean_watermark_column_name(df)
    df = clean_special_chars(df)
    df = clean_pk_column(df)
    df = clean_table_size(df)
    df = validate_data_types(df)
    df = stdr_is_watermark_null(df)
    df = clean_watermark_data_type(df)
    df = clean_pk_data_type(df)

    df['extraction_type_rules'] = df.apply(determine_extraction_type, axis=1)

    required_cols = [
        'pk_column_name', 'pk_data_type',
        'watermark_column_name', 'watermark_data_type',
        'is_watermark_null', 'table_size', 'type_of_table', 'extraction_mode'
    ]
    
    valid_mask = df[required_cols].notnull().any(axis=1)
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()

    print(f"Total records: {len(df)}, Valid: {len(df_valid)}, Invalid: {len(df_invalid)}")

    return df_valid, df_invalid


import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, 'data')

input_file = os.path.join(data_dir, 'DataGlossary.xlsx')
df_clean, df_invalid = ingestion_pipeline(input_file)
print(df_clean.head())

output_path = os.path.join(data_dir, 'DataGlossary_clean.xlsx')

print('done')
df_clean.to_excel(output_path, index=False)
print('done weite')