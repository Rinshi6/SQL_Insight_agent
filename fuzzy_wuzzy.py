import pandas as pd
from sqlalchemy import create_engine,  text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, Float, String
from rapidfuzz import process, fuzz

engine = create_engine('mysql+mysqlconnector://root:admin123@localhost/txt2sql')


def get_best_fuzzy_match(input_value, choices):

    match, score, _ = process.extractOne(input_value, choices, scorer=fuzz.token_set_ratio)
    return match, score



def get_values(table_name, column_name):

    # SQL query to get distinct values
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"

    # Execute query and load results into DataFrame
    df = pd.read_sql(query, con=engine)

    # Optionally, convert to list if you want raw values
    unique_values = df[column_name].dropna().tolist()

    return unique_values


def call_match(val):
    final = []

    if isinstance(val, str):
        print("Warning: input 'val' is a string, expected a filter list. Skipping fuzzy match.")
        return final

    if isinstance(val, dict):
        print("Warning: input 'val' is a dict, expected a filter list. Skipping fuzzy match.")
        return final

    if not isinstance(val, (list, tuple)) or len(val) == 0:
        print("Warning: input 'val' is empty or not a list/tuple.")
        return final

    # If the agent output includes a leading decision token like ["yes", [...], ...]
    if isinstance(val[0], str) and val[0].lower() in ("yes", "no"):
        if val[0].lower() == "no":
            return final
        items = val[1:]
    else:
        items = val

    for item in items:
        table = None
        column = None
        values = None

        if isinstance(item, dict):
            table = item.get("table") or item.get("table_name")
            column = item.get("column")
            values = item.get("values") or item.get("filter_value") or item.get("filter_values")
        elif isinstance(item, (list, tuple)):
            if len(item) < 3:
                print(f"Skipping invalid item (expected at least 3 values): {item}")
                continue
            table, column, values = item[0], item[1], item[2]
        else:
            print(f"Skipping unsupported filter item type: {type(item).__name__}")
            continue

        if not table or not column or values is None:
            print(f"Skipping incomplete filter item: {item}")
            continue

        if isinstance(values, str):
            str_lst = [i.strip() for i in values.split(",") if i.strip()]
        elif isinstance(values, (list, tuple)):
            str_lst = [str(i).strip() for i in values if str(i).strip()]
        else:
            str_lst = [str(values).strip()]

        if not str_lst:
            continue

        try:
            unq_col_val = get_values(table, column)
        except Exception as exc:
            print(f"Failed to load values for {table}.{column}: {exc}")
            continue

        if not unq_col_val:
            continue

        unq_col_val = [str(i) for i in unq_col_val]

        for subval in str_lst:
            best_match, score = get_best_fuzzy_match(subval, unq_col_val)
            final.append(["table name:" + table, "column_name:" + column, "filter_value:" + best_match])

    return final
