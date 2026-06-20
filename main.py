from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, TypedDict, Annotated
from operator import add
import pickle
from IPython.display import Image
import importlib

from router_agent import agent_2
from customer_agent import graph_final
from langgraph.constants import Send

import customer_helper
from customer_helper import chain_filter_extractor, chain_query_extractor, chain_query_validator
from fuzzy_wuzzy import call_match
from datetime import datetime
import json
import tqdm
import time


import pandas as pd
from sqlalchemy import create_engine,  text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import Integer, Float, String
from mlflow_setup import (
    is_mlflow_enabled,
    log_json_artifact,
    log_text_artifact,
    observe_node,
    setup_mlflow,
)

setup_mlflow()

d_store = {
    "customer" : ['customer', 'sellers'],
    "orders" : ['order_items', 'order_payments', 'order_reviews', 'orders'],
    "product": ["products", "category_translation"]
}


with open('kb.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

engine = create_engine('mysql+mysqlconnector://root:admin123@localhost/txt2sql')


def remove_duplicates(f):
    s = set()
    final = []
    for k, v in f.items():
        if k in ('cust_out', 'order_out', 'product_out'):
            for item in v['column_extract']:
                key = tuple(item)
                if key not in s:
                    final.append(item)
                    s.add(key)
    return final


class finalstate(TypedDict):
    user_query: str
    router_out: list[str]
    cust_out: str
    order_out: str
    product_out: str
    filtered_col : str
    filter_extractor: list[str]
    fuzz_match: list[str]
    sql_query: str
    final_query: str

@observe_node("router")
def router(state: finalstate):
    q = state['user_query']
    o = agent_2(q)
    return {"router_out": o}

def route_request(state: finalstate):
    routes = state['router_out']
    print("Routed request to"+str(routes)+' agents')
    return routes

def filter_condition(state: finalstate):
    if len(state['filter_extractor'])==1:
        return "no"
    else:
        return "yes"

@observe_node("customer")
def customer(state: finalstate):
    q = state['user_query']
    print("Extracting relavant tables and columns from customer agent................")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store['customer']})
    return {"cust_out": sub}

@observe_node("orders")
def orders(state: finalstate):
    q = state['user_query']
    print("Extracting relavant tables and columns from orders agent................")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store['orders']})
    return {"order_out": sub}

@observe_node("product")
def product(state: finalstate):
    q = state['user_query']
    print(q)
    
    print("Extracting relavant tables and columns from product agent................")
    sub = graph_final.invoke({"user_query": q, "table_lst": d_store['product']})
    print(sub)
    return {"product_out": sub}

@observe_node("filter_check")
def filter_check(state: finalstate):
    q = state['user_query']
    f = {}
    col_f = []
    for key in ['order_out', 'cust_out', 'product_out']:
        if key in state:
            f[key] = state.get(key)
            col_f.append(state[key])
    col_details = remove_duplicates(f)
    print("Checking the need for filter................")
    response = chain_filter_extractor.invoke({"columns": str(col_details), "query": q}).filter_needed
    return {'filter_extractor': response, 'filtered_col': str(col_details)}

@observe_node("fuzz_match")
def fuzz_match(state: finalstate):
    val = state['filter_extractor']
    print("Solving for getting right filter values.........")
    lst = call_match(val)
    print("done filtering...........................")
    return {"fuzz_match": lst}

@observe_node("query_generation")
def query_generation(state: finalstate):
    q = state['user_query']
    tab_cols = state['filtered_col']
    if state.get('fuzz_match'):
        filters = state.get('fuzz_match')
    else:
        filters = ''
    print("Generating SQL query.........")
    final_query = chain_query_extractor.invoke({"columns": tab_cols, "query": q, "filters": filters})
    return {"sql_query": final_query}

@observe_node("query_validation")
def query_validation(state: finalstate):
    print("validating and generating final query........")
    o = chain_query_validator.invoke({"columns": state['filtered_col'], "query": state['user_query'], "filters": state.get('fuzz_match'), 'sql_query':state['sql_query']})
    return {'final_query': o}

def graph_sql(state: finalstate):
     if not is_mlflow_enabled():
          return _invoke_graph_sql(state)

     import mlflow

     run_name = f"txt2sql-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
     started_at = time.perf_counter()

     with mlflow.start_run(run_name=run_name):
          mlflow.log_param("user_query", state.get("user_query", ""))
          mlflow.log_param("tracking_backend", "postgresql")
          log_json_artifact("input/state.json", state)

          try:
               result = _invoke_graph_sql(state)
          except Exception as exc:
               mlflow.log_param("status", "error")
               mlflow.log_metric("total_duration_seconds", time.perf_counter() - started_at)
               log_text_artifact("error.txt", repr(exc))
               raise

          mlflow.log_param("status", "success")
          mlflow.log_metric("total_duration_seconds", time.perf_counter() - started_at)
          log_json_artifact("output/final_state.json", result)

          if result.get("sql_query"):
               log_json_artifact("output/generated_sql.json", result["sql_query"])
          if result.get("final_query"):
               log_json_artifact("output/validated_sql.json", result["final_query"])

          return result


def _invoke_graph_sql(state: finalstate):
     builder_final = StateGraph(finalstate)

     builder_final.add_node("router", router)  # Add explicit node names

     builder_final.add_node("customer", customer)
     builder_final.add_node("orders", orders)
     builder_final.add_node("product", product)

     builder_final.add_node("filter_check", filter_check)
     builder_final.add_node("fuzz_filter", fuzz_match)
     builder_final.add_node("query_generator", query_generation)
     builder_final.add_node("query_validation", query_validation)

     builder_final.add_edge(START, "router")

     builder_final.add_conditional_edges("router", route_request, ["customer", "orders", "product"])

     builder_final.add_edge("customer", "filter_check")
     builder_final.add_edge("orders", "filter_check")
     builder_final.add_edge("product", "filter_check")

     builder_final.add_conditional_edges(
     "filter_check",
     filter_condition,
     {
          "no": "query_generator",
          "yes": "fuzz_filter"
     }
     )

     builder_final.add_edge("fuzz_filter", "query_generator")

     builder_final.add_edge("query_generator", "query_validation")

     builder_final.add_edge("query_validation", END)

     graph_main = builder_final.compile()
     return graph_main.invoke(state)


if(__name__ == "__main__"):
    q = 'Give me list of customers from  São Paulo state that made atleast 1 payment through credit card'

    f = graph_sql({"user_query": q})
    print(f)
