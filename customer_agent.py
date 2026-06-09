import os
import pickle
import re
from enum import StrEnum
from typing import List, Dict, Any, TypedDict, Annotated
from operator import add

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

load_dotenv(override=True)

# ==========================================
# 1. INITIALIZE DATA & METADATA
# ==========================================
# Fallback dictionary for testing if kb.pkl is missing or incomplete
DEFAULT_KB = {
    "customer": ["Details about customers and sellers", ["customer_id", "customer_state", "seller_id", "seller_state"]],
    "orders": ["Details about orders, payments, and reviews", ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "review_score"]],
    "product": ["Details about products and categories", ["product_id", "product_category_name", "price"]]
}

try:
    with open('kb.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
except FileNotFoundError:
    print("Warning: 'kb.pkl' not found. Using default mock metadata.")
    loaded_dict = DEFAULT_KB

# Mapping reference
d_store = {
    "customer": ['customer', 'sellers'],
    "orders": ['order_items', 'order_payments', 'order_reviews', 'orders'],
    "product": ["products", "category_translation"]
}

# ==========================================
# 2. DEFINE NATIVE PYDANTIC CLASSES FOR OUTPUTS
# ==========================================
class CategoryEnum(StrEnum):
    CUSTOMER = "customer"
    ORDERS = "orders"
    PRODUCT = "product"

# --- Schemas for Step 1: Subquestion Routing ---
class SubQuestionRoute(BaseModel):
    sub_question: str = Field(description="A distinct sub-question extracted from the main query.")
    assigned_table: CategoryEnum = Field(description="The exact database agent category mapping to this specific sub-question.")

class MultiRouterOutput(BaseModel):
    # This field absorbs the model's natural thinking process to avoid JSON formatting breaks
    thought_process: str = Field(description="Step-by-step reasoning explaining why these subquestions and target domains were chosen.")
    analysis: List[SubQuestionRoute] = Field(description="List of all isolated sub-questions paired with their target domains.")

# --- Schemas for Step 2: Column Selection ---
class ColumnSelection(BaseModel):
    selected_columns: List[str] = Field(description="List of strictly relevant column names needed to answer the question.")

# ==========================================
# 3. DEFINE LANGGRAPH STATE STRUCTURE
# ==========================================
class overallstate(TypedDict):
    user_query: str
    table_lst: list[str]
    table_extract: Annotated[list[list[str]], add]  # Format: [[sub_question, table_name], ...]
    column_extract: Annotated[list[list[str]], add] # Format: [[metadata, details], ...]

# ==========================================
# 4. INITIALIZE CHATGROQ MODEL
# ==========================================
model = ChatGroq(temperature=0, model_name='meta-llama/llama-4-scout-17b-16e-instruct')

# ==========================================
# 5. SUBQUESTION ROUTING CHAIN
# ==========================================
template_subquestion = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent text-to-SQL query router. Deconstruct the user query into precise "
        "sub-questions. Match each sub-question with its most relevant target database domain from the provided table metadata.\n\n"
        "CRITICAL: Do not output any chat filler, introduction, markdown headers, or standalone paragraphs outside the function call tool wrapper. "
        "Populate your thinking inside the 'thought_process' field of the schema, then instantly populate the structure."
    )),
    ("user", "User Query: {user_query}\n\nAvailable Tables/Metadata:\n{tables}")
])

structured_subquestion_llm = model.with_structured_output(MultiRouterOutput)
chain_subquestion = template_subquestion | structured_subquestion_llm

def solve_subquestion(q: str, lst: list) -> list:
    final = []
    for tab in lst:
        # Guard clause: Ensure list key matches pickle metadata formatting
        if tab not in loaded_dict:
            continue
        desc = loaded_dict[tab][0] if isinstance(loaded_dict[tab], list) else loaded_dict[tab]
        final.append([tab, desc])
    
    result_dict = {item[0]: item[1] for item in final}

    response_object = chain_subquestion.invoke({
        "tables": str(result_dict), 
        "user_query": q
    })
    
    return [[item.sub_question, item.assigned_table.value] for item in response_object.analysis]

# ==========================================
# 6. COLUMN SELECTION CHAIN
# ==========================================
template_column = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert database column selector. Look at the primary user intent, "
        "the isolated sub-query, and the available table columns. Return only the column "
        "names that are strictly necessary to compose the SQL query.\n\n"
        "CRITICAL: Do not output any chat filler. Return ONLY the structural JSON format tool payload matching the schema."
    )),
    ("user", "Main Question: {main_question}\nSub-Query: {query}\nAvailable Columns: {columns}")
])

structured_column_llm = model.with_structured_output(ColumnSelection)
chain_column_extractor = template_column | structured_column_llm

def solve_column_selection(main_q: str, list_sub: list) -> list:
    final_col = []
    
    for tab in list_sub:
        if not tab or len(tab) < 2:
            continue
        
        question = tab[0]
        table_name = tab[1]
        
        if table_name not in loaded_dict:
            continue
            
        columns = loaded_dict[table_name][1] if len(loaded_dict[table_name]) > 1 else loaded_dict[table_name]
        
        response_object = chain_column_extractor.invoke({
            "columns": str(columns), 
            "query": question, 
            "main_question": main_q
        })
        
        for col_name in response_object.selected_columns:
            new_col = [f"name of table:{table_name}", col_name]
            final_col.append(new_col)
            
    return final_col

# ==========================================
# 7. LANGGRAPH NODES
# ==========================================
def sq_node(state: overallstate):
    q = state['user_query']
    lst = state['table_lst']
    
    extracted_data = solve_subquestion(q, lst)
    return {"table_extract": extracted_data}

def column_node(state: overallstate):
    subq = state['table_extract']
    mq = state['user_query']
    
    extracted_columns = solve_column_selection(mq, subq)
    return {"column_extract": extracted_columns}

# ==========================================
# 8. BUILD AND COMPILE THE LANGGRAPH
# ==========================================
builder_final = StateGraph(overallstate)
builder_final.add_node("subquestion", sq_node)
builder_final.add_node("column_e", column_node)

builder_final.add_edge(START, "subquestion")
builder_final.add_edge("subquestion", "column_e")
builder_final.add_edge("column_e", END)

graph_final = builder_final.compile()

# ==========================================
# 9. EXECUTION EXAMPLE
# ==========================================
if __name__ == "__main__":
    initial_state = {
        "user_query": "Show me customer orders for high value products ordered this week.",
        # Fixed: Changed "products" to "product" to match your Enum and d_store keys cleanly
        "table_lst": ["customer", "orders", "product"],
        "table_extract": [],
        "column_extract": []
    }
    
    print("--- Executing Text-to-SQL Extraction Graph ---")
    final_output = graph_final.invoke(initial_state)
    
    print("\n--- Final Graph State Output ---")
    import pprint
    pprint.pprint(final_output)
