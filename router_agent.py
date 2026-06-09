from enum import StrEnum
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Using the requested model
llama = 'meta-llama/llama-4-scout-17b-16e-instruct'
model = ChatGroq(temperature=0, model_name=llama) 

# CLEANED PROMPT: Removed raw string list examples and forced JSON structure
template = ChatPromptTemplate.from_messages([
    ("system", """
You are an intelligent router in a text-to-sql system that understands the user question and 
determines which agents might have the answer based on agent descriptions. Multiple agents might answer a given user question.

Your output must be a valid JSON object matching the requested schema. Do not output any explanation or conversational text.
"""),

    ("human", '''
Below are descriptions of different agents:
customer agent : It contains all the details about customer and seller locations and their unique identifiers
orders agent : It contains details about all the orders like product identifier, order identifier, products in an order, no. of items of a product in order, price of order, freight value, order time, delivery status and its time, payment etc.
product agent : It contains details about product like product identifier, product category, description, dimensions of product

STEP BY STEP TABLE SELECTION PROCESS:
1. Split the question into different subquestions.
2. For each subquestion, carefully review each AGENT description and determine which agent holds the answer.
3. Select all required agents to answer the full question.
     
User question:
{question}
''')
])

class CategoryEnum(StrEnum):
    CUSTOMER = "customer"
    ORDERS = "orders"
    PRODUCT = "product"

class CategorizedItems(BaseModel):
    tags: List[CategoryEnum] = Field(
        description="A list of matching category tags for the given text."
    )

# CHANGED: Switched back to default tool calling (remove method="json_mode")
# Groq handles Pydantic much better through tool-calling natively.
structured_llm = model.with_structured_output(CategorizedItems)
chain = template | structured_llm

def agent_2(q):
    # Returns the list of CategoryEnum values directly
    response = chain.invoke({"question": q}).tags
    return [tag.value for tag in response] 
    

if __name__ == "__main__":
    print(agent_2("Give me list of customers from São Paulo state that made atleast 1 payment through credit card"))
