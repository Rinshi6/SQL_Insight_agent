## SQL_Insight_agent
An intelligent Text-to-SQL AI agent designed to query over 100K+ Olist e-commerce records. This system automatically translates complex business analytics questions into validated SQL queries across 5 interconnected datasets, generating 10+ distinct report types. It features automated SQL validation, query refinement, and MLflow observability to track prompts, query execution, and performance metrics. 
## 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.
## 1. Clone the Repository 

git clone https://github.com/Rinshi6/SQL_Insight_agent
cd SQL_Insight_agent

## 2. Set Up Environment Variables
Create a .env file in the root directory of the project and add your API keys:

# .env example
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
KAGLE_API_TOKEN=your_kaggle_api_token_here

## 3. Install Dependencies
It is recommended to use a virtual environment to manage your packages. 

## Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
## Install the required packages
pip install -r requirements.txt

## 4. Configure MLflow Observability
MLflow is configured to write tracking data to a local PostgreSQL database named `mlflowdb`.

Create the database before running the agent:

```sql
CREATE DATABASE mlflowdb;
```

By default, the app uses:

```env
MLFLOW_TRACKING_URI=postgresql+psycopg2://postgres:postgres@localhost:5432/mlflowdb
# Optional: set this only if you want a fixed experiment name.
# If unset, the app creates txt2sql-agent-<uuid> each time.
MLFLOW_EXPERIMENT_NAME=txt2sql-agent
```

If your PostgreSQL username, password, host, or port is different, add the correct `MLFLOW_TRACKING_URI` to `.env`.

## 5. Run the Project
Launch Jupyter Lab or Jupyter Notebook to open and execute the main pipeline.

jupyter lab

Once the interface opens, navigate to and run main.ipynb.
## 🛠️ Project Structure

* main.ipynb - The primary Jupyter notebook containing the execution logic.
* requirements.txt - Python dependencies needed to run the notebook.
* .env - Configuration file for API tokens.
