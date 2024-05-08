# This is a sample Python script.
from WordEmbeddings import WordEmbeddings
from semantic_search import SemanticSearch
from decision_tree.decision_tree import DecisionTree
from openai_init import OpenaiInit
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

origins = [
    "http://127.0.0.1:5000",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/embeddings")
async def root():
    openai_init = OpenaiInit()
    word_embeddings = WordEmbeddings(openai_init.openai_key)
    word_embeddings.embedding()
    response_headers = {
        "Access-Control-Allow-Origin": "*",
    }
    return Response(content={"status": "Categories were successfully embedded"}, headers=response_headers)


@app.get("/semantic_search")
async def perform_semantic_search(sentence: str = Query(..., description="The sentence for semantic search")):
    openai_init = OpenaiInit()
    semantic_search = SemanticSearch(openai_init.openai_key, sentence)
    semantic_search.get_sorted_categories()
    result = semantic_search.relevant_categories()
    with open('account_number_mapping.json', 'r') as file:
        json_account_numbers = json.load(file)
    with open('account_name_mapping.json', 'r') as file:
        json_account_names = json.load(file)
    response = [{json_account_names[str(value)]: json_account_numbers[str(value)]} for index, value in result.items() if
                str(value) in json_account_names and str(value) in json_account_numbers]
    return response


@app.get("/decision_tree")
async def make_prediction(account1: int = Query(..., description="Value for account1"),
                          account2: int = Query(..., description="Value for account2"),
                          account3: int = Query(..., description="Value for account3"),
                          document: str = Query(..., description="document_label"), ):
    accounts = [account1, account2, account3]
    decision_tree = DecisionTree("./decision_tree/dataset.csv", "./decision_tree/dataset_combinations.csv", accounts,
                                 document)
    if os.path.exists("./decision_tree/decision_tree_model_md.pkl") and os.path.exists(
            "./decision_tree/decision_tree_model_d.pkl"):
        pass
    else:
        decision_tree.train_model()
    response = decision_tree.make_prediction()
    return response
