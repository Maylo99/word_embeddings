# This is a sample Python script.
from WordEmbeddings import WordEmbeddings
from semantic_search import SemanticSearch
from openai_init import OpenaiInit
from fastapi import FastAPI,Query
import json
app = FastAPI()
@app.get("/embeddings")
async def root():
    openai_init = OpenaiInit()
    word_embeddings = WordEmbeddings(openai_init.openai_key)
    word_embeddings.embedding()
    return {"status": "Categories were successfully embedded"}

@app.get("/semantic_search")
async def perform_semantic_search(sentence: str = Query(..., description="The sentence for semantic search")):
    openai_init = OpenaiInit()
    semantic_search = SemanticSearch(openai_init.openai_key,sentence)
    semantic_search.get_sorted_categories()
    result=semantic_search.relevant_categories()
    with open('account_number_mapping.json', 'r') as file:
        json_account_numbers = json.load(file)
    with open('account_name_mapping.json', 'r') as file:
        json_account_names = json.load(file)
    response = [{json_account_names[str(value)]: json_account_numbers[str(value)]} for index, value in result.items()]
    return response
