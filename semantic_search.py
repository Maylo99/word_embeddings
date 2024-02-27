import openai
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

class SemanticSearch:
    def __init__(self, open_ai_config):
        self.open_ai_config = open_ai_config
        openai.api_key = self.open_ai_config

    def get_sorted_categories(self):
        df = self._semantic_search()
        return self._sort_by_similarity(df)
    def _semantic_search(self):
        df = pd.read_csv('categories_embeddings.csv')
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)
        search_term = input('Enter accounting case: ')
        search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")
        df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
        return df

    def _sort_by_similarity(self, df):
        sorted_df = df.sort_values("similarities", ascending=False).head(20)
        sorted_df.to_csv('sorted_categories.csv', index=False)