import openai
import pandas as pd
from openai.embeddings_utils import get_embedding


class WordEmbeddings:
    def __init__(self, open_ai_config):
        self.open_ai_config = open_ai_config
        openai.api_key = self.open_ai_config

    def embedding(self):
        df_categories = self._read_categories_data()
        self._calculate_word_embeddings(df_categories)

    def _read_categories_data(self):
        df = pd.read_csv('categories.csv')
        print(df)
        return df

    def _calculate_word_embeddings(self, df):
        df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
        df.to_csv('categories_embeddings.csv')