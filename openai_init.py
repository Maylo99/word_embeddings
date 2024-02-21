import json


class OpenaiInit:
    def __init__(self):
        with open('config.json') as f:
            config = json.load(f)
        self.openai_key = config['openai_key']
