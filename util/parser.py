import json
import pandas as pd
from random import choice


class JSONParser:
    def __init__(self):
        self.text = []
        self.intents = []
        self.responses = {}

    def parse(self, json_path):
        with open(json_path) as data_file:
            self.data = json.load(data_file)

        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                self.text.append(pattern)
                self.intents.append(intent['tag'])
            for resp in intent['responses']:
                if intent['tag'] in self.responses.keys():
                    self.responses[intent['tag']].append(resp)
                else:
                    self.responses[intent['tag']] = [resp]

        self.df = pd.DataFrame({'text_input': self.text,
                                'intents': self.intents})

        print(
            f"[INFO] Data JSON converted to DataFrame with shape : {self.df.shape}")

    def get_dataframe(self):
        return self.df

    def get_response(self, intent):
        return choice(self.responses[intent])
