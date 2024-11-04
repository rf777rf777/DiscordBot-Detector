import json

class Config:
    def __init__(self):
        pass

    @staticmethod
    def get_config():
        """
        get config content from config.json
        """
        with open('configs/config.json') as f:
            return json.load(f)