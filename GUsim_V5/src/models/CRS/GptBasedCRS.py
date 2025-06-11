# GptBasedCRS
# -*- coding: utf-8 -*-
# @Author : LuyuChen
# @Time : 2024/7/23 16:24
from datatypes import Agent
from openai import OpenAI

class GptBasedCRS(Agent):
    def __init__(self, name: str ="BaseCRS",
                 description: str = "A Conversational Recommender System based on GPT-4o.",
                 prompt: str = None,
                 model: str = "gpt-4o-mini"):
        super().__init__(name+model, description)
        self.client = OpenAI()
        self._class = "GptBasedCRS"
        default_prompt = """
        You are a helpful Conversational Recommender System (CRS) that helps users find the items they want.
        You are a friendly and helpful assistant that can provide recommendations based on the user's preferences.
        You can also provide information about the items, such as the price, availability, and reviews.
        You can also help users make decisions by providing pros and cons of different items.
        You can also answer questions about the items and provide additional information.
        Now, given the interaction between you and the user:
        {interaction_history}
        please provide a helpful response to the user's request{user_last_utterance}.
        Please recommend short and concise responses to the user's requests.
        """
        self.prompt = prompt if prompt is not None else default_prompt
        self.interaction_history = ""
        self.model = model

    def interact(self, str_input: str) -> str:
        self.interaction_history += "User: " + str_input + "\n"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt.format(interaction_history=self.interaction_history,
                                                                 user_last_utterance=str_input)}
            ]
        )
        self.interaction_history += "Recommender System: " + response.choices[0].message.content + "\n"
        return response.choices[0].message.content

    def clear_interaction_history(self):
        self.interaction_history = ""
