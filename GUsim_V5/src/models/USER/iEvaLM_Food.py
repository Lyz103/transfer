# iEvaLM
# -*- coding: utf-8 -*-
# @Time : 2024/7/23 16:07
from datatypes import Agent
from openai import OpenAI

import random


class iEvaLM_User_Food(Agent):
    def __init__(self, target_item: str, item_info: str, model_name="gpt-3.5-turbo", name="iEvaLMUserFood",
                 description="A user model for iEvaLM"):
        super().__init__(name+model_name, description)
        prompt = """
        You are a seeker chatting with a recommender for recommendation. Your target items: {target_item}. You must follow the instructions below during chat.
        Here is the information about target item: {item_info}.
        If the recommender recommend {target_item}, you should accept.
        If the recommender recommend other items, you should refuse them and provide the information about {target_item}. You should never directly tell the target item title.
        If the recommender asks for your preference, you should provide the information about {target_item}. You should never directly tell the target item title.
        Always remember that you are a seeker and you are looking for the target item. Do not provide recommendation to the recommender.
        请注意，您扮演的是一个寻找推荐的用户，而不是推荐者。请时刻注意要寻求推荐，而不是提供推荐。
        Please speak in Chinese.
        """
        self.prompt = prompt
        self.interaction_history = []
        self.target_item = target_item
        self.item_info = item_info
        self.model_name = model_name

        self.client = OpenAI()
    def _call_llm(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def interact(self, str_input) -> str:
        if str_input is not None:
            self.interaction_history.append({
                "role": "system",
                "content": str_input
            })
        messages = [{"role": "system",
                     "content": self.prompt.format(target_item=self.target_item, item_info=self.item_info)}]
        for interaction in self.interaction_history:
            messages.append(interaction)
        response = self._call_llm(messages)
        self.interaction_history.append({
            "role": "user",
            "content": response
        })
        return response

    def clear_interaction_history(self):
        self.interaction_history = []

    def get_profile(self):
        return {
            "target_item": self.target_item,
            "item_info": self.item_info
        }

    def _get_random_item(self):
        prompt = """
        You are a seeker chatting with a recommender for food recommendation. 
        Please randomly choose a target item and provide its' attributes.
        Possible target items can be "饺子混沌", "汉堡薯条", "炸鸡炸串", "意面披萨", "包子粥店", "快餐便当", "米粉面馆", "麻辣烫冒菜"
        The attributes about the target item includes the taste, the ingredients, the cooking method, etc.
        However, you can also provide your own target item.
        Please provide the target item using the following format:
        [item]str(in Chinese)[/item]
        Please provide the all the attributes about the target item using one [info] [/info] tag:
        [info]string(in Chinese)[/info]
        Note that you just need to list out the attributes about the target item, not introduce it.
        """
        messages = [{"role": "system",
                     "content": prompt}]
        model = self.model_name
        self.model_name = "gpt-4o"
        response = self._call_llm(messages)
        self.model_name = model
        try:
            target_item = response.split("[item]")[1].split("[/item]")[0]
        except:
            target_item = random.choice(["饺子混沌", "汉堡薯条", "炸鸡炸串", "意面披萨", "包子粥店", "快餐便当", "米粉面馆", "麻辣烫冒菜"])
        try:
            item_info = response.split("[info]")[1].split("[/info]")[0]
        except:
            item_info = response
        return target_item, item_info

    def _get_item_info(self, target_item):
        prompt = """
        You are a seeker chatting with a recommender for food recommendation. 
        Your target item is : {target_item}.
        Determine the target item’s attributes.
        The attributes about the target item includes the taste, the ingredients, the cooking method, etc.
        Please provide the all the attributes about the target item using one [info] [/info] tag:
        [info]string(in Chinese)[/info]
        Note that you just need to list out the attributes about the target item, not introduce it.
        """
        model = self.model_name
        self.model_name = "gpt-4o"
        self.model_name = model
        messages = [{"role": "system",
                     "content": prompt.format(target_item=target_item)}]
        response = self._call_llm(messages)
        try:
            item_info = response.split("[info]")[1].split("[/info]")[0]
        except:
            item_info = response
        return item_info


    def refresh_profile(self,
                        target_item: str = None,
                        item_info: str = None):
        if target_item is None:
            target_item, item_info = self._get_random_item()
        elif item_info is None:
            item_info = self._get_item_info(target_item)
        self.target_item = target_item
        self.item_info = item_info


if __name__ == "__main__":
    openai.api_base = ""
    openai.api_key = ""
    user = iEvaLM_User("川菜-朝天门老火锅", "麻辣鲜香，是川菜的代表之一", model_name="gpt-4o-mini")
    for i in range(5):
        input_str = input("Recommender: ")
        response = user.interact(input_str)
        print("User: ", response)
    print(user.get_profile())
    user.refresh_profile()
    for i in range(5):
        input_str = input("Recommender: ")
        response = user.interact(input_str)
        print("User: ", response)
    print(user.get_profile())
    user.refresh_profile("川菜-重庆小面")
    for i in range(5):
        input_str = input("Recommender: ")
        response = user.interact(input_str)
        print("User: ", response)
    print(user.get_profile())
