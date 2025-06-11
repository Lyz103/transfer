
import os
import sys

from datatypes import Agent
from tenacity import Retrying, retry_if_not_exception_type
import openai
from openai import OpenAI


planner_prompt = """
You are chatting with a user, now you need to choose an action based on the user's reply.

You can choose from the following actions:

Ask: If the user's preferences are not clear, you could ask the user for more information to better understand their preferences.
Recommend: If the user's preferences are clear, you can give the user a personalized recommendation based on the user's preferences.
Answer: If the user asks a question, you can provide a corresponding answer. For example, if the user asks about the price of a dish, you can provide the price information.
Other: You choose this action because none of the above actions are suitable.
Output the action you choose and output the thought for your choice as a reason.

User reply: Can you recommend food?
User preferences: No preference detected.
Output: Ask # The user's preferences are not clear, so I need to ask the user for more information to better understand their preferences.

User reply: Can you recommend some dishes?
User preferences: Spicy dishes, Indian cuisine.
Output: Recommend # I can give the user a personalized recommendation based on their preferences for Indian cuisine and spicy dishes.

User reply: How much does the dish cost?
User preferences: No preference detected.
Output: Answer # I need to provide the user with the price information of the dish.

User reply: Thanks for the recommendation.
User preferences: Spicy dishes, Indian cuisine.
Output: Other # None of the above actions are suitable. I need to say "You're welcome" to the user.

User reply: {user_input}
User preferences: {preferences}
"""

summarize_prompt = """
Prompt: Given a conversation, summarize the user's preferences about food.

Conversation:

User: I am looking for some spicy dishes.
Recommender: Do you prefer any specific cuisine?
User: I prefer Indian cuisine.
Recommender: Do you have any dietary restrictions?
User: No, all options are fine.
Output: The user prefers spicy dishes from Indian cuisine and does not have any specific dietary restrictions.

Conversation: {conversation}
"""

ask_prompt = """
You are chatting with a user. The user's reply is: {user_input}, and you want to get the user's preferences through the conversation in the field of food.
Your task is to gather the user's preferences through natural conversation.


The following information may help you better tap into user preferences: [cuisine, flavor profile, ingredients, dietary restrictions, meal type, cooking method].

Please remember the following rules:

1. Do not ask for all the information. Choose 1 to 3 of the most important information you want to ask.
2. Give the user some examples of the information you want to ask for.
3. Don't ask for information that is already known to be relevant to the user's preferences: {preferences}
4. Note that your output is also a reply to the user's message: {user_input}

Please speak in Chinese.
"""

recommend_prompt = """
You are an excellent food recommender. Your goal is to provide a personalized recommendation based on the user's preferences and user's reply.

Now the user's preferences are: {preferences}
The user's reply is: {user_input}

Please remember the following rules:

1. Select the most relevant {k} dishes according to the user's preferences.
2. The higher the relevance, the higher the ranking in the recommendation list.
3. Output a list of recommended dishes or restaurants, for example: I recommend you [dish/restaurant name1, dish/restaurant name2, dish/restaurant name3, ..., dish/restaurant name{k}]
4. Then briefly explain why you recommend these dishes or restaurants to the user.

Please speak in Chinese.
"""

answer_prompt = """
You are chatting with a user who is seeking for food recommendation. The user's reply is: {user_input}, and you need to provide the user with the corresponding answer.

Please provide the user with the information they need based on the user's reply.

For example, if the user asks about the price of a dish, you can provide the price information.

User reply: {user_input}

Please speak in Chinese.
"""

other_prompt = """
You are food expert chatting with a user who is seeking for food recommendation. The user's reply is: {user_input}, and you need to give a response.

Please speak in Chinese.
"""



class CRSAgent():
    def __init__(self, k=5, model_name="gpt-4o-mini"):
        self.k = k
        self.model_name = model_name

        self.oai_messages = list()
        self.last_action = None
        self.last_preferences = None
        self.action_count = 0
        self.client = OpenAI()
        # self.action_list = ['Ask', 'Recommend']


    def call_openai_response(self, messages):
        request_timeout = 20
        MAX_TRY = 10
        try:
            for _ in range(MAX_TRY):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2048,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Failed to call LLM model: {e}")
            return None
        return response.choices[0].message.content

    def reset(self):
        self.oai_messages = list()
        self.last_action = None
        self.last_preferences = None
        self.action_count = 0

    def step(self, user_input):
        self.oai_messages.append({"role": "user", "content": user_input})
        user_preferences = self._summrize_user_preferences()

        action = self._get_action(user_preferences, user_input)
        # print(action)
        action = action.split("#")[0].strip()


        if 'Ask' in action:
            response = self._ask_action(user_preferences, user_input)
        elif 'Recommend' in action:
            response = self._recommend_action(user_preferences, user_input)
        elif 'Answer' in action:
            response = self._answer_action(user_input)
        else:
            response = self._other_action(user_input)
            # raise ValueError(f"Unknown action: {action}")

        self.oai_messages.append({"role": "assistant", "content": response})
        # print(self.oai_messages)

        return response, action

    def _get_action(self, user_preferences, user_input):
        messages = [{"role": "system", "content": planner_prompt.format(user_input=user_input, preferences=user_preferences)}]
        return self.call_openai_response(messages)

    def _summrize_user_preferences(self):
        conversation = self._prepare_conversation()
        messages = [{"role": "system", "content": summarize_prompt.format(conversation=conversation)}]
        return self.call_openai_response(messages)

    def _ask_action(self, user_preferences, user_input):
        messages = [{"role": "system", "content": ask_prompt.format(preferences=user_preferences, user_input=user_input)}]
        messages.extend(self.oai_messages)
        return self.call_openai_response(messages)

    def _recommend_action(self, user_preferences, user_input):
        messages = [{"role": "system", "content": recommend_prompt.format(preferences=user_preferences, k=self.k, user_input=user_input)}]
        return self.call_openai_response(messages)

    def _answer_action(self, user_input):
        messages = [{"role": "system", "content": answer_prompt.format(user_input=user_input)}]
        return self.call_openai_response(messages)

    def _other_action(self,user_input):
        messages = [{"role": "system", "content": other_prompt.format(user_input=user_input)}]
        return self.call_openai_response(messages)

    def _prepare_conversation(self):
        conversation = ""
        for message in self.oai_messages:
            if message['role'] == 'user':
                conversation += f"- User: {message['content']}\n"
            elif message['role'] == 'assistant':
                conversation += f"- Recommender: {message['content']}\n"
            else:
                raise ValueError(f"Unknown role: {message['role']}")

        return conversation


class AgentCRS(Agent):
    def __init__(self, name: str = "AgentCRS",
                 description: str = "A Conversational Recommender System based on LLM Agent.",
                 model: str = "gpt-4o-mini"):
        super().__init__(name+model, description)
        self.model = model
        self.crs_agent = CRSAgent(model_name=model)

    def interact(self, str_input: str) -> str:
        response, action = self.crs_agent.step(str_input)
        return response

    def clear_interaction_history(self):
        self.crs_agent.reset()

if __name__ == "__main__":
    # os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech"

    # os.environ["OPENAI_API_KEY"] = "sk-G6XflAR04SvxaTl1QhmIl6HimueZ0ZPYFeEVD78gELMRqth5"
    # client = OpenAI()
    # def call_openai_response(messages):
    #     request_timeout = 20
    #     try:
    #         response = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=messages,
    #             temperature=0,
    #             max_tokens=2048,
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         LOGGER.warning(f"Failed to call LLM model: {e}")
    #         return None
    # import os
    # messages = [{"role": "system", "content": "hello"}]
    # call_openai_response(messages)


    crs = AgentCRS(model="gpt-4o-mini")


    while True:
        user_input = input("User: ")
        print(user_input)
        response = crs.interact(user_input)
        print(f"CRS: {response}")