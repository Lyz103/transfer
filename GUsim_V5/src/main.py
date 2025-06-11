import os
import sys
from typing import List, Dict, Tuple, Union, Callable
import re
import logging
from datetime import datetime
import time
import random

from models.CRS.AgentCRS import AgentCRS
from models.CRS.GptBasedCRS import GptBasedCRS

# from config.LLM_config import LLM_Model_Chat, LLM_Model_Action

# print(sys.path)
from config.Config import Simulator_Config
# from modules.Memory.default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY
from config.Memory_config import DEFAULT_PROFILEMEMORY, DEFAULT_ACTIONMEMORY, DEFAULT_CHATMEMORY
from Memory.config.Config import MemoryConfig
from Memory import ChatMemory, ActionMemory, ProfileMemory
from utils import *
from Prompts.Prompts_ma import *
from Prompts.style_prompts import *


# 创建 Logger 对象
LOGGER = logging.getLogger('my_logger')
LOGGER.setLevel(logging.DEBUG)  # 设置全局日志级别

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)

# 文件处理器
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)  # 仅记录错误及以上级别
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 添加两个处理器
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)


# from Policy import LLM_Policy
class UserMemory:
    def __init__(self, profile):
        self.chat_memory_config = MemoryConfig(DEFAULT_CHATMEMORY)
        self.action_memory_config = MemoryConfig(DEFAULT_ACTIONMEMORY)
        self.profile_memory_config = MemoryConfig(DEFAULT_PROFILEMEMORY)
        self.chat_memory = ChatMemory(self.chat_memory_config)
        self.action_memory = ActionMemory(self.action_memory_config)
        self.profile_memory = ProfileMemory(self.profile_memory_config)
        self.profile_memory.store(profile)
    
    def recall(self) -> Union[str, str, str]:
        return self.profile_memory.recall(""), self.chat_memory.recall(""), self.action_memory.recall("")
    
    def _change_profile(self, new_profile) -> None:
        self.profile_memory.reset()
        self.profile.store(new_profile)
    
    def _store_action(self, action) -> None:
        self.action_memory.store(action)
    
    def _store_chat(self, chat_his) -> None:
        self.chat_memory.store(chat_his)
    
    def reset(self, new_profile) -> None:
        self.profile_memory.reset()
        self.profile_memory.store(new_profile)
        self.action_memory.reset()
        self.chat_memory.reset()

class UserAct:
        def __init__(self,
                 usermemory: UserMemory,
                 config
                #  action_description_dir: str = None,
                #  action_space_prompt_dir: str = None,
                #  action_prompt_dir: str = None,
                #  action_incontext_dir: str = None,
                #  action_end_conversation_dir: str = None,
                #  response_prompt_dir: str = None,
                #  response_incontext_dir: str = None,
                #  reward_prompt_dir: str = None,
                #  reward_incontext_dir: str = None,
                #  reward_judge_dir: str = None,
                 ):
            
            self.gen_action_llm = config.llm_action
            self.gen_dialogue_llm = config.llm_chat
            print("self.gen_action_llm:", self.gen_action_llm.model)
            print("self.gen_dialogue_llm:", self.gen_dialogue_llm.model)
            self.user_memory = usermemory
            self.sample_method = config.Sample_Method
            self.turn_end = 5
            self.config = config


        def _gen_actions(self, turn) -> Union[List, str, bool]:
            user_profile, his_chat, his_action = self.user_memory.recall()

            last_sys_output = re.split(r'\[turn \d+\]', his_chat)[-1]
            LOGGER.debug("last_sys_output: %s", last_sys_output)
            chat_action_his = put_action_chat_together(his_chat, his_action)

            user_profile = eval(user_profile)

            input_message = {}
            input_message.update(user_profile)
            input_message.update({"chat_action_his": chat_action_his})
            input_message.update({'domain_sys':self.config.Domain_SYS})

            if self.config.Domain_SYS == "美食推荐系统":
                input_message.update(user_profile['domain']['Foodie']['specific'])
                input_message.update(user_profile['domain']['Foodie']['context'])
            elif self.config.Domain_SYS == "美妆推荐系统":
                input_message.update(user_profile['domain']['Skincare']['specific'])
                input_message.update(user_profile['domain']['Skincare']['context'])
            elif self.config.Domain_SYS == "虚拟商店推荐系统":
                input_message.update(user_profile['domain']['VMALL']['specific'])
                input_message.update(user_profile['domain']['VMALL']['context'])
            elif self.config.Domain_SYS == "闲聊系统":
                pass

            LOGGER.debug("input_message: %s", input_message)
            LOGGER.debug("end_turn: %s", self.turn_end)

            if turn >= self.turn_end:
                end_info = self.gen_action_llm(JUDGE_END.render(input_message))
                LOGGER.debug(f"end_info :{end_info}")
                end_info = eval(end_info)
                if_end = "YES" in end_info['判断']
                LOGGER.debug(f"if_end :{if_end}")
            else:
                if_end = False
            if if_end:
                sampled_action = "结束对话"
                sampled_reason = end_info["原因"]
                LOGGER.info("采样动作: %s", sampled_action)
                LOGGER.info("采样原因: %s", sampled_reason)
            
            if if_end == False:
                if self.sample_method == "TTS":
                    sampled_action, sampled_reason = self._TTS(input_message)
                elif self.sample_method == "Uniform_Sample":
                    sampled_action, sampled_reason = self._random_sample(input_message)
                elif self.sample_method == "Prob_Sample":
                    sampled_action, sampled_reason = self._prob_sample(input_message)

            return sampled_action, sampled_reason, if_end



        def _gen_dialogue(self, turn) -> Union[str,bool]:
            # do not foget to add
            actions, reason, if_end = self._gen_actions(turn)
            user_profile, his_chat, his_action = self.user_memory.recall()
            last_sys_output = re.split(r'\[turn \d+\]', his_chat)[-1]
            user_profile = eval(user_profile)
            # style = user_profile['user_behavior']
            # key, value = next(iter(user_profile['user_behavior'].items()))
            # style_prompt = (eval(f"{key}_{value}"))
            chat_action_his = put_action_chat_together(his_chat, his_action)
            style = user_profile['user_behavior']
            style_prompt = ""
            for key, value in style.items():
                style_prompt += (eval(f"{key}_{value}")) + '\n'
                
            # input_message = {}
            # input_message.update(user_profile['basic_info'])
            # input_message.update(user_profile['environment_info'])
            # input_message.update(user_profile['preference'])
            # input_message.update({"style_prompt":style_prompt})
            # input_message.update({"chat_action_his": chat_action_his})

            # print("input_message:", input_message)


            input_message = {}
            input_message.update(user_profile)
            input_message.update({"action_names": actions})
            input_message.update({"inner_voice": reason})
            input_message.update({"chat_action_his": chat_action_his})
            input_message.update({'domain_sys':self.config.Domain_SYS})
            input_message.update({"style_prompt":style_prompt})


            print("self.config.Domain_SYS:", self.config.Domain_SYS)
            if self.config.Domain_SYS == "美食推荐系统":
                input_message.update(user_profile['domain']['Foodie']['specific'])
                input_message.update(user_profile['domain']['Foodie']['context'])
            elif self.config.Domain_SYS == "美妆推荐系统":
                input_message.update(user_profile['domain']['Skincare']['specific'])
                input_message.update(user_profile['domain']['Skincare']['context'])
            elif self.config.Domain_SYS == "虚拟商店推荐系统":
                input_message.update(user_profile['domain']['VMALL']['specific'])
                input_message.update(user_profile['domain']['VMALL']['context'])
            elif self.config.Domain_SYS == "闲聊系统":
                pass

            input = RESPONSE.render(input_message)
            LOGGER.info("*" * 100)
            LOGGER.info("input4chat: %s", input)
            LOGGER.info("*" * 100)

            dialogue_output = eval(self.gen_dialogue_llm(input))
            response = dialogue_output.get("回复", "")
            reasons = dialogue_output.get("原因", "")
            reflect1 = dialogue_output.get("思考1", "")
            reflect2 = dialogue_output.get("思考2", "")
            # response_common = dialogue_output.get("正常回复", "")

            # print("response:", response)
            # print("reasons:", reasons)
            LOGGER.info("思考1: %s", reflect1)
            LOGGER.info("思考2: %s", reflect2)
            LOGGER.info("response: %s", response)
            LOGGER.info("reasons: %s", reasons)
            # LOGGER.info("正常回复: %s", response_common)

            self._update_memory_User(actions, response)
            return response, if_end


        def _update_memory_User(self, actions, response) -> None:
            self.user_memory._store_action("[" + str(actions) + "]")
            self.user_memory._store_chat(response)

        def _update_memory_Sys(self, response) -> None:
            self.user_memory._store_chat(response)
        

        def _prob_sample(self, input_message):
            input = CRS_CHOOSE_ACTION.render(input_message)
            LOGGER.info("*" * 100)
            LOGGER.info("input4action: %s", input)
            LOGGER.info("*" * 100)

            actions_with_prob = self.gen_action_llm(input)
            LOGGER.info("actions_with_prob: %s", actions_with_prob)
            actions_with_prob_dict, reason, reflect = parse_actions_withprob(actions_with_prob)
            # print("5个动作:", actions_with_prob_dict)
            # print("原因:", reason)
            LOGGER.info("思考: %s", reflect)
            LOGGER.info("动作: %s", actions_with_prob_dict)
            LOGGER.info("原因: %s", reason)


            sampled_action, sampled_reason = top_p_sample(actions_with_prob_dict, str(reason), p=0.1)
            # print("采样动作:", sampled_action)
            # if_end = self._judge_end(actions_with_prob_dict, sampled_action, his_action)
            # if end: if_end = True
        
            return sampled_action, sampled_reason



class UserSimulator:
    """
    User Simulator to simulate the user's behavior.
    """
    def __init__(self, profile: str, config=None, llm_action=None, llm_chat=None) -> None:
        self.memory = UserMemory(profile)
        self.llm_action = config.llm_action
        self.llm_chat = config.llm_chat
        self.name = config.name
        self.config = config
        self.UserAct = UserAct(self.memory, config)
        self._class = "UserSimulator"


    def interact(self, turn) -> str:
        return self.UserAct._gen_dialogue(turn)
    
    def _update_memory(self, response):
        self.memory._store_chat(response)
        
    def refresh_profile(self, i) -> None:
        import json
        import random
        profiles = []
        profile_path = self.config.Profile_Path
        with open(f"../synthesized_profiles.jsonl", 'r') as f:
            for line in f:
                profiles.append(str(line))
        self.memory.reset(profiles[i])
        # self.UserAct.turn_end = random.randint(2, 4)

    def __str__(self) -> str:
        return "User Simulator"
    
    def __repr__(self) -> str:
        return "User Simulator"
        # OUTPUT: SimulatedUserAct, str:

    def __call__(self, turn: int = 0) \
            -> str:
        """
        Simulate the user behavior based on the user input.
        :param crs_input: for the first turn, the crs_input can be None, otherwise, it should be the crs response
        :return: the user action(name, description), the user response(str) and the reward(int) to the crs
        """
        a = self.interact(turn)
        time.sleep(0.3)
        return a


if __name__ == "__main__":
    cfg = Simulator_Config()
    profiles = []
    with open("profiles_w_f.jsonl", 'r') as f:
        for line in f:
            profiles.append(str(line))
    random.shuffle(profiles)
    for profile in profiles:
        try:
            LOGGER.info("\n\n\n\n\n\n\n")

            num = 0
            today = datetime.now()
            tody = today.strftime("%Y%m%d")
            file_dir = f"./data/{tody}——1"
            os.makedirs(file_dir, exist_ok=True)
            files_dir = os.listdir(file_dir)
            while f"4omini_{num}.txt" in files_dir:
                num += 1
            file_path = os.path.join(file_dir, f"4omini_{num}.txt")

            with open(file_path, 'w', encoding='utf-8') as f:

                crs = GptBasedCRS(model="gpt-4o-mini")
                usersimulator = UserSimulator(profile, config=cfg)
                print("\n" * 10)
                f.write("用户资料:"  + str(profile) + '\n\n\n\n')

                for turn in range(5):
                    user_chat, if_end = usersimulator.interact(turn)
                    print("turn_" + str(turn + 1) + "    User: ", user_chat)
                    f.write("[turn_" + str(turn + 1) + "]User: " +  ("None" if user_chat == None else user_chat) +  '\n\n\n\n')
                    # print("crs.crs_agent.oai_messages：", crs.crs_agent.oai_messages)
                    if if_end:
                        print("turn:", turn + 1)
                        print("\n" * 10)
                        break
                    sys_chat = crs.interact(user_chat)
                    usersimulator._update_memory(sys_chat)
                    print("Sys:", sys_chat )
                    f.write("[turn_" + str(turn + 1) + "]Sys:" + sys_chat + '\n\n\n\n')
        except Exception as e:
            LOGGER.error("Error occurred: %s", e)
            print(f"Error: {e}")
            continue

    print("Finish!")
        