import openai
import os
import json
import sys
from .Config import Simulator_Config
from utils.LLM_call import OPENAI_call
from Eval.Metrics import LengthJudger, InfoAmountJudger, FormalityJudger
from Eval.Metrics import CRS_Rater
from Eval.Metrics import User_comparator
from main import UserSimulator
from models.USER import iEvaLM_User_Beauty, iEvaLM_User_Food
from models.CRS import GptBasedCRS, AgentCRS
"""Sys path 会变"""




class RecUserBeauty:
    def __init__(self, llm=None):
        if llm is None:
            self.name = "RecUser"
        else:
            self.name = "RecUserBeauty" + llm
class CSHIUserBeauty:
    def __init__(self, a, b, model_name=None):
        if model_name is None:
            self.name = "CSHIUserBeauty"
        else:
            self.name = "CSHIUserBeauty" + model_name

class RecUserFood:
    def __init__(self, llm=None):
        if llm is None:
            self.name = "RecUser"
        else:
            self.name = "RecUserFood" + llm

class RecUser2:
    def __init__(self, llm=None):
        if llm is None:
            self.name = "RecUser"
        else:
            self.name = "RecUser2" + llm

class RecUser_now:
    def __init__(self, llm=None):
        if llm is None:
            self.name = "RecUser"
        else:
            self.name = "RecUser" + llm +"_now"


profiles = []
with open("../synthesized_profiles.jsonl", 'r') as f:
    for line in f:
        profiles.append(str(line))

LLM_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.5,
    "max_tokens": 9000,
    "seed": 39,
}

LLM_Model = OPENAI_call(LLM_config["model"])
LLM_Model.set_temperature(LLM_config["temperature"])
LLM_Model.set_max_tokens(LLM_config["max_tokens"])
LLM_Model.set_seed(LLM_config["seed"])


LLM_config1 = {
    "model": "deepseek-reasoner",
    "temperature": 0.5,
    "max_tokens": 9000,
    "seed": 39,
}

LLM_Model1 = OPENAI_call(LLM_config1["model"])
LLM_Model1.set_temperature(LLM_config1["temperature"])
LLM_Model1.set_max_tokens(LLM_config1["max_tokens"])
LLM_Model1.set_seed(LLM_config1["seed"])




LLM_config2 = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_tokens": 4096,
    "seed": 39,
}

LLM_Model2 = OPENAI_call(LLM_config2["model"])
LLM_Model2.set_temperature(LLM_config2["temperature"])
LLM_Model2.set_max_tokens(LLM_config2["max_tokens"])
LLM_Model2.set_seed(LLM_config2["seed"])

LLM_config3 = {
    "model": "deepseek-v3",
    "temperature": 0.5,
    "max_tokens": 4096,
    "seed": 39,
}

LLM_Model3 = OPENAI_call(LLM_config3["model"])
LLM_Model3.set_temperature(LLM_config3["temperature"])
LLM_Model3.set_max_tokens(LLM_config3["max_tokens"])
LLM_Model3.set_seed(LLM_config3["seed"])

LLM_config4 = {
    "model": "gpt-4.1-nano",
    "temperature": 0.5,
    "max_tokens": 4096,
    "seed": 39,
}

LLM_Model4 = OPENAI_call(LLM_config4["model"])
LLM_Model4.set_temperature(LLM_config4["temperature"])
LLM_Model4.set_max_tokens(LLM_config4["max_tokens"])
LLM_Model4.set_seed(LLM_config4["seed"])

class Evaluation_Config_TTS(Simulator_Config):
    llm_chat = LLM_Model           # 用于回复的 LLM 模型
    llm_action = LLM_Model       # 用于动作生成的 LLM 模型
    Sample_Method = "TTS"  # 可选: 'Uniform_Sample', 'Prob_Sample', 'TTS'
    name = f"OurUser_{llm_action.llm_model}_{llm_chat.llm_model}_{Sample_Method}"  # 用户名带模型信息



class Evaluation_Config_random(Simulator_Config):
    llm_chat = LLM_Model           # 用于回复的 LLM 模型
    llm_action = LLM_Model       # 用于动作生成的 LLM 模型
    Sample_Method = "Uniform_Sample"  # 可选: 'Uniform_Sample', 'Prob_Sample', 'TTS'
    name = f"OurUser_{llm_action.llm_model}_{llm_chat.llm_model}_{Sample_Method}"  # 用户名带模型信息



    name = f"OurUser_{llm_action.llm_model}_{llm_chat.llm_model}_{Sample_Method}"  # 用户名带模型信息


class Evaluation_Config_prob(Simulator_Config):
    llm_chat = LLM_Model         # 用于回复的 LLM 模型
    llm_action = LLM_Model       # 用于动作生成的 LLM 模型
    Sample_Method = "Prob_Sample"  # 可选: 'Uniform_Sample', 'Prob_Sample', 'TTS'
    Domain_SYS = "美食推荐系统"
    name = f"OurUser_{llm_action.llm_model}_{llm_chat.llm_model}_{Sample_Method}"  # 用户名带模型信息

tts_config = Evaluation_Config_TTS()
random_config = Evaluation_Config_random()
prob_config = Evaluation_Config_prob()


############# CRS #############
crs_2 = GptBasedCRS(model="gpt-4o-mini")


############# USER #############
# user2 = UserSimulator(profile=profiles[0], llm_action=LLM_Model1, llm_chat=LLM_Model)
user4 = RecUserBeauty("gpt-4o-mini")
# user5 = RecUser("")
# user6 = RecUser_now("gpt-4o-mini")
# user7 = RecUser1("gpt-4o-mini")
# user8 = RecUser2("gpt-4o-mini")
# user3 = RecUser("gpt-3.5-turbo")
# user1 = UserSimulator(profile=profiles[0], config=tts_config)
# user2 = UserSimulator(profile=profiles[0], config=random_config)
user3 = UserSimulator(profile=profiles[0], config=prob_config)
# LLM_Model.llm_model = "deepseek-reasoner"
# user2 = UserSimulator(profile=profiles[0], llm_action=LLM_Model, llm_chat=LLM_Model)
# LLM_Model.llm_model = "gpt-3.5-turbo"
# user3 = UserSimulator(profile=profiles[0], llm_action=LLM_Model, llm_chat=LLM_Model)

# user2 = []
# with open("/data/liyuanzi/HUAWEI/User_simulator/RecUserSim/agents/outputs/iEvaLMUser(gpt-3.5-turbo)_BaseCRS(gpt-4o-mini)/profile.jsonl", 'r') as f:
#     datas = json.load(f)
#     for data in datas[:10]:
#         user2.append(iEvaLM_User(data["target_item"], data["item_info"], model_name="deepseek-reasoner"))
# user2 = iEvaLM_User_Beauty("爽肤水", "", model_name="gpt-3.5-turbo")
# user3 = iEvaLM_User("火锅", "麻辣鲜香，是川菜的代表之一", model_name="gpt-4o-mini")
# user4 = iEvaLM_User("火锅", "麻辣鲜香，是川菜的代表之一", model_name="deepseek-reasoner")
user5 = CSHIUserBeauty("爽肤水","",model_name="gpt-4o-mini")
user6 = iEvaLM_User_Beauty("爽肤水", "",model_name="gpt-4o-mini")


############# JUGER #############
judge_llm = OPENAI_call("gpt-4.1-mini")
# judge_llm = OPENAI_call("deepseek-v3")
judge_llm.set_max_tokens(9000)
comparator = User_comparator(llm_call=judge_llm)
evaluation_metrics = [LengthJudger(), InfoAmountJudger(llm_call=judge_llm), FormalityJudger(llm_call=judge_llm)]
rater = CRS_Rater(llm_call=judge_llm)


EVAL_CONFIG = {
    "user_agent": [user3],
    "crs_agent": [crs_2],
    "language": "zh",
    "encoding": "utf-8",
    "save_path": os.path.join(os.path.dirname(__file__), "..", "test_Beauty"),
    "interaction_config": {
        "do_interaction": True,
        "init_type": "random_sample",
        "interaction_num": 100,
        "max_turn": 5,
    },
    "compare_config": {
        "do_compare": False,
        "main_agent": user3,
        "compare_times": 100,
        "comparator": comparator
    },
    "evaluation_config": {
        "do_evaluation": False,
        "metrics": evaluation_metrics,
        "evaluated_max_turn": 5,
        "judge_boost": 1,
        "start_from_raw": False,
        "distinct_ngram": None
    },
    # 弃用
    "rating_config": {
        "do_rating": False,
        "rater": rater
    },

}
