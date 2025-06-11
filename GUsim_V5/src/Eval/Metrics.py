# Metrics
# -*- coding: utf-8 -*-
# @Author : LuyuChen
# @Time : 2024/7/24 14:53
import os
import sys
from typing import List,Union

import openai
from datatypes import Counter
from utils.LLM_call import OPENAI_call, LLM_call
from .prompts import PromptTemplate
from utils.loadf import load_file_utf8


class Judger:
    """
    Base class for judger
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def judge(self, history: dict, counter: Counter):
        raise NotImplementedError


class LengthJudger(Judger):
    """
    Judge the length of the user's response
    """

    def __init__(self,
                 name="LengthJudger",
                 description="Judge the length of the user's response",
                 mode="ch"):
        super().__init__(name, description)
        self.mode = mode

    def judge(self, history: Union[dict, List[dict]], counter: Union[Counter, List[Counter]]):
        if isinstance(history, dict):
            count_list = []
            for key in history.keys():
                print(key)
                if "turn_" in key:
                    if ("zh" or "ch" in self.mode.lower()) and history[key]['user'] != None:
                        count_list.append(len(history[key]["user"]))
                    elif "en" in self.mode.lower():
                        count_list.append(len(history[key]["user"].split(" ")))
                    else:
                        # raise ValueError("Only support Chinese and English")
                        continue
            counter.update(count_list)
        elif isinstance(history, list):
            turn_count = [0] * len(history)
            j = 0
            count_list = []
            for i in range(len(history)):
                for key in history[i].keys():
                    if "turn_" in key:
                        if "zh" or "ch" in self.mode.lower():
                            count_list.append(len(history[i][key]["user"]))
                        elif "en" in self.mode.lower():
                            count_list.append(len(history[i][key]["user"].split(" ")))
                        else:
                            raise ValueError("Only support Chinese and English")
                        j += 1
                turn_count[i] = j

            for i in range(len(history)):
                if i == 0:
                    counter[i].update(count_list[:turn_count[i]])
                else:
                    counter[i].update(count_list[turn_count[i-1]:turn_count[i]])



class InfoAmountJudger(Judger):
    """
    Judge the amount of information provided by the user
    """

    def __init__(self,
                 llm_call: LLM_call,
                 name="InfoAmountJudger",
                 description="Judge the amount of information provided by the user",
                 prompt_dir=os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "User_Sim_Judge", "InfoAmountCount.txt")
                 ):

        super().__init__(name, description)
        self.llm_call = llm_call
        self.prompt = PromptTemplate(template=load_file_utf8(prompt_dir))

    def judge(self, history: Union[dict, List[dict]], counter: Union[Counter, List[Counter]]):
        # history:{"turn_j": {"crs": "", "user": "", "action": [], "reward": ""}
        # Count the number of information provided by the user
        # count_list = []
        if isinstance(history, dict):
            utterance = ""
            i = 1
            for key in history.keys():
                if "turn_" in key:
                    utterance += f"{i}. {history[key]['user']}\n"
                    i += 1

            counter.update(self.count_infoAmount(utterance))
        elif isinstance(history, list):
            turn_count = [0] * len(history)
            j = 0
            utterance = ""
            for i in range(len(history)):
                for key in history[i].keys():
                    if "turn_" in key:
                        utterance += f"{j+1}. {history[i][key]['user']}\n"
                        j += 1
                turn_count[i] = j

            result = self.count_infoAmount(utterance)
            for i in range(len(history)):
                if i == 0:
                    counter[i].update(result[:turn_count[i]])
                else:
                    counter[i].update(result[turn_count[i-1]:turn_count[i]])


    def count_infoAmount(self, utterance) -> list:
        llm_input = self.prompt(utterance=utterance)
        llm_output = self.llm_call(llm_input)
        # get the count list
        llm_output = llm_output[llm_output.find("["):llm_output.find("]") + 1]
        # change the string to list
        count_list = eval(llm_output)
        return count_list

class FormalityJudger(Judger):
    """
    Judge the formality of the user's response
    """

    def __init__(self,
                 llm_call: LLM_call,
                 name="FormalityJudger",
                 description="Judge the formality of the user's response",
                 prompt_dir=os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "User_Sim_Judge", "FormalityJudge.txt")
                 ):

        super().__init__(name, description)
        self.llm_call = llm_call
        self.prompt = PromptTemplate(template=load_file_utf8(prompt_dir))

    def judge(self, history: Union[dict, List[dict]], counter: Union[Counter, List[Counter]]):
        # history:{"turn_j": {"crs": "", "user": "", "action": [], "reward": ""}
        # Count the number of information provided by the user
        # count_list = []
        if isinstance(history, dict):
            utterance = ""
            i = 1
            for key in history.keys():
                if "turn_" in key:
                    utterance += f"{i}. {history[key]['user']}\n"
                    i += 1

            counter.update(self.judge_formality(utterance))
        elif isinstance(history, list):
            turn_count = [0] * len(history)
            j = 0
            utterance = ""
            for i in range(len(history)):
                for key in history[i].keys():
                    if "turn_" in key:
                        utterance += f"{j + 1}. {history[i][key]['user']}\n"
                        j += 1
                turn_count[i] = j

            result = self.judge_formality(utterance)
            for i in range(len(history)):
                if i == 0:
                    counter[i].update(result[:turn_count[i]])
                else:
                    counter[i].update(result[turn_count[i - 1]:turn_count[i]])



    def judge_formality(self, utterance) -> list:
        llm_input = self.prompt(utterance=utterance)
        llm_output = self.llm_call(llm_input)
        # get the count list
        llm_output = llm_output[llm_output.find("["):llm_output.find("]") + 1]
        # change the string to list
        count_list = eval(llm_output)
        return count_list


class CRS_Rater:
    """
    CRS rater
    """
    def __init__(self,
                 llm_call: LLM_call,
                 prompt_dir=os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "CRS_Rater", "CRS_Rater.txt"),
                 rating_metrics=None,
                 ):
        if rating_metrics is None:
            rating_metrics = ["naturalness", "understandability", "recommendation", "recommendation_subjective"]
        self.llm_call = llm_call
        self.prompt = PromptTemplate(template=load_file_utf8(prompt_dir))
        self.rating_metrics = rating_metrics

        self.num_rate = len(rating_metrics)
        self.rate_standard = ""
        for i in range(self.num_rate):
            self.rate_standard += f"{i+1}. {rating_metrics[i]}\n"
            self.rate_standard += load_file_utf8(os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "CRS_Rater", f"{rating_metrics[i]}.txt"))

    def llm_output2rate(self, llm_output):
        # 每个分数由[rating_metrics[i]][/rating_metrics[i]]包括
        rate_dict = {}
        for i in range(self.num_rate):
            rate_dict[self.rating_metrics[i]] = \
                llm_output.split(f"[{self.rating_metrics[i]}]")[1].split(f"[/{self.rating_metrics[i]}]")[0]
            rate_dict[self.rating_metrics[i]] = int(rate_dict[self.rating_metrics[i]])
        return rate_dict

    def rate(self, dialogue, user_profile):
        llm_input = self.prompt(dialogue=dialogue,
                                user_profile=user_profile,
                                num_rate=self.num_rate,
                                rate_standard=self.rate_standard)
        llm_output = self.llm_call(llm_input)
        rate_dict = self.llm_output2rate(llm_output)
        return rate_dict


class User_comparator:
    """
    User compare
    """
    def __init__(self,
                 llm_call: LLM_call,
                 prompt_dir=os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "User_Sim_Compare", "User_compare.txt"),
                 compare_metrics=None,
                 ):
        if compare_metrics is None:
            compare_metrics = ["naturalness", "clarity", "adaptability", "relevance", "roleplay", "realism"]
        self.llm_call = llm_call
        self.prompt = PromptTemplate(template=load_file_utf8(prompt_dir))
        self.compare_metrics = compare_metrics

        self.num_metrics = len(compare_metrics)
        self.compare_standard = ""
        for i in range(self.num_metrics):
            self.compare_standard += f"{i+1}. {compare_metrics[i]}\n"
            self.compare_standard += load_file_utf8(os.path.join(os.path.dirname(__file__), "prompts", "evaluation", "User_Sim_Compare", f"{compare_metrics[i]}.txt"))

    def llm_output2rate(self, llm_output):
        # 每个分数由[compare_metrics[i]][/compare_metrics[i]]包括
        rate_dict = {}
        for i in range(self.num_metrics):
            try:
                rate_dict[self.compare_metrics[i]] = \
                llm_output.split(f"[{self.compare_metrics[i]}]")[1].split(f"[/{self.compare_metrics[i]}]")[0]
                rate_dict[self.compare_metrics[i]] = int(rate_dict[self.compare_metrics[i]])
            except:
                rate_dict[self.compare_metrics[i]] = 0
        try:
            rate_dict["overall"] = llm_output.split("[overall]")[1].split("[/overall]")[0]
            rate_dict["overall"] = int(rate_dict["overall"])

        except:
            rate_dict["overall"] = 0
        return rate_dict

    def compare(self, dialogue_A, dialogue_B):
        llm_input = self.prompt(dialogue_A=dialogue_A,
                                dialogue_B=dialogue_B,
                                num_metrics=self.num_metrics,
                                compare_standard=self.compare_standard)
        # print(llm_input)
        llm_output = self.llm_call(llm_input)
        # print(llm_output)
        compare_dict = self.llm_output2rate(llm_output)
        return compare_dict


if __name__ == "__main__":
    llm_call = OPENAI_call(llm_model="gpt-4o")
    llm_call.set_max_tokens(9000)

    # infoAmountJudger = InfoAmountJudger(llm_call)
    # counter1 = Counter("infoAmount")
    #
    # lengthJudger = LengthJudger()
    # counter2 = Counter("Length")
    #
    # formalJudger = FormalityJudger(llm_call)
    # counter3 = Counter("Formality")
    #
    openai.api_base = ""
    openai.api_key = ""
    # history = {
    #     "turn_1": {"crs": "", "user": "推荐点快餐便当吧。", "action": [], "reward": ""},
    #     "turn_2": {"crs": "", "user": "请推荐一些适合午餐的快餐便当或韩料，最好在世贸天阶附近。我们现在有时间，希望能尝尝美味的餐点。", "action": [], "reward": ""},
    #     "turn_3": {"crs": "", "user": "请推荐一些适合庆祝升职的快餐。", "action": [], "reward": ""},
    #     "turn_4": {"crs": "", "user": "推荐点吃的。", "action": [], "reward": ""},
    # }
    # infoAmountJudger.judge(history, counter1)
    # lengthJudger.judge(history, counter2)
    # formalJudger.judge(history, counter3)
    # print(counter1)
    # print(counter2)
    # print(counter3)
    dialog_A = {"turn_1": {"crs": "", "user": "附近有推荐的鲁菜或炸鸡吗？  "}, "turn_2": {"crs": "当然可以！如果您在找鲁菜，可以试试“老山东鲁菜馆”，他们的特色菜包括五香驴肉和糖醋鲤鱼，评价不错，环境也很舒适。价格大约在每人100元左右。\n\n如果您想吃炸鸡，可以考虑“红烧炸鸡店”，他们的炸鸡外脆里嫩，搭配多种酱料，价格在每份30-50元之间。 \n\n如果您需要具体地址或更多选择，请告诉我！", "user": "我不喜欢五香驴肉和糖醋鲤鱼，想找高档炸鸡店。  "}, "turn_3": {"crs": "好的！如果您在找高档炸鸡店，我推荐“炸鸡与啤酒”，他们的手工炸鸡使用新鲜食材，搭配多种独特酱料，非常受欢迎，价格大约在每份60-100元之间。此外，店内环境优雅，非常适合聚会。\n\n另外还有“食尚炸鸡”，他们的炸鸡口味丰富，有创意的调味，价格也在每份70-120元。\n\n如果您需要地址或更多选择，请告诉我！", "user": "附近有好吃的鲁菜炸鸡店吗？  "}, "turn_4": {"crs": "当然！我推荐“鲁味炸鸡坊”，他们把鲁菜的风味与炸鸡结合，特别受欢迎的菜品包括酱爆炸鸡和鲁式炸鸡翅，口感独特，价格在每份50-80元之间。环境也很不错，适合聚餐。\n\n另外，“鲁风炸鸡”也很值得一试，他们的炸鸡使用传统鲁菜调味，价格在每份60-90元，评价也很好。\n\n如果您需要更多信息或地址，请告诉我！", "user": "请给我鲁味炸鸡坊和鲁风炸鸡的地址。  "}, "turn_5": {"crs": "鲁味炸鸡坊的地址是：XXXX路123号。\n\n鲁风炸鸡的地址是：XXXX街456号。\n\n如果您需要导航或更多信息，请随时告诉我！", "user": "谢谢推荐，我去试试。  "}}
    dialog_B = {"turn_1": {"crs": "", "user": "我正在寻找一些食物来尝试。我比较喜欢披萨面团、番茄酱、奶酪，以及中国风味的配料，比如叉烧或腊肠，还有面条。希望能找到符合我的口味的美食！"}, "turn_2": {"crs": "您可以尝试一下“中式披萨”！这些披萨通常使用披萨面团、番茄酱和奶酪，搭配中国风味的配料，如叉烧、腊肠等，甚至可以加入一些特色面条，从而结合不同风味。此外，也可以尝试将这些元素作为意大利面的配料使用，创造出融合风味的意面佳肴。希望您喜欢！", "user": "我不太喜欢您推荐的这些食物。"}, "turn_3": {"crs": "当然，没问题！如果您喜欢中国风味和面条，您可以试试“四川担担面”或者“红烧牛肉面”，它们都有丰富的调味和绝佳的口感。如果您喜欢披萨面的感觉，可以尝试“葱油饼夹炸鸡”，结合了中式刺激的味道和西式的爽脆口感。希望这些推荐更符合您的口味！", "user": "我不太喜欢您推荐的这些食物。"}, "turn_4": {"crs": "抱歉没有推荐到您喜欢的食物！也许您可以试试“锅贴”或者“煎饺”，它们有香脆的外皮和丰富的馅料，可能正好符合您的口味。或者，如果您感兴趣，也可以尝试“拉面伴鲍鱼汁”，是一道以面条为主的美味中餐。希望这些选项能更加合您的心意！", "user": "我不太喜欢您推荐的这些食物。"}, "turn_5": {"crs": "抱歉没有推荐到您喜欢的食物！如果您喜欢面条和中国风味，您可以尝试“汤包面”或“锅巴卷”。另外，您也可以试试“花式煎饼果子”，这是一种结合了煎饼与多种食材的美味小吃，也许这会更符合您的喜好！希望这些推荐能满足您的味蕾。", "user": "我不太喜欢您推荐的这些食物。"}}
    dialog_a_txt = ""
    dialog_b_txt = ""
    for key in dialog_A.keys():
        if "turn_" in key:
            dialog_a_txt += key + " " + "\nCRS: \n" + dialog_A[key]["crs"] + "\nUser: \n" + dialog_A[key]["user"] + "\n"
    for key in dialog_B.keys():
        if "turn_" in key:
            dialog_b_txt += key + " " + "\nCRS: \n" + dialog_B[key]["crs"] + "\nUser: \n" + dialog_B[key]["user"] + "\n"


    print(dialog_a_txt)
    print(dialog_b_txt)
    comparator = User_comparator(llm_call)
    print(comparator.compare(dialog_a_txt, dialog_b_txt))
