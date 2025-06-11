# Evaluation
# -*- coding: utf-8 -*-
# @Author : LuyuChen
# @Time : 2024/7/23 14:28
import logging
import os
import sys
import json
from typing import List

import openai

LOGGER = logging.getLogger("UserSimulator")
sys.path[0] = os.path.join(os.path.dirname(__file__), '..', '..')
print(sys.path[0])
from config.Evaluation_config import EVAL_CONFIG
from datatypes import SimulatedUserAct, Counter

from tqdm import tqdm
import random


class Evaluation:
    def __init__(self, config: dict = EVAL_CONFIG):
        print("Test Start!")
        self.interaction_config = config["interaction_config"]

        # basic configuration
        self.user_agent = config["user_agent"]
        self.crs_agent = config["crs_agent"]
        self.language = config["language"]
        self.encoding = config["encoding"]
        self.save_path = config["save_path"]

        # interaction configuration
        self.do_interaction = self.interaction_config["do_interaction"]
        self.interaction_num = self.interaction_config["interaction_num"]
        self.max_turn = self.interaction_config["max_turn"]
        self.init_type = self.interaction_config["init_type"]

        # compare configuration
        self.do_compare = config["compare_config"]["do_compare"]
        self.comparator = config["compare_config"]["comparator"]
        self.main_agent = config["compare_config"]["main_agent"]
        self.compare_times = config["compare_config"]["compare_times"]

        # evaluation configuration
        self.do_evaluation = config["evaluation_config"]["do_evaluation"]
        self.metrics = config["evaluation_config"]["metrics"]
        self.distinct_ngram = config["evaluation_config"]["distinct_ngram"]
        self.evaluated_max_turn = config["evaluation_config"]["evaluated_max_turn"]
        self.judge_boost = config["evaluation_config"]["judge_boost"]
        self.start_from_raw = config["evaluation_config"]["start_from_raw"]

        # rating configuration
        self.do_rating = config["rating_config"]["do_rating"]
        self.rater = config["rating_config"]["rater"]

        self.counter = []

    def calculate_distinct_ngram(self, history_list: List[dict], counter: Counter, ngram: List[int] = None):
        if ngram is None:
            ngram = [1, 2, 3, 4]
        total_words = ""
        for history in history_list:
            for key in history.keys():
                if "turn_" in key:
                    total_words += history[key]["user"]
                    if "en" in self.language.lower():
                        total_words += " "
                    elif "zh" or "ch" in self.language.lower():
                        total_words += ""
        total_words = total_words.strip()
        if "en" in self.language.lower():
            total_words = total_words.split(" ")
        elif "zh" or "ch" in self.language.lower():
            total_words = list(total_words)
        else:
            raise ValueError("Only support Chinese and English")
        for n in ngram:
            ngram_list = []
            distinct_ngram = 0
            for i in range(len(total_words) - n + 1):
                now_gram = "".join(total_words[i:i + n])
                if now_gram not in ngram_list:
                    ngram_list.append(now_gram)
                    distinct_ngram += 1
            # Distinct n-gram = distinct n-gram / total n-gram
            counter.update_dict({f"{n}-gram distinct": distinct_ngram / (len(total_words) - n + 1)})

    def update_history(self, history, user_response, crs_response, turn):
        if type(user_response) == SimulatedUserAct:
            history.update({f"turn_{turn}": {"crs": crs_response, "user": user_response.user_response,
                                             "action": [x.name for x in user_response.action],
                                             "reward": user_response.reward}})
        else:
            history.update({f"turn_{turn}": {"crs": crs_response, "user": user_response}})
        return history

    def save_history(self, history, user_profile, save_path):
        # 追加写
        history.update({"user_profile": user_profile})
        with open(save_path, "a+", encoding=self.encoding) as f:
            f.write(json.dumps(history, ensure_ascii=False) + "\n")

    def load_history(self, path) -> List[dict]:
        history_list = []
        with open(path, "r", encoding=self.encoding) as f:
            for line in f:
                his = json.loads(line)
                history_list.append(his)

        return history_list

    def give_history_id(self, path):
        # give the id to the history
        history_path = os.path.join(path, f"history_{self.max_turn}.jsonl")
        with open(history_path, "r", encoding=self.encoding) as f:
            lines = f.readlines()
        with open(history_path, "w", encoding=self.encoding) as f:
            for i, line in enumerate(lines):
                his = json.loads(line)
                his.update({"id": i + 1})
                line = json.dumps(his, ensure_ascii=False) + "\n"
                f.write(line)

    def interact(self):
        for user in self.user_agent:
            for crs in self.crs_agent:
                LOGGER.info(f"Start interaction between {user.name} and {crs.name}")
                # create the save path
                path = os.path.join(self.save_path, f"{user.name}_{crs.name}")
                if not os.path.exists(path):
                    os.makedirs(path)

                for i in range(self.interaction_num):
                    # try:
                    user.refresh_profile(i)
                    # user.clear_interaction_history()
                    crs.clear_interaction_history()
                    LOGGER.info(f"Start interaction {i + 1}")
                    history = {}
                    user_response, _ = user()
                    history = self.update_history(history, user_response, "", 1)
                    if type(user_response) == SimulatedUserAct:
                        crs_input = user_response.user_response
                    else:
                        crs_input = user_response
                    LOGGER.info(f"Turn 1:\nUser: {crs_input}")
                    for j in range(2, self.max_turn + 1):

                        crs_response = crs(crs_input)
                        if "our" in user.name.lower():
                            user._update_memory(crs_response)
                            user_response, if_end = user(j)
                        else :
                            user_response = user(crs_response)
                        history = self.update_history(history, user_response, crs_response, j)

                        if type(user_response) == SimulatedUserAct:
                            crs_input = user_response.user_response
                        else:
                            crs_input = user_response
                        if if_end:
                            break
                        LOGGER.info(f"Turn {j}:\ncrs: {crs_response}\nuser: {crs_input}")
                        if str(user_response) == "None":
                            break

                    # save the history
                    user_profile, _, _ = user.memory.recall()
                    self.save_history(history,
                                    user_profile=user_profile,
                                    save_path=os.path.join(path, f"history_{self.max_turn}.jsonl"))
                    # user.refresh_profile(i)
                    # except Exception as e:
                    #     LOGGER.error(f"Error in {path}: {e}")
                    #     continue

                    # give the id to the history
                    self.give_history_id(path)

    def save_result(self, result, save_path):
        with open(save_path, "a", encoding=self.encoding) as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def save_total_result(self, save_path_raw, save_path):
        # the information won't be lost, history result still remain
        if os.path.exists(save_path):
            result = json.load(open(save_path, "r", encoding=self.encoding))
        else:
            result = {}
        with open(save_path_raw, "r", encoding=self.encoding) as f:
            for line in f:
                res = json.loads(line)
                if str(res["history_id"]) not in result.keys():
                    result.update({str(res["history_id"]): res})
                else:
                    result[str(res["history_id"])].update(res)
        for ctr in self.counter:
            result.update({ctr.name: ctr.counter})
        with open(save_path, "w", encoding=self.encoding) as f:
            f.write(json.dumps(result, ensure_ascii=False))
        # delete the raw result
        # os.remove(save_path_raw)

    def evaluate(self):
        for user in self.user_agent:
            for crs in self.crs_agent:
                self.counter = []
                LOGGER.info(f"Start evaluation between {user.name} and {crs.name}")
                # load the history
                path = os.path.join(self.save_path, f"{user.name}_{crs.name}")
                history_path = os.path.join(path, f"history_{self.evaluated_max_turn}.jsonl")
                result_path_raw = os.path.join(path, f"result_{self.evaluated_max_turn}_raw.jsonl")
                result_path = os.path.join(path, f"result_{self.evaluated_max_turn}.json")
                try:
                    self.give_history_id(path)
                    history_list = self.load_history(history_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"History file not found: {history_path}, "
                                            f"Please ensure do the interaction first."
                                            f"or check the parameter evaluated_max_turn.")

                # calculate the metrics
                if self.start_from_raw:
                    try:
                        lines = open(result_path_raw, "r", encoding=self.encoding).readlines()
                    except:
                        lines = []

                    for metric in self.metrics:
                        LOGGER.info(f"Start calculate the metric: {metric.name}")
                        now_counter = Counter(f"Total {metric.name}")
                        # load the raw result


                        with tqdm(total=len(history_list)) as pbar:
                            for h in history_list:
                                if len(lines) == 0:
                                    temp_counter = Counter(f"Temp counter")
                                    metric.judge(h, temp_counter)
                                    self.save_result({"history_id": h["id"], f"{metric.name}": temp_counter.counter},
                                                     result_path_raw)
                                    now_counter.update_dict(temp_counter.counter)
                                    pbar.update(1)
                                else:
                                    line = lines.pop(0)
                                    res = json.loads(line)
                                    now_counter.update_dict(res[f"{metric.name}"])
                                    pbar.update(1)

                        self.counter.append(now_counter)

                else:
                    for metric in self.metrics:
                        LOGGER.info(f"Start calculate the metric: {metric.name}")
                        now_counter = Counter(f"Total {metric.name}")
                        with tqdm(total=len(history_list)) as pbar:
                            for h in history_list:
                                print("Before delete:", h)
                                h_copy = h.copy()  # 复制字典，避免修改原始 h

                                m = 0
                                for ii in range(1, 6):
                                    if f"turn_{ii}" in h_copy:
                                        m = ii

                                h_copy.pop(f"turn_{m}", None)  # 删除最后一个存在的 turn_i
                                print("After delete:", h_copy)

                                temp_counter = Counter(f"Temp counter")
                                metric.judge(h, temp_counter)
                                self.save_result({"history_id": h["id"], f"{metric.name}": temp_counter.counter},
                                                 result_path_raw)
                                now_counter.update_dict(temp_counter.counter)
                                pbar.update(1)
                        self.counter.append(now_counter)
                # calculate the distinct n-gram
                if self.distinct_ngram is not None:
                    now_counter = Counter(f"Total distinct n-gram")
                    self.calculate_distinct_ngram(history_list, now_counter, self.distinct_ngram)
                    self.counter.append(now_counter)

                # save the result
                self.save_total_result(save_path_raw=result_path_raw, save_path=result_path)
                LOGGER.info(f"Finish evaluation between {user.name} and {crs.name}")
                self.counter = []

        LOGGER.info("Evaluation finished.")
        self.counter = []

    def compare(self):
        for user in self.user_agent:
            if user == self.main_agent:
                continue
            for crs in self.crs_agent:
                LOGGER.info(f"Start compare between {user.name} and {self.main_agent.name} in crs setting {crs.name}")
                # load the history
                main_user_path = os.path.join(self.save_path, f"{self.main_agent.name}_{crs.name}")
                main_user_history_path = os.path.join(main_user_path, f"history_{self.evaluated_max_turn}.jsonl")
                path = os.path.join(self.save_path, f"{user.name}_{crs.name}")
                history_path = os.path.join(path, f"history_{self.evaluated_max_turn}.jsonl")

                result_path_raw = os.path.join(main_user_path,
                                               f"compare_result_{user.name}_{self.evaluated_max_turn}_raw.jsonl")
                result_path = os.path.join(main_user_path, f"compare_result_{user.name}_{self.evaluated_max_turn}.json")
                try:
                    history_list_main = self.load_history(main_user_history_path)
                    history_list_compare = self.load_history(history_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"History file not found: {history_path} or {main_user_history_path}, "
                                            f"Please ensure do the interaction first."
                                            f"or check the parameter evaluated_max_turn.")

                # compare the user
                with tqdm(total=self.compare_times) as pbar:
                    for compare_times in range(self.compare_times):
                        # 随机选择一个对话
                        h_main = random.choice(history_list_main)
                        h_compare = random.choice(history_list_compare)
                        main_dialogue = {}
                        compare_dialogue = {}
                        for i in range(1, self.evaluated_max_turn + 1):
                            if f"turn_{i}" in h_main.keys() and str(h_main[f"turn_{i}"]["user"]) != 'null':
                                main_dialogue.update({f"turn_{i}": {"crs": h_main[f"turn_{i}"]["crs"],
                                                                    "user": h_main[f"turn_{i}"]["user"]}})

                            else:
                                break
                        for i in range(1, self.evaluated_max_turn + 1):
                            if f"turn_{i}" in h_compare.keys():
                                compare_dialogue.update({f"turn_{i}": {"crs": h_compare[f"turn_{i}"]["crs"],
                                                                       "user": h_compare[f"turn_{i}"]["user"]}})
                            else:
                                break
                        # sequence of the dialogue may affect the result
                        result_1 = self.comparator.compare(dialogue_A=main_dialogue, dialogue_B=compare_dialogue)
                        result_2 = self.comparator.compare(dialogue_A=compare_dialogue, dialogue_B=main_dialogue)
                        result = {}
                        for key in result_1.keys():
                            if result_1[key] == result_2[key]:
                                result.update({key: 0})
                            elif result_1[key] > result_2[key]:
                                result.update({key: 1})
                            elif result_1[key] < result_2[key]:
                                result.update({key: -1})
                        print("h_main:", h_main)
                        self.save_result(result={"history_id": h_main["id"],
                                                 "compared_history_id": h_compare["id"],
                                                 "compare_result": result},
                                         save_path=result_path_raw)
                        pbar.update(1)
                    self.save_total_result(save_path_raw=result_path_raw, save_path=result_path)
                LOGGER.info(f"Finish compare between {user.name} and {self.main_agent.name} in crs setting {crs.name}")

    def rating(self):
        for user in self.user_agent:
            for crs in self.crs_agent:
                LOGGER.info(f"Start rating for interaction between {user.name} and {crs.name}")
                # load the history
                path = os.path.join(self.save_path, f"{user.name}_{crs.name}")
                history_path = os.path.join(path, f"history_{self.evaluated_max_turn}.jsonl")
                result_path_raw = os.path.join(path, f"result_{self.evaluated_max_turn}_raw.jsonl")
                result_path = os.path.join(path, f"result_{self.evaluated_max_turn}.json")
                try:
                    history_list = self.load_history(history_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"History file not found: {history_path}, "
                                            f"Please ensure do the interaction first."
                                            f"or check the parameter evaluated_max_turn.")

                # calculate the rating
                for h in history_list:
                    dialogue = {}
                    for i in range(1, self.evaluated_max_turn + 1):
                        if f"turn_{i}" in h.keys():
                            dialogue.update({f"turn_{i}": {"crs": h[f"turn_{i}"]["crs"]},
                                             "user": h[f"turn_{i}"]["user"]})
                        else:
                            break
                    rating = self.rater.rate(dialogue=dialogue,
                                             user_profile=user.get_profile())
                    self.save_result(result={"history_id": h["id"], "rating": rating},
                                     save_path=result_path_raw)
                self.save_total_result(save_path_raw=result_path_raw, save_path=result_path)
                LOGGER.info(f"Finish rating for interaction between {user.name} and {crs.name}")

    def run(self):
        if self.do_interaction:
            self.interact()
        if self.do_compare:
            self.compare()
        if self.do_evaluation:
            self.evaluate()
        # if self.do_rating:
        #     self.rating()


if __name__ == '__main__':
    print("TEST START!!!")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)
    LOGGER.setLevel(logging.INFO)

    evaluator = Evaluation()
    evaluator.run()
