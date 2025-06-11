import os
import sys
import json
import re
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
from utils import OPENAI_call
from evaluator.eval_prompts import language, recommendation, action, evaluator_prompt, corrector
# 先把错误修正过来，然后再进行测试


import json
from typing import List, Tuple, Dict, Any

def data_process_for_correction(data_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    从对话数据文件中提取用户语句、CRS 回复以及用户画像。

    参数:
        data_path (str): JSON 格式数据文件的路径，每行一个对话记录。

    返回:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
            - 用户语句列表，每个元素包含语句内容和原始索引
            - CRS 回复列表，每个元素包含回复内容和原始索引
            - 用户画像列表
    """
    with open(data_path, 'r', encoding='utf-8') as file:
        datas = [json.loads(line.strip()) for line in file if line.strip()]

    user_utterances: List[Dict[str, Any]] = []
    crs_utterances: List[Dict[str, Any]] = []
    profiles: List[str] = []

    for idx, data in enumerate(datas):
        # 提取用户画像
        profile = data.get("user_profile", "")
        profiles.append(profile)

        # 获取最大轮次
        max_turn = 0
        while f"turn_{max_turn + 1}" in data:
            max_turn += 1

        # 提取交替的用户语句与 CRS 回复
        for turn_num in range(2, max_turn + 1):
            current_turn_key = f"turn_{turn_num}"
            prev_turn_key = f"turn_{turn_num - 1}"

            if prev_turn_key in data and 'user' in data[prev_turn_key]:
                user_utterances.append({
                    "text": data[prev_turn_key]['user'],
                    "original_index": idx
                })

            if current_turn_key in data and 'crs' in data[current_turn_key]:
                crs_utterances.append({
                    "text": data[current_turn_key]['crs'],
                    "original_index": idx
                })

    return user_utterances, crs_utterances, profiles

def group_utterances(utterances: List[Dict[str, Any]], group_size: int = 5) -> List[List[Dict[str, Any]]]:
    """
    将对话列表按指定大小分组。

    参数:
        utterances (List[str]): 用户语句列表。
        group_size (int): 每组包含的语句数量，默认为5。

    返回:
        List[List[str]]: 分组后的二维列表。
    """
    grouped = []
    for i in range(0, len(utterances), group_size):
        grouped.append(utterances[i:i + group_size])
    return grouped

def correct(utterances: str):
    grouped_utterances = group_utterances(utterances_user, group_size=5)
    
    strs_to_be_corrected = []
    for i in range(len(grouped_utterances)):
        temp_str = ""
        for j in range(len(grouped_utterances[i])):
            temp_str += str(j + 1) + "." + grouped_utterances[i][j]["text"] + "\n"
        strs_to_be_corrected.append(temp_str)
    
    correct_res = []
    for i in range(len(strs_to_be_corrected)):
        input_message = {"utterances": strs_to_be_corrected[i]}
        prompt = corrector.render(input_message)
        
        # 调用 LLM 进行纠错
        corrected_result = llm_judge(prompt).strip().split("修正后的完整语句：")[-1].strip()
        correct_res.append(corrected_result)
        print(f"Corrected Result for Group {i+1}:\n{corrected_result}")
        
        # 这里可以将纠错结果保存或进一步处理
        # print(f"Group {i+1} Prompt:\n{prompt}\n")
    
    return correct_res





def eval(utterances, profiles, llm_judge):
    results = []
    for i in range(len(utterances)):
        input_message = {}
        input_message.update({"dialogue": utterances[i], "user_profile": profiles[i]})
        input_message.update({"language": language,"recommendation": recommendation, "action": action})
        prompt = evaluator_prompt.render(input_message)
        print(f"Prompt for Dialogue {i+1}:\n{prompt}\n")
        print("\n" * 5)
        
        result = llm_judge(prompt)

        print(f"Result for Dialogue {i+1}:\n{result}\n")
        
        # 匹配评分数字
        naturalness_score = re.search(r'\[naturalness\](\d+)\[/naturalness\]', result)
        recommendation_score = re.search(r'\[recommendation\](\d+)\[/recommendation\]', result)
        understandability_score = re.search(r'\[understandability\](\d+)\[/understandability\]', result)

        # 提取并打印结果
        scores = {
            "naturalness": naturalness_score.group(1) if naturalness_score else None,
            "recommendation": recommendation_score.group(1) if recommendation_score else None,
            "understandability": understandability_score.group(1) if understandability_score else None
        }
        
        results.append(scores)
        print(f"Dialogue {i+1} Scores: {scores}")
    
    return results





if __name__ == "__main__":

    data_path = "/data/liyuanzi/HUAWEI/User_simulator/GUsim_V3/test/OurUser_gpt-4o-mini_deepseek-reasoner_Prob_Sample_BaseCRSgpt-4o-mini_carelessness/history_5.jsonl"
    llm_judge = OPENAI_call("gpt-4.1-mini")
    utterances_user, utterances_crs, profiles_u = data_process_for_correction(data_path)

    # correct_res = correct(utterances_user)
    
    # # results = eval(utterances, profiles, llm_judge)

    # # 保存纠错结果到文件
    output_path = "/data/liyuanzi/HUAWEI/GUsim_V3/src/evaluator/corrected_results.json"
    with open(output_path, 'r', encoding='utf-8') as output_file:
        correct_res = json.load(output_file)

    # print(f"Corrected results have been saved to {output_path}")

    utterances_user_corrected = []

    for i in range(len(correct_res)):
        # 取出字符串并按数字编号进行分割
        raw_text = correct_res[i]
        # 使用正则表达式按编号分割句子

        pattern = r'\d+\.\s'  # 匹配类似 '1. ', '2. ' 等格式
        parts = re.split(pattern, raw_text)

        # 去除第一个空元素，并去除每句前后空白
        sentences = [sentence.strip() for sentence in parts if sentence.strip()]

        # 输出结果验证
        print(sentences)
        utterances_user_corrected.extend(sentences)
    
    # assert len(utterances_user_corrected) == len(utterances_user), "Corrected utterances count does not match original count."
    utterances = []
    profiles = []
    temp_str = ""
    j = 0

    for i in range(len(utterances_user_corrected)):
        if utterances_user[i]["original_index"] != j:
            temp_str = ""
            j += 1
        profiles.append(profiles_u[j])
        temp_str += "用户: " + utterances_user_corrected[i] + "\n"
        temp_str += "推荐系统: " + utterances_crs[i]["text"] + "\n"
        utterances.append(temp_str.strip())
    
    # assert len(utterances) == len(profiles), "Utterances count does not match profiles count."

    eval_results = eval(utterances, profiles, llm_judge)
    print("eval_results:", eval_results)

    language_score = sum(int(res['naturalness']) for res in eval_results) / len(eval_results)
    recommendation_score = sum(int(res['recommendation']) for res in eval_results) / len(eval_results)
    understandability_score = sum(int(res['understandability']) for res in eval_results) / len(eval_results)

    print(f"Average Language Score: {language_score:.2f}")
    print(f"Average Recommendation Score: {recommendation_score:.2f}")
    print(f"Average Understandability Score: {understandability_score:.2f}")
