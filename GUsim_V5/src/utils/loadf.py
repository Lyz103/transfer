import re
import json
import random
import heapq
from typing import Dict, List, Tuple
import re
import ast

def parse_mixed_list(text):
    if isinstance(text, list):
        return text
    # 去除首尾中括号
    text = text.strip()[1:-1]

    # 分割所有以逗号分隔的条目（允许有空格）
    items = [item.strip() for item in text.split(";")]

    # 对每一项判断是否已经被英文引号包裹，如果没有就加上
    quoted_items = []
    for item in items:
        # 只保留内容，加上引号（避免双引号或单引号冲突）
        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
            quoted_items.append(item.strip())
        else:
            quoted_items.append(f"'{item}'")

    # 拼接为标准 Python 列表字符串并解析
    fixed_text = "[" + ", ".join(quoted_items) + "]"
    return ast.literal_eval(fixed_text)



def load_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def load_file_utf8(path: str) -> str:
    with open(path, "r", encoding='utf-8') as f:
        return f.read()



def parse_actions_woprob(action_str):
    data = json.loads(action_str)  # 解析 JSON 字符串
    action_text = data.get("动作", "")  # 获取 "动作" 对应的字符串
    reason = data.get("原因", "")  # 获取 "原因"
    think = data.get("思考", "")
    actions = parse_mixed_list(action_text)
    reason = parse_mixed_list(reason)
    return actions, reason, think

def parse_actions_withprob(action_str):
    data = json.loads(action_str)  # 解析 JSON 字符串
    action_text = data.get("动作", "")  # 获取 "动作" 对应的字符串
    reason = data.get("原因", "")  # 获取 "原因"
    reflect = data.get("思考", "")
    pattern = r'([\w\u4e00-\u9fff]+):\s*(\d+)'  # 修改正则，避免匹配到括号
    pattern = r'[\[|\{]?\s*[\'\"]?([\w\u4e00-\u9fff]+)[\'\"]?:\s*(\d+)\s*[\]|\}]?'  
    matches = re.findall(pattern, action_text)
    actions = {action: int(score) for action, score in matches}
    return actions, reason, reflect

def get_topk(scores, k):
    # 取得前k大的元素及其下标，按得分从高到低排序
    if k == 1:
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        sampled_idx = random.choice(max_indices)
        return [sampled_idx], [max_score]

def top_p_sample(action_dict: Dict[str, float], reason: str, p: float = 0.4) -> Tuple[List[str], List[str]]:
    """
    根据给定的动作字典和概率阈值 p，选择累积概率达到 p 的动作，并返回对应的动作列表和原因列表。

    参数:
        action_dict (Dict[str, float]): 动作及其对应的分数（权重）。
        reason (str): 原因字符串，格式为 "[reason1;reason2;...]"。
        p (float): 累积概率阈值，默认值为 0.4。

    返回:
        Tuple[List[str], List[str]]: 
            - 第一个元素是按累积概率选择的动作名称列表。
            - 第二个元素是与动作数量对应的原因列表。
    """
    # 检查输入字典是否为空
    if not action_dict:
        return [], []

    # 按分数从高到低排序动作字典
    sorted_actions: List[Tuple[str, float]] = sorted(
        action_dict.items(), key=lambda x: x[1], reverse=True
    )

    total_score: float = sum(action_dict.values())  # 计算总分
    cumulative_score: float = 0.0
    selected_actions: List[Tuple[str, float]] = []

    # 遍历排序后的动作，直到累积概率达到阈值 p
    for action, score in sorted_actions:
        cumulative_score += score
        selected_actions.append((action, score))
        if cumulative_score >= total_score * p:
            break

    # 提取选中的动作名称和对应的权重
    sampled_actions, weights = zip(*selected_actions)

    # 解析原因字符串并截取与动作数量对应的部分
    # parsed_reasons: List[str] = reason.strip("[]").split(";")
        # 去掉字符串两端的方括号
    stripped_reason = reason.strip("[]")
    
    # 使用正则表达式按英文分号或中文分号分割
    split_reasons = re.split(r'[；;]', stripped_reason)
    
    # 清理每个分割后的字符串，去除前后空格，并过滤掉空字符串
    parsed_reasons = [item.strip() for item in split_reasons if item.strip()]
    sampled_reasons: List[str] = parsed_reasons[:len(sampled_actions)]

    return list(sampled_actions), sampled_reasons




def put_action_chat_together(chat_his, action_his):
    res = ""
    # print("chat_his:", chat_his)
    turns_chat = re.findall(r'(\[turn \d+\][^\[]+)', chat_his.strip(), re.DOTALL)
    turns_chat = [turn.strip() for turn in turns_chat]
    # print("turns_chat:", turns_chat)
    turns_action = action_his.strip().split("\n")
    turns_chat_user = turns_chat[::2]
    turns_chat_sys = turns_chat[1::2]
    if len(turns_chat) == 0:
        return None
    for turn_action, turn_chat_user, turn_chat_sys in zip(turns_action, turns_chat_user, turns_chat_sys):
        # print("turn_action:", turn_action)
        # print("turn_chat_user:", turn_chat_user)
        # print("turn_chat_sys:", turn_chat_sys)

        match = re.match(r"\[turn (\d+)\]\s*(.*)", turn_action)
        turn_number, text = match.groups()
        res += (turn_chat_user + ' ' + text + '\n' + turn_chat_sys + '\n')
    # print("res:", res)
    return res
