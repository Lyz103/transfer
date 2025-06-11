import os
import sys
import json
import re
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
from utils import OPENAI_call
from evaluator.eval_prompts import language, recommendation, action, evaluator_prompt



def data_process(data_path):
    datas = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            datas.append(data)
    
    utterances = []
    profiles = []
    for data in datas:
        temp_dialogue = ""
        temp_profile = ""
        max_turn = 1
        for i in range(1, 6):
            if f"turn_{i}" in data:
                max_turn = i
        for i in range(2, max_turn + 1):
            if f"turn_{i}" in data:
                temp_dialogue += f"用户: {data[f'turn_{i - 1}']['user']}\n"
                temp_dialogue += f"推荐系统: {data[f'turn_{i}']['crs']}\n"
                utterances.append(temp_dialogue.strip())
                profiles.append(data['user_profile'])
    
    return utterances, profiles


def eval(utterances, profiles, llm_judge):
    results = []
    for i in range(len(utterances)):
        input_message = {}
        input_message.update({"dialogue": utterances[i], "user_profile": profiles[i]})
        input_message.update({"language": language, "recommendation": recommendation, "action": action})
        prompt = evaluator_prompt.render(input_message)
        
        result = llm_judge(prompt)
        
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

    data_path = "/data/liyuanzi/HUAWEI/User_simulator/GUsim_V1_gendata/test/OurUser_gpt-4o-mini_gpt-4o-mini_Prob_Sample_BaseCRSgpt-3.5-turbo-ca/history_5.jsonl"
    llm_judge = OPENAI_call("gpt-4.1-mini")
    utterances, profiles = data_process(data_path)

    eval_results = eval(utterances[:100], profiles[:100], llm_judge)

    print("eval_results:", eval_results)

    language_score = sum(int(res['naturalness']) for res in eval_results) / len(eval_results)
    recommendation_score = sum(int(res['recommendation']) for res in eval_results) / len(eval_results)
    understandability_score = sum(int(res['understandability']) for res in eval_results) / len(eval_results)

    print(f"Average Language Score: {language_score:.2f}")
    print(f"Average Recommendation Score: {recommendation_score:.2f}")
    print(f"Average Understandability Score: {understandability_score:.2f}")
    
