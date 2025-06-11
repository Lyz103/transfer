import os
import sys
import json
import re
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
from utils import OPENAI_call
from evaluator.eval_prompts import language, recommendation, action, evaluator_prompt



def data_process(data_path, j):
    datas = []
    utterances = []
    profiles = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            datas.append(data)
    for data in datas:
        temp_str = ""
        contexts = data['context']
        for i in range(len(contexts)):
            if 'user' in contexts[i]:
                temp_str += "用户: " + contexts[i]['user'] + "\n"
            if 'crs' in contexts[i]:
                temp_str += "推荐系统: " + contexts[i]['crs'] + "\n"
        utterances.append([temp_str, data['crs1'], data['crs2']])
    
    with open("synthesized_profiles.jsonl",'r', encoding='utf-8') as f:
        for line in f:
            profile = json.loads(line.strip())
            profiles.append(profile)
    
    profile = profiles[j]
            


        
    
    return utterances, profile


def eval(utterances, profile, llm_judge):
    results = []
    for i in range(len(utterances)):
        utt, cr1, cr2 = utterances[i]
        input_message = {}
        input_message.update({"context": utt, "user_profile": profile})
        input_message.update({"language": language, "recommendation": recommendation, "action": action})
        input_message.update({"CRS1_reply": cr2, "CRS2_reply": cr1})
        prompt = evaluator_prompt.render(input_message)
        # print(f"Evaluating dialogue {i+1} with prompt:\n{prompt}\n")
        
        result = llm_judge(prompt)
        # print(f"Dialogue {i+1} evaluation result:\n{result}\n")
        # 匹配评分数字

        naturalness_score = re.search(r'\[naturalness\](-?\d+)\[/naturalness\]', result)
        recommendation_score = re.search(r'\[recommendation\](-?\d+)\[/recommendation\]', result)
        understandability_score = re.search(r'\[understandability\](-?\d+)\[/understandability\]', result)

        # 提取并打印结果
        scores = {
            "naturalness": naturalness_score.group(1) if naturalness_score else None,
            "recommendation": recommendation_score.group(1) if recommendation_score else None,
            "understandability": understandability_score.group(1) if understandability_score else None
        }
        
        results.append(scores)
        print(f"Dialogue {i+1} Scores: {scores}")

    return results


def main():
    pass

if __name__ == "__main__":

    sum_eval_results = []
    for i in range(40):
        try:
            print(f"Processing data file {i}...")
            data_path = f"/data/liyuanzi/HUAWEI/GUsim_V5/src/evaluator/eval_data/A4ominiVSB4o-mini_U4o-mini/output{i}.jsonl"
            print("data_path", data_path)
            llm_judge = OPENAI_call("gpt-4o-ca")
            utterances, profile = data_process(data_path, i)

            print("utterances:", len(utterances))
            print("profile:", profile)
            eval_results = eval(utterances, profile, llm_judge)

            print("eval_results:", eval_results)
            sum_eval_results.extend(eval_results)
        except Exception as e:
            print(f"Error processing data file {i}: {e}")
            continue

    eval_results = sum_eval_results
    # Count occurrences of 1, 0, and -1 for each score type
    score_counts = {
        "understandability": {"1": 0, "0": 0, "-1": 0},
        "naturalness": {"1": 0, "0": 0, "-1": 0},
        "recommendation": {"1": 0, "0": 0, "-1": 0},
    }
    # 统计各个评分的数量
    for res in eval_results:
        for key in score_counts:
            val = res.get(key, None)
            if val in ["1", "0", "-1"]:
                score_counts[key][val] += 1

    # 计算平均分
    total = len(eval_results)

    understandability_score1 = score_counts["understandability"]["1"] / total * 100
    understandability_score0 = score_counts["understandability"]["0"] / total * 100
    understandability_scoref1 = score_counts["understandability"]["-1"] / total * 100

    naturalness_score1 = score_counts["naturalness"]["1"] / total * 100
    naturalness_score0 = score_counts["naturalness"]["0"] / total * 100
    naturalness_scoref1 = score_counts["naturalness"]["-1"] / total * 100

    recommendation_score1 = score_counts["recommendation"]["1"] / total * 100
    recommendation_score0 = score_counts["recommendation"]["0"] / total * 100
    recommendation_scoref1 = score_counts["recommendation"]["-1"] / total * 100

    # 打印结果
    print("Understandability Scores:")
    print(f"  Score 1: {understandability_score1:.1f}")
    print(f"  Score 0: {understandability_score0:.1f}")
    print(f"  Score -1: {understandability_scoref1:.1f}")

    print("\nNaturalness Scores:")
    print(f"  Score 1: {naturalness_score1:.1f}")
    print(f"  Score 0: {naturalness_score0:.1f}")
    print(f"  Score -1: {naturalness_scoref1:.1f}")

    print("\nRecommendation Scores:")
    print(f"  Score 1: {recommendation_score1:.1f}")
    print(f"  Score 0: {recommendation_score0:.1f}")
    print(f"  Score -1: {recommendation_scoref1:.1f}")

