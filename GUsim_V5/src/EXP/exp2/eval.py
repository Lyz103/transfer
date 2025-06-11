from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import numpy as np
from openai import OpenAI
import re

def call_llm(llm_input):
    # try:
    # print("self.model:", self.model)
    client = OpenAI()
    output = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": llm_input}
        ],
        temperature=0,
        max_tokens=1024,
        seed=42
    )
    # print(output)
    return output.choices[0].message.content
    # except Exception as e:
    #     LOGGER.warning(f"Failed to call LLM model: {e}")
    #     return None






def test_length(test_path):
    with open(test_path + "history_5.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip() for line in lines if line.strip()]

    best_f1 = 0
    best_left = 0
    best_right = 0
    best_y_true = []
    best_y_pred = []

    # 遍历阈值范围
    for left in range(5, 20):
        for right in range(left + 1, 30):  # right > left
            y_true = []
            y_pred = []

            for data in datas:
                try:
                    data_dict = eval(data)
                    profile = eval(data_dict['user_profile'])
                    state = profile['user_behavior']['conciseness']

                    for i in range(6):
                        turn_key = f"turn_{i}"
                        if turn_key in data_dict:
                            turn_value = data_dict[turn_key]['user']
                            length = len(turn_value)

                            # 预测
                            if length >= right:
                                y_pred.append(-1)
                            elif left <= length < right:
                                y_pred.append(0)
                            else:
                                y_pred.append(1)

                            # 真实标签
                            if state == "f1":
                                y_true.append(-1)
                            elif state == "1":
                                y_true.append(1)
                            elif state == "0":
                                y_true.append(0)

                except Exception as e:
                    print(f"Error processing data: {e}")
                    continue

            current_f1 = f1_score(y_true, y_pred, average='macro')
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_left = left
                best_right = right
                best_y_true = y_true
                best_y_pred = y_pred

    # 计算整体 Accuracy 和 Macro Recall
    acc = accuracy_score(best_y_true, best_y_pred)
    macro_recall = recall_score(best_y_true, best_y_pred, average='macro')
    print("===========================Length===================================")
    print("=====================================================================")
    print("======================================================================")
    print("======================================================================")

    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Thresholds: left={best_left}, right={best_right}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    return {
        'accuracy': acc,
        'macro_recall': macro_recall,
        'best_left': best_left,
        'best_right': best_right
    }

def test_formality(test_path):
    with open(test_path + "history_5.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip() for line in lines if line.strip()]
    y_pred = []
    y_true = []
    user_utt = []
    for data in datas:
        try:
            data_dict = eval(data)
            profile = eval(data_dict['user_profile'])
            state = profile['user_behavior']['formality']
            for i in range(6):
                turn_key = f"turn_{i}"
                if turn_key in data_dict:
                    turn_value = data_dict[turn_key]['user']
                    user_utt.append(turn_value)
                    if state == "f1":
                        y_true.append(-1)
                    elif state == "1":
                        y_true.append(1)
                    elif state == "0":
                        y_true.append(0)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue

    prompts = """
        正式程度指的是句子中表达的正式程度。判断句子是否正式的规则如下：

        正式的句子通常包含诸如"你好"、"您"、"抱歉"、"请"、"不好意思"等词语。同时，正式的句子也可能不包含这些词，但是结构可能较复杂，语法较为严谨，可能包含完整的从句和复杂的句式，这些也算是正式的句子。
        非正式的句子通常不包含这些词语，并且可能包含一些口语化词汇。同时，非正式句子语法结构可能较简单，有时会省略主语或动词。
        你只需要为每个句子打1分或0分。1分表示句子正式，0分表示句子非正式。
        注意，句子不可能绝对正式或非正式，你只需要判断句子更倾向于哪一方。 例如，“请问你可以给我推荐一个动作电影吗” 是一个正式的句子，而 “来点动作电影推荐” 是一个非正式的句子。

        你需要先判断每个句子的正式程度，并输出你的判断原因，每行一个句子。
        请注意，你全程只有最后一行能用一次方括号，其他地方不能使用。
        然后，在最后一行输出每个句子的正式程度，以列表形式，用方括号括起来，并用逗号分隔各个数字：

        比如，对于以下对话：
        1. 我最近特别想吃一些辣味十足的食物，能否请您帮我推荐一些既美味又具有辣味的菜肴，最好是那种让人回味无穷的，我真的很期待能够品尝到这样的美食，您能给我一些建议吗？
        2. 不喜欢墨西哥菜，推荐点别的
        3. 你好！我在寻找一些美味的火锅推荐。请问你有什么好的建议吗
        4. 我在春熙路附近，想找一些适合我和朋友的川菜，比如麻辣烫，天气阴天，想吃点热乎的。
        5. 能告诉我这些餐厅的具体位置和环境吗？

        输出:
        1. 这个句子使用了“能否请您帮我推荐一些”这样的正式表达，结构复杂且语法严谨，整体上显得非常正式。
        2. 这个句子不包含任何正式表达，语气较为随意，语法结构简单，因此显得不够正式。
        3. 这个句子开头使用了“你好”，后面使用了“请问你有什么好的建议吗”，整体语气较为正式。
        4. 这个句子虽然内容丰富，但是没有使用正式表达，语气较为随意，因此显得不够正式。
        5. 虽然这句话没有"请问"等正式表达，但是整体语气较为正式，因此可以认为是较为正式的句子。

        [1, 0, 1, 0, 1]


        现在，你需要判断以下句子的正式程度：
        {utterance}

        输出:
        """

    def group_and_number(user_utt, group_size=5):
        result = []
        for i in range(0, len(user_utt), group_size):
            group = user_utt[i:i + group_size]
            numbered_group = [f"{j+1}. {text}" for j, text in enumerate(group)]
            result.append("\n".join(numbered_group))
        return result
    grouped_texts = group_and_number(user_utt)

    for grouped_text in grouped_texts:
        prompt = prompts.format(utterance=grouped_text)
        # print(f"Prompt: {prompt}")
        response = call_llm(prompt)
        try:
            # print(response.strip().split('\n')[-1])
            res = eval(response.strip().split('\n')[-1])
            for i in range(len(res)):
                if res[i] == 0:
                    res[i] -= 1
            y_pred.extend(res)

        except Exception as e:
            print(f"Error processing response: {e}")
            continue
    print("=============================Formality==========================")
    print("================================================================")
    print("================================================================")
    print("================================================================")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"y_true length: {len(y_true)}")
    print(f"y_pred length: {len(y_pred)}")

    # 计算整体 Accuracy 和 Macro Recall
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))



def test_emoji(test_path):
    with open(test_path + "history_5.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip() for line in lines if line.strip()]
    y_pred = []
    y_true = []
    user_utt = []
    for data in datas:
        try:
            data_dict = eval(data)
            profile = eval(data_dict['user_profile'])
            state = profile['user_behavior']['emojis']
            for i in range(6):
                turn_key = f"turn_{i}"
                if turn_key in data_dict:
                    turn_value = data_dict[turn_key]['user']
                    user_utt.append(turn_value)
                    if state == "f1":
                        y_true.append(-1)
                    elif state == "1":
                        y_true.append(1)
                    elif state == "0":
                        y_true.append(0)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue
    emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # 表情符号（emoticons）
    "\U0001F300-\U0001F5FF"  # 符号 & 图标
    "\U0001F680-\U0001F6FF"  # 交通 & 地图符号
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Symbols and Pictographs
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols, etc.
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+", flags=re.UNICODE)
    for text in user_utt:
        if emoji_pattern.search(text):
            y_pred.append(1)
        else:
            y_pred.append(0)
    print("=============================Emojis==========================")
    print("================================================================")
    print("================================================================")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"y_true length: {len(y_true)}")
    print(f"y_pred length: {len(y_pred)}")
    # 计算整体 Accuracy 和 Macro Recall
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")

    print(classification_report(y_true, y_pred, digits=4))
            

def test_sentiment(test_path):
    with open(test_path + "history_5.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip() for line in lines if line.strip()]
    y_pred = []
    y_true = []
    user_utt = []
    for data in datas:
        try:
            data_dict = eval(data)
            profile = eval(data_dict['user_profile'])
            state = profile['user_behavior']['sentiment']
            for i in range(6):
                turn_key = f"turn_{i}"
                if turn_key in data_dict:
                    turn_value = data_dict[turn_key]['user']
                    user_utt.append(turn_value)
                    if state == "f1":
                        y_true.append(-1)
                    elif state == "1":
                        y_true.append(1)
                    elif state == "0":
                        y_true.append(0)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue

    prompts = """
        情感倾向指的是句子中表达的情绪偏向程度。判断句子情绪是否积极的规则如下：

        积极的句子通常包含诸如“太棒了”、“非常喜欢”、“真不错”、“好开心”等正面词语。同时，积极的句子也可能不包含这些词语，但整体语气乐观、热情、鼓舞人心，这些也算是积极的句子。
        中性的句子通常不包含强烈情绪色彩，语气平淡、客观，注重事实和理性表达。
        消极的句子通常包含如“失望”、“糟糕”、“不满意”、“讨厌”等负面词语，或者语气显得冷淡、不满或悲观。
        你只需要为每个句子打一个分数：
        1 分表示情绪积极，0 分表示情绪中性，-1 分表示情绪消极。
        注意，句子不可能绝对积极或消极，你只需要判断它更倾向于哪一方。

        你需要先判断每个句子中表达的情绪偏向，并输出你的判断原因，每行一个句子。
        请注意，你全程只有最后一行能用一次方括号，其他地方不能使用。
        然后，在最后一行输出每个句子的情绪分类，以列表形式，用方括号括起来，并用逗号分隔各个数字：

        比如，对于以下对话：
        1. 我今天心情很失望，刚升职却没有什么值得庆祝的感觉。能不能帮我找找附近有没有意面或披萨店？我真的不想再去那些无趣的地方了。
        2. 我在江汉路，真希望能找到一个值得庆祝的意面或披萨店，但这天气还在下小雨，感觉真是糟糕透了。希望你能推荐一些好地方。
        3. 我想了解一下这些湘菜餐厅的推荐菜品。请问每家餐厅的特色菜品具体是什么？
        4. 请问这些湘菜餐厅的环境如何？适合和同事聚餐吗？
        5. 你好！我想找附近的快餐便当店，今天心情很好，想吃点美味的！你能推荐一些吗？
        6. 太棒了！谢谢你的推荐！我对这些快餐便当店非常感兴趣，特别是麦当劳和真功夫的选择。你能告诉我这些店铺的具体位置吗？我想看看哪个离我最近！

        输出:
        1. 句子中出现了“失望”、“无趣”等负面词语，表达了对升职后心情的不满和对去过的地方的不喜欢，情绪明显偏消极。  
        2. 句子中使用了“真希望”、“糟糕透了”等表达，虽然有期待但整体语气因天气和环境而显得消极，情绪偏消极。  
        3. 句子表达了对推荐菜品的兴趣，语气礼貌且中性，没有明显的情绪色彩。  
        4. 句子询问餐厅环境是否适合聚餐，语气平和，注重事实，没有情绪倾向。 
        5. 句子表达了“今天心情很好，想吃点美味的”，语气积极且带有愉快的期待感，属于积极情绪。  
        6. 句子开头使用了“太棒了！”，表达了强烈的正面情绪，对推荐内容非常感兴趣，明显积极。

        [-1, -1, 0, 0, 1, 1]
        现在，你需要判断以下句子的情绪分类：
        {utterance}

        输出:
        """
    def group_and_number(user_utt, group_size=5):
        result = []
        for i in range(0, len(user_utt), group_size):
            group = user_utt[i:i + group_size]
            numbered_group = [f"{j+1}. {text}" for j, text in enumerate(group)]
            result.append("\n".join(numbered_group))
        return result
    grouped_texts = group_and_number(user_utt)
    for grouped_text in grouped_texts:
        prompt = prompts.format(utterance=grouped_text)
        # print(f"Prompt: {prompt}")
        response = call_llm(prompt)
        try:
            # print(f"Response: {response}")
            # print(response.strip().split('\n')[-1])
            res = eval(response.strip().split('\n')[-1])
            y_pred.extend(res)

        except Exception as e:
            print(f"Error processing response: {e}")
            continue


    
    print("=============================Sentiment==========================")
    print("================================================================")
    print("================================================================")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"y_true length: {len(y_true)}")
    print(f"y_pred length: {len(y_pred)}")
    # 计算整体 Accuracy 和 Macro Recall
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))

def test_carelessness(test_path):
    with open(test_path + "history_5.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip() for line in lines if line.strip()]
    y_pred = []
    y_true = []
    user_utt = []
    for data in datas:
        try:
            data_dict = eval(data)
            profile = eval(data_dict['user_profile'])
            state = profile['user_behavior']['carelessness']
            for i in range(6):
                turn_key = f"turn_{i}"
                if turn_key in data_dict:
                    turn_value = data_dict[turn_key]['user']
                    user_utt.append(turn_value)
                    if state == "f1":
                        y_true.append(-1)
                    elif state == "1":
                        y_true.append(1)
                    elif state == "0":
                        y_true.append(0)
        except Exception as e:
            print(f"Error processing data: {e}")
            continue
    
    prompts = """
        语句正确性指的是一个句子在拼写、语法、用词和逻辑等方面是否存在明显问题。判断句子是否有错误的规则如下：

        拼写错误包括：词语打错、错别字等；
        语法错误包括：主谓不一致、搭配不当、语序混乱等；
        用词不当包括：词义不符合语境、近义词误用、搭配错误等；
        逻辑错误包括：句意矛盾、表达不清、前后语义不连贯等。
        即便语句基本通顺，如果存在上述任一问题，仍视为“有错误”。

        你需要为每个句子做一个判断：
        0 表示语句完全正确，无任何拼写、语法、用词或逻辑问题；
        1 表示语句存在上述任意一种错误。

        你需要先判断每个句子中是否存在上述的错误，并输出你的判断原因，每行一个句子。
        请注意，你全程只有最后一行能用一次方括号，其他地方不能使用。
        然后，在最后一行输出每个句子的分类，以列表形式，用方括号括起来，并用逗号分隔各个数字：

        比如，对于以下对话：
        1.好的，谢谢你推荐的意大利风情餐厅，我觉得听起来不错，准备去试试！再见！
        2.嘿，有什么好吃的湘菜餐厅推荐吗？我现在在海港城，想找点好吃的！
        3.嘿，有没有附近的意面店呀？我想吃点意面的说。
        4.王付井附近有没有好吃的香菜？

        输出:
        1.句子结构完整，语义清晰，无明显错误。
        2.表达清楚，符合口语习惯，无明显错误。
        3.这句话末尾“我想吃点意面的说”语法混乱，“的说”使用不当，存在语法和用词问题。
        4.句子中“香菜”应为“湘菜”，且“王付井”可能是“王府井”的误写，存在拼写错误。


        现在，你需要判断以下句子的错误分类：
        {utterance}

        输出:
        """
    def group_and_number(user_utt, group_size=5):
        result = []
        for i in range(0, len(user_utt), group_size):
            group = user_utt[i:i + group_size]
            numbered_group = [f"{j+1}. {text}" for j, text in enumerate(group)]
            result.append("\n".join(numbered_group))
        return result
    grouped_texts = group_and_number(user_utt)
    for grouped_text in grouped_texts:
        prompt = prompts.format(utterance=grouped_text)
        # print(f"Prompt: {prompt}")
        response = call_llm(prompt)
        try:
            # print(f"Response: {response}")
            # print(response.strip().split('\n')[-1])
            res = eval(response.strip().split('\n')[-1])
            y_pred.extend(res)

        except Exception as e:
            print(f"Error processing response: {e}")
            continue


    
    print("=============================carelessness==========================")
    print("================================================================")
    print("================================================================")
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"y_true length: {len(y_true)}")
    print(f"y_pred length: {len(y_pred)}")
    # 计算整体 Accuracy 和 Macro Recall
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(classification_report(y_true, y_pred, digits=4))





if __name__ == "__main__":
    llm_model = "deepseek-reasoner"
    test_path = "/data/liyuanzi/HUAWEI/User_simulator/GUsim_V3/test/OurUser_gpt-4o-mini_" + llm_model + "_Prob_Sample_BaseCRSgpt-4o-mini"
    test_path1 = test_path + "_conciseness/"
    test_path2 = test_path + "_formality/"
    test_path3 = test_path + "_emojis/"
    test_path4 = test_path + "_sentiment/"
    test_path5 = test_path + "_carelessness/"
    print(test_length(test_path1))  # 调用函数并传入路径
    test_formality(test_path2)
    test_emoji(test_path3)
    test_sentiment(test_path4)
    test_carelessness(test_path5)