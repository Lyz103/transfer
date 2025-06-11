import os
import sys
import copy
import json
sys.path[0] = os.path.join(os.path.dirname(__file__), '..')
from utils import OPENAI_call

from config.Config import Simulator_Config
from user_simulator import UserSimulator
from models.CRS import GptBasedCRS, AgentCRS
from collections import deque
cfg = Simulator_Config()


def BFS_BAS_VS_BASE(profile, filename):
    utterances_total = []
    utterances = []
    actions = []
    crs1 = GptBasedCRS(model="gpt-4o-mini")
    crs2 = GptBasedCRS(model="gpt-3.5-turbo")

    usersimulator = UserSimulator(profile, config=cfg)


    # 初始化队列 Q，包含初始问题
    Q = deque()
    user_chat, if_end, action = usersimulator.interact(1)
    utterances.append({"user": user_chat})
    actions.append(action)

    print("user_chat:", user_chat, "if_end:", if_end, "action:", action)
    Q.append((user_chat, if_end, copy.deepcopy(actions), copy.deepcopy(utterances), 1))

    while Q:
        user_chat, if_end, actions, utterances, t = Q.popleft()
        print("user_chat:", user_chat, "if_end:", if_end, "actions:", actions, "utterances:", utterances, "t:", t)

        utterances1 = copy.deepcopy(utterances)
        utterances2 = copy.deepcopy(utterances)

        actions1 = copy.deepcopy(actions)
        actions2 = copy.deepcopy(actions)



        if if_end:
            continue
        if t > 5:
            continue
        crs1 = GptBasedCRS(model="gpt-4o-mini")
        crs2 = GptBasedCRS(model="gpt-3.5-turbo")


        #### crs1 ---------------------------------------------------------------------------------------------------------------------------------------------------------
        history_temp = ""
        if len(utterances1) > 2:
            for i in range(len(utterances1) - 2):
                if i % 2 == 0:
                    history_temp += "User: " + utterances1[i]['user'] + "\n"
                else:
                    history_temp += "CRS: " + utterances1[i]['crs'] + "\n"
        crs1.interaction_history = history_temp
        crs_response1 = crs1.interact(user_chat)
        utterances1.append({"crs": crs_response1})
        # utterances_total.append(copy.deepcopy(utterances1))

        usersimulator_1 = UserSimulator(profile, config=cfg)
        for i in range(len(utterances1)):
            if i % 2 == 0:
                usersimulator_1.UserAct._update_memory_User(actions1[i // 2], utterances1[i]['user'])
            else:
                usersimulator_1.UserAct._update_memory_Sys(utterances1[i]['crs'])
        
        print("CRS1:","Memory:", usersimulator_1.UserAct.user_memory.recall())
        user_chat_1, if_end_1, action_1 = usersimulator_1.interact(t + 1)
        utterances1.append({"user": user_chat_1})
        actions1.append(action_1)

        print("CRS1:","crs_response1:", crs_response1,"user_chat:", user_chat_1, "if_end:", if_end_1, "action:", action_1,  "\n\n\n")
        if if_end_1 is not True or t <= 5:

            Q.append((user_chat_1, if_end_1, copy.deepcopy(actions1), copy.deepcopy(utterances1), t + 1))
        ### crs2-----------------------------------------------------------------------------------------------------------------------------------------------------
        history_temp = ""
        if len(utterances2) > 2:
            print("len(utterances2):", len(utterances2))
            for i in range(len(utterances2) - 2):
                if i % 2 == 0:
                    history_temp += "User: " + utterances2[i]['user'] + "\n"
                else:
                    history_temp += "CRS: " + utterances2[i]['crs'] + "\n"
        crs2.interaction_history = history_temp
        crs_response2 = crs2.interact(user_chat)
        utterances2.append({"crs": crs_response2})
        # utterances_total.append(copy.deepcopy(utterances2))
        usersimulator_2 = UserSimulator(profile, config=cfg)

        for i in range(len(utterances2)):
            if i % 2 == 0:
                usersimulator_2.UserAct._update_memory_User(actions2[i // 2], utterances2[i]['user'])
            else:
                usersimulator_2.UserAct._update_memory_Sys(utterances2[i]['crs'])
        
        print("CRS2:","Memory:", usersimulator_2.UserAct.user_memory.recall())
        user_chat_2, if_end_2, action_2 = usersimulator_2.interact(t + 1)

        utterances2.append({"user": user_chat_2})
        actions2.append(action_2)

        utterances_total.append({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2})
        print("CRS2:","crs_response2:", crs_response2,"user_chat:", user_chat_2, "if_end:", if_end_2, "action:", action_2,  "\n\n\n")
        with open(filename, "a", encoding="utf-8") as f:
            json.dump({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2}, f, ensure_ascii=False)
            f.write("\n")


        # if if_end_2 is not True or t <= 5:
        #     Q.append((user_chat_2, if_end_2, copy.deepcopy(actions2), copy.deepcopy(utterances2), t + 1))
    return utterances_total



def BFS_BAS_VS_AGENT(profile, filename):
    utterances_total = []
    utterances = []
    actions = []
    crs1 = GptBasedCRS(model="gpt-4o-mini")
    crs2 = AgentCRS(model="gpt-4o-mini")

    usersimulator = UserSimulator(profile, config=cfg)


    # 初始化队列 Q，包含初始问题
    Q = deque()
    user_chat, if_end, action = usersimulator.interact(1)
    utterances.append({"user": user_chat})
    actions.append(action)

    print("user_chat:", user_chat, "if_end:", if_end, "action:", action)
    Q.append((user_chat, if_end, copy.deepcopy(actions), copy.deepcopy(utterances), 1))

    while Q:
        user_chat, if_end, actions, utterances, t = Q.popleft()
        print("user_chat:", user_chat, "if_end:", if_end, "actions:", actions, "utterances:", utterances, "t:", t)

        utterances1 = copy.deepcopy(utterances)
        utterances2 = copy.deepcopy(utterances)

        actions1 = copy.deepcopy(actions)
        actions2 = copy.deepcopy(actions)



        if if_end:
            continue
        if t > 5:
            continue
        crs1 = GptBasedCRS(model="gpt-4o-mini")
        crs2 = AgentCRS(model="gpt-4o-mini")


        #### crs1 ---------------------------------------------------------------------------------------------------------------------------------------------------------
        history_temp = ""
        if len(utterances1) > 2:
            for i in range(len(utterances1) - 2):
                if i % 2 == 0:
                    history_temp += "User: " + utterances1[i]['user'] + "\n"
                else:
                    history_temp += "CRS: " + utterances1[i]['crs'] + "\n"
        crs1.interaction_history = history_temp
        crs_response1 = crs1.interact(user_chat)
        utterances1.append({"crs": crs_response1})
        # utterances_total.append(copy.deepcopy(utterances1))

        usersimulator_1 = UserSimulator(profile, config=cfg)
        for i in range(len(utterances1)):
            if i % 2 == 0:
                usersimulator_1.UserAct._update_memory_User(actions1[i // 2], utterances1[i]['user'])
            else:
                usersimulator_1.UserAct._update_memory_Sys(utterances1[i]['crs'])
        
        print("CRS1:","Memory:", usersimulator_1.UserAct.user_memory.recall())
        user_chat_1, if_end_1, action_1 = usersimulator_1.interact(t + 1)
        utterances1.append({"user": user_chat_1})
        actions1.append(action_1)

        print("CRS1:","crs_response1:", crs_response1,"user_chat:", user_chat_1, "if_end:", if_end_1, "action:", action_1,  "\n\n\n")
        if if_end_1 is not True or t <= 5:

            Q.append((user_chat_1, if_end_1, copy.deepcopy(actions1), copy.deepcopy(utterances1), t + 1))
        ### crs2-----------------------------------------------------------------------------------------------------------------------------------------------------
        if len(utterances2) > 2:
            print("len(utterances2):", len(utterances2))
            for i in range(len(utterances2) - 2):
                if i % 2 == 0:
                    crs2.crs_agent.oai_messages.append({"role": "user", "content": utterances2[i]['user']})
                else:
                    crs2.crs_agent.oai_messages.append({"role": "assistant", "content": utterances2[i]['crs']})

        crs_response2 = crs2.interact(user_chat)
        utterances2.append({"crs": crs_response2})
        # utterances_total.append(copy.deepcopy(utterances2))
        usersimulator_2 = UserSimulator(profile, config=cfg)

        for i in range(len(utterances2)):
            if i % 2 == 0:
                usersimulator_2.UserAct._update_memory_User(actions2[i // 2], utterances2[i]['user'])
            else:
                usersimulator_2.UserAct._update_memory_Sys(utterances2[i]['crs'])
        
        print("CRS2:","Memory:", usersimulator_2.UserAct.user_memory.recall())
        user_chat_2, if_end_2, action_2 = usersimulator_2.interact(t + 1)

        utterances2.append({"user": user_chat_2})
        actions2.append(action_2)

        utterances_total.append({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2})
        print("CRS2:","crs_response2:", crs_response2,"user_chat:", user_chat_2, "if_end:", if_end_2, "action:", action_2,  "\n\n\n")
        with open(filename, "a", encoding="utf-8") as f:
            json.dump({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2}, f, ensure_ascii=False)
            f.write("\n")


        # if if_end_2 is not True or t <= 5:
        #     Q.append((user_chat_2, if_end_2, copy.deepcopy(actions2), copy.deepcopy(utterances2), t + 1))
    return utterances_total
            
def BFS_AGENT_VS_AGENT(profile, filename):
    utterances_total = []
    utterances = []
    actions = []
    crs1 = AgentCRS(model="gpt-4o-mini")
    crs2 = AgentCRS(model="gpt-3.5-turbo")


    usersimulator = UserSimulator(profile, config=cfg)


    # 初始化队列 Q，包含初始问题
    Q = deque()
    user_chat, if_end, action = usersimulator.interact(1)
    utterances.append({"user": user_chat})
    actions.append(action)

    print("user_chat:", user_chat, "if_end:", if_end, "action:", action)
    Q.append((user_chat, if_end, copy.deepcopy(actions), copy.deepcopy(utterances), 1))

    while Q:
        user_chat, if_end, actions, utterances, t = Q.popleft()
        print("user_chat:", user_chat, "if_end:", if_end, "actions:", actions, "utterances:", utterances, "t:", t)

        utterances1 = copy.deepcopy(utterances)
        utterances2 = copy.deepcopy(utterances)

        actions1 = copy.deepcopy(actions)
        actions2 = copy.deepcopy(actions)



        if if_end:
            continue
        if t > 5:
            continue
        crs1 = AgentCRS(model="gpt-4o-mini")
        crs2 = AgentCRS(model="gpt-3.5-turbo")

        #### crs1 ---------------------------------------------------------------------------------------------------------------------------------------------------------
        if len(utterances1) > 2:
            for i in range(len(utterances1) - 2):
                if i % 2 == 0:
                    crs1.crs_agent.oai_messages.append({"role": "user", "content": utterances2[i]['user']})
                else:
                    crs1.crs_agent.oai_messages.append({"role": "assistant", "content": utterances2[i]['crs']})
        crs_response1 = crs1.interact(user_chat)
        utterances1.append({"crs": crs_response1})
        # utterances_total.append(copy.deepcopy(utterances1))

        usersimulator_1 = UserSimulator(profile, config=cfg)
        for i in range(len(utterances1)):
            if i % 2 == 0:
                usersimulator_1.UserAct._update_memory_User(actions1[i // 2], utterances1[i]['user'])
            else:
                usersimulator_1.UserAct._update_memory_Sys(utterances1[i]['crs'])
        
        print("CRS1:","Memory:", usersimulator_1.UserAct.user_memory.recall())
        user_chat_1, if_end_1, action_1 = usersimulator_1.interact(t + 1)
        utterances1.append({"user": user_chat_1})
        actions1.append(action_1)

        print("CRS1:","crs_response1:", crs_response1,"user_chat:", user_chat_1, "if_end:", if_end_1, "action:", action_1,  "\n\n\n")
        if if_end_1 is not True or t <= 5:

            Q.append((user_chat_1, if_end_1, copy.deepcopy(actions1), copy.deepcopy(utterances1), t + 1))
        ### crs2-----------------------------------------------------------------------------------------------------------------------------------------------------
        if len(utterances2) > 2:
            print("len(utterances2):", len(utterances2))
            for i in range(len(utterances2) - 2):
                if i % 2 == 0:
                    crs2.crs_agent.oai_messages.append({"role": "user", "content": utterances2[i]['user']})
                else:
                    crs2.crs_agent.oai_messages.append({"role": "assistant", "content": utterances2[i]['crs']})
        crs_response2 = crs2.interact(user_chat)
        utterances2.append({"crs": crs_response2})
        # utterances_total.append(copy.deepcopy(utterances2))
        usersimulator_2 = UserSimulator(profile, config=cfg)

        for i in range(len(utterances2)):
            if i % 2 == 0:
                usersimulator_2.UserAct._update_memory_User(actions2[i // 2], utterances2[i]['user'])
            else:
                usersimulator_2.UserAct._update_memory_Sys(utterances2[i]['crs'])
        
        print("CRS2:","Memory:", usersimulator_2.UserAct.user_memory.recall())
        user_chat_2, if_end_2, action_2 = usersimulator_2.interact(t + 1)

        utterances2.append({"user": user_chat_2})
        actions2.append(action_2)

        utterances_total.append({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2})
        print("CRS2:","crs_response2:", crs_response2,"user_chat:", user_chat_2, "if_end:", if_end_2, "action:", action_2,  "\n\n\n")
        with open(filename, "a", encoding="utf-8") as f:
            json.dump({"context": copy.deepcopy(utterances), "crs1": crs_response1, "crs2": crs_response2}, f, ensure_ascii=False)
            f.write("\n")


        # if if_end_2 is not True or t <= 5:
        #     Q.append((user_chat_2, if_end_2, copy.deepcopy(actions2), copy.deepcopy(utterances2), t + 1))
    return utterances_total



if __name__ == "__main__":
    profiles = []
    with open("synthesized_profiles.jsonl", 'r') as f:
        for line in f:
            profiles.append(str(line))

    # for i in range(len(profiles[:40])):
    #     try:
    #         directory = f"eval_data/B4ominiVSB3.5_U4o-mini"
    #         os.makedirs(directory, exist_ok=True)  # exist_ok=True 表示目录存在也不报错

    #         filename = f"{directory}/output{i}.jsonl"
    #         a = BFS_BAS_VS_BASE(profiles[i], filename)

    #         # with open(filename, "w", encoding="utf-8") as f:
    #         #     json.dump(a, f, ensure_ascii=False, indent=4)
    #     except Exception as e:
    #         print(f"Error processing profile {i}: {e}")
    #         continue
    # for i in range(len(profiles[:40])):
    #     try:
    #         directory = f"eval_data/A4ominiVSA3.5_U4o-mini"
    #         os.makedirs(directory, exist_ok=True)  # exist_ok=True 表示目录存在也不报错

    #         filename = f"{directory}/output{i}.jsonl"
    #         a = BFS_AGENT_VS_AGENT(profiles[i], filename)

    #         # with open(filename, "w", encoding="utf-8") as f:
    #         #     json.dump(a, f, ensure_ascii=False, indent=4)
    #     except Exception as e:
    #         print(f"Error processing profile {i}: {e}")
    #         continue
    for i in range(23, 40):
        try:
            directory = f"eval_data/A4ominiVSB4o-mini_U4o-mini"
            os.makedirs(directory, exist_ok=True)  # exist_ok=True 表示目录存在也不报错

            filename = f"{directory}/output{i}.jsonl"
            print("file_name:", filename)
            a = BFS_BAS_VS_AGENT(profiles[i], filename)

            # with open(filename, "w", encoding="utf-8") as f:
            #     json.dump(a, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing profile {i}: {e}")
            continue