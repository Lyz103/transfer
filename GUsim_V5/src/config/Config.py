from .LLM_config import LLM_Model_Chat, LLM_Model_Action

class Simulator_Config:
    """
    模拟器配置类，整合聊天模型、动作模型、采样方式等参数。
    """

    # ------- 模型配置 -------
    llm_chat = LLM_Model_Chat           # 用于回复的 LLM 模型
    llm_action = LLM_Model_Action       # 用于动作生成的 LLM 模型
    Domain_SYS = "美食推荐系统"
    name = f"OurUser_{llm_action.llm_model}_{llm_chat.llm_model}"  # 用户名带模型信息

    # ------- 采样方式 -------
    Sample_Method = "Prob_Sample"  # 可选: 'Uniform_Sample', 'Prob_Sample', 'TTS'
    Profile_Path = "/data/liyuanzi/HUAWEI/GUsim_V3/src/synthesized_profiles.jsonl"



    @classmethod
    def describe(cls):
        """打印当前配置概览"""
        print(f"[配置总览]")
        print(f"用户名称：{cls.name}")
        print(f"聊天模型：{cls.llm_chat.llm_model}")
        print(f"动作模型：{cls.llm_action.llm_model}")
        print(f"采样方式：{cls.Sample_Method}")
        if cls.Sample_Method == "TTS":
            print(f"TTS 投票次数：{cls.Sample_Num}")
    

