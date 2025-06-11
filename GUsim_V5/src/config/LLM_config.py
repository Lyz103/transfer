import os
import sys
from utils.LLM_call import OPENAI_call, Qwen_call
# from agents.LLM_call import GLM_call

# OPENAI API CALL
LLM_config_chat = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 4096,
    "seed": 39,
}

LLM_Model_Chat = OPENAI_call(LLM_config_chat["model"])
# LLM_Model_Chat.set_temperature(LLM_config_chat["temperature"])
# LLM_Model_Chat.set_max_tokens(LLM_config_chat["max_tokens"])
# LLM_Model_Chat.set_seed(LLM_config_chat["seed"])

# LLM_Model_Chat = Qwen_call(LLM_config_chat["model"])
LLM_Model_Chat.set_temperature(LLM_config_chat["temperature"])
LLM_Model_Chat.set_max_tokens(LLM_config_chat["max_tokens"])
LLM_Model_Chat.set_seed(LLM_config_chat["seed"])


LLM_config_action = {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 9000,
    "seed": 39,
}

LLM_Model_Action = OPENAI_call(LLM_config_action["model"])
LLM_Model_Action.set_temperature(LLM_config_action["temperature"])
LLM_Model_Action.set_max_tokens(LLM_config_action["max_tokens"])
LLM_Model_Action.set_seed(LLM_config_action["seed"])


# # GLM MODEL CALL
# LLM_config = {
#     "model": "glm-4-9b-chat",
#     "max_tokens": 2500,
#     "top_k": 1,
#     "do_sample": True
# }
# LLM_Model = GLM_call(LLM_config["model"])
# LLM_Model.set_max_tokens(LLM_config["max_tokens"])
# LLM_Model.set_top_k(LLM_config["top_k"])
# LLM_Model.set_do_sample(LLM_config["do_sample"])