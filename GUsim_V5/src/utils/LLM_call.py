from openai import OpenAI
import logging
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from retrying import retry

LOGGER = logging.getLogger("UserSimulator")
# GLM_DIR = os.path.join(os.path.dirname(__file__), "../../models/glm-4-9b-chat")

class LLM_call:
    def __init__(self, llm_model=None):
        self.llm_model = llm_model

    def call_llm(self, llm_input):
        raise NotImplementedError

    def __call__(self, llm_input):
        return self.call_llm(llm_input)


class OPENAI_call(LLM_call):
    def __init__(self, llm_model=None):
        super(OPENAI_call, self).__init__(llm_model)
        self.model = llm_model
        self.temperature = 0
        self.max_tokens = 512
        self.seed = 39
        self.client = OpenAI()


    # @retry(stop_max_attempt_number=10)
    def call_llm(self, llm_input):
        # try:
        # print("self.model:", self.model)
        output = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": llm_input}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed
        )
        # print(output)

        return output.choices[0].message.content
        # except Exception as e:
        #     LOGGER.warning(f"Failed to call LLM model: {e}")
        #     return None

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_seed(self, seed: int):
        self.seed = seed

# class GLM_call(LLM_call):
#     def __init__(self, llm_model=None):
#         super(GLM_call, self).__init__(llm_model)
#         self.llm_model = llm_model
#         self.max_tokens = 2500
#         self.top_k = 1
#         self.do_sample = True
#
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(GLM_DIR, trust_remote_code=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#                 GLM_DIR,
#                 torch_dtype=torch.bfloat16,
#                 low_cpu_mem_usage=True,
#                 trust_remote_code=True
#             ).to(self.device).eval()
#
#     def call_llm(self, llm_input):
#         try:
#             inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": llm_input}],
#                                                     add_generation_prompt=True,
#                                                     tokenize=True,
#                                                     return_tensors="pt",
#                                                     return_dict=True
#                                                     ).to(self.device)
#             gen_kwargs = {"max_length": self.max_tokens, "do_sample": self.do_sample, "top_k": self.top_k}
#             with torch.no_grad():
#                 outputs = self.model.generate(**inputs, **gen_kwargs)
#                 outputs = outputs[:, inputs['input_ids'].shape[1]:]
#                 return_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 # 删除段前回车,有可能有多个
#                 return_str = return_str.strip("\n")
#                 return return_str
#
#         except Exception as e:
#             LOGGER.warning(f"Failed to call LLM model: {e}")
#             return None
#
#     def set_max_tokens(self, max_tokens):
#         self.max_tokens = max_tokens
#
#     def set_seed(self, seed: int):
#         self.seed = seed
#
#     def set_top_k(self, top_k: int):
#         self.top_k = top_k
#
#     def set_do_sample(self, do_sample: bool):
#         self.do_sample = do_sample

class Qwen_call(LLM_call):
    def __init__(self, llm_model=None):
        super(Qwen_call, self).__init__(llm_model)
        self.model = llm_model
        self.temperature = 0
        self.max_tokens = 512
        self.seed = 39
        self.enable_thinking = False
        self.client = OpenAI(base_url="http://localhost:8093/v1", api_key="qwen_fxy")  # qwen-fxy 随便填写，只是为了通过接口参数校验


    # @retry(stop_max_attempt_number=10)
    def call_llm(self, llm_input):
        # try:
        # print("self.model:", self.model)
        output = self.client.chat.completions.create(
            model=self.model,
            extra_body={
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": llm_input}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed
        )
        # print(output)

        return output.choices[0].message.content
        # except Exception as e:
        #     LOGGER.warning(f"Failed to call LLM model: {e}")
        #     return None

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_seed(self, seed: int):
        self.seed = seed
    
    def set_thinking(self, enable_thinking: bool):
        self.enable_thinking = enable_thinking



if __name__ == "__main__":
    import os
    os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech"

    os.environ["OPENAI_API_KEY"] = "sk-G6XflAR04SvxaTl1QhmIl6HimueZ0ZPYFeEVD78gELMRqth5"
    # openai.api_base = "https://api.chatanywhere.tech"
    llm_call = OPENAI_call("deepseek-r1")
    print(llm_call("Hello, how are you?"))

