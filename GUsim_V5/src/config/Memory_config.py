import os
import sys

#################################### Overall ####################################

DEFAULT_GLOBAL_CONFIG = {
    'usable_gpu': '3'
}

DEFAULT_OPENAI_APIKEY = '[YOUR_API KEY]'
DEFAULT_OPENAI_APIBASE = '[YOUR_API_BASE]'

DEFAULT_LLAMA3_8B_INSTRUCT_PATH = '[YOUR_PATH]'
DEFAULT_E5_BASE_V2_PATH = '[YOUR_PATH]'


#################################### Storage ####################################

DEFAULT_LINEAR_STORAGE = {
    
}





#################################### Recall ####################################
# ----- Truncation -----
DEFAULT_LMTRUNCATION = {
    # For truncation, you can choose two policy (word/token):
    # If the 'mode' is 'word', you just need provide the 'number' of words.
    # If the 'mode' is 'token', you need provide both the 'number' of words and the 'path' of tokenizer.
    'method': 'LMTruncation',
    'mode': 'word',
    'number': 1024,
    'path': DEFAULT_LLAMA3_8B_INSTRUCT_PATH
}

DEFAULT_TRUNCATION = DEFAULT_LMTRUNCATION

# ----- Utilization -----
DEFAULT_CONCATE_CHATUTILIZATION = {
    'method': 'ChatConcateUtilization',
    'prefix': '',
    'suffix': '',
    'list_config': {
        'index': True,
        'sep': '\n'
    },
    'dict_config': {
        'key_format': '(%s)',
        'key_value_sep': '\n',
        'item_sep': '\n'
    }
}

DEFAULT_CONCATE_ACTIONUTILIZATION = {
    'method': 'ActionConcateUtilization',
    'prefix': '',
    'suffix': '',
    'list_config': {
        'index': True,
        'sep': '\n'
    },
    'dict_config': {
        'key_format': '(%s)',
        'key_value_sep': '\n',
        'item_sep': '\n'
    }
}

DEFAULT_CONCATE_PROFILEUTILIZATION = {
    'method': 'ChatConcateUtilization',
    'prefix': '',
    'suffix': '',
    'list_config': {
        'index': False,
        'sep': '\n'
    },
    'dict_config': {
        'key_format': '(%s)',
        'key_value_sep': '\n',
        'item_sep': '\n'
    }
}

# DEFAULT_UTILIZATION = DEFAULT_CONCATE_UTILIZATION



DEFAULT_ChATMEMORY_RECALL = {
    'method': 'ChatMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_CONCATE_CHATUTILIZATION,
    'empty_memory': 'None'
}


DEFAULT_ACTIONMEMORY_RECALL = {
    'method': 'ActionMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_CONCATE_ACTIONUTILIZATION,
    'empty_memory': 'None'
}

DEFAULT_PROFILEMEMORY_RECALL = {
    'method': 'ProfileMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_CONCATE_PROFILEUTILIZATION,
    'empty_memory': 'None'
}




#################################### Store ####################################

DEFAULT_FUMEMORY_STORE = {
    'method': 'FUMemoryStore'
}


#################################### DisPlay ####################################

DEFAULT_SCREEN_DISPLAY = {
    'method': 'ScreenDisplay',
    'prefix': '----- Current Memory Start (%s) -----',
    'suffix': '----- Current Memory End -----',
    'key_format': '(%s)',
    'key_value_sep': '\n',
    'item_sep': '\n'
}

DEFAULT_FILE_DISPLAY = {
    'method': 'FileDisplay',
    'prefix': '----- Current Memory Start (%s) -----',
    'suffix': '----- End -----',
    'key_format': '(%s)',
    'key_value_sep': '\n',
    'item_sep': '\n',
    'output_path': 'logs/sample.log'
}

DEFAULT_DISPLAY = DEFAULT_FILE_DISPLAY

DEFAULT_CHATMEMORY = {
    'name': 'ChatMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_ChATMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_ACTIONMEMORY = {
    'name': 'ActionMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_ACTIONMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}

DEFAULT_PROFILEMEMORY = {
    'name': 'ProfileMemory',
    'storage': DEFAULT_LINEAR_STORAGE,
    'recall': DEFAULT_PROFILEMEMORY_RECALL,
    'store': DEFAULT_FUMEMORY_STORE,
    'display': DEFAULT_DISPLAY,
    'global_config': DEFAULT_GLOBAL_CONFIG
}