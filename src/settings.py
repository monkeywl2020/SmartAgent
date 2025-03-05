import ast
import os
from typing import List, Literal
from dotenv import load_dotenv

# 后续涉及到的一些通用经常修改的配置可以放到环境变量中去，这样就不用改代码运行直接修改环境变量

# 下面是放置环境变量的 .env文件
dotenv_path = "/home/ubuntu/wlwork/.env"
load_dotenv(dotenv_path=dotenv_path)


# Settings for LLMs
'''
DEFAULT_MAX_INPUT_TOKENS: int = int(os.getenv(
    'QWEN_AGENT_DEFAULT_MAX_INPUT_TOKENS', 30000))  # The LLM will truncate the input messages if they exceed this limit
'''

# Settings for agents
#MAX_LLM_CALL_PER_RUN: int = int(os.getenv('QWEN_AGENT_MAX_LLM_CALL_PER_RUN', 8))

# Settings for tools
# 这个是nuwa-agent的环境的工作目录，从 .env 中读取 
DEFAULT_WORKSPACE: str = os.getenv('NUWA-AGENT_DEFAULT_WORKSPACE', 'workspace')

# Settings for RAG
'''
DEFAULT_MAX_REF_TOKEN: int = int(os.getenv('QWEN_AGENT_DEFAULT_MAX_REF_TOKEN',
                                           20000))  # The window size reserved for RAG materials
DEFAULT_PARSER_PAGE_SIZE: int = int(os.getenv('QWEN_AGENT_DEFAULT_PARSER_PAGE_SIZE',
                                              500))  # Max tokens per chunk when doing RAG
DEFAULT_RAG_KEYGEN_STRATEGY: Literal['None', 'GenKeyword', 'SplitQueryThenGenKeyword', 'GenKeywordWithKnowledge',
                                     'SplitQueryThenGenKeywordWithKnowledge'] = os.getenv(
                                         'QWEN_AGENT_DEFAULT_RAG_KEYGEN_STRATEGY', 'GenKeyword')
DEFAULT_RAG_SEARCHERS: List[str] = ast.literal_eval(
    os.getenv('QWEN_AGENT_DEFAULT_RAG_SEARCHERS',
              "['keyword_search', 'front_page_search']"))  # Sub-searchers for hybrid retrieval

'''