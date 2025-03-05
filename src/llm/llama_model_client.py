import json
import asyncio
from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence

from .base import BaseChatModelClient, register_llm

#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
@register_llm('llama')
class LlamaChatModelClient(BaseChatModelClient):

    # llm 客户端适配
    def __init__(self, 
                 config_list: Optional[List[Dict[str, Any]]] = None, 
                 **kwargs: Any):
        pass
