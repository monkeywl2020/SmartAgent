# -*- coding: utf-8 -*-
""" Base class for Agent """
from typing import Any, Dict, Iterator, List, Literal, Optional, Union,AsyncIterator
from ..msg.message import  ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER,FUNCTION,TOOL, Message,ContentItem
from loguru import logger

from ..msg.message import  Message
from ..core.base_agent import BaseAgent,register_agent

'''
智商模型代理，专门负任务工具调用等问答，主要是指令型模型调用 tool call 为主
'''

# 智商模型代理 TaskActionAgent 继承自 BaseAgent
@register_agent('qwen_taskaction_agent')
class QwenTaskActionAgent(BaseAgent):

    #初始化
    def __init__(
        self,
        name: str,
        sys_prompt: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        tools_list: Optional[List[Union[str, Dict]]] = None,
        description: str = None,
        **kwargs: Any,
    ):
        logger.info("QwenTaskActionAgent-----------__init__:{}",name)
        super().__init__(name=name, sys_prompt=sys_prompt, llm_config=llm_config, description=description, tools_list = tools_list, **kwargs)
        # 添加其他的初始化

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    async def a_generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any
        ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info("QwenTaskActionAgent::a_generate_rsp begin!")

        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        num_llm_calls_available = kwargs.pop('num_llm_calls', 2)
        logger.info(f"QwenTaskActionAgent::a_generate_rsp  ---- num_llm_calls_available:{num_llm_calls_available}!")

        # 添加system prompt
                
        #----------------------------------------------------------
        #   qwen-agent的 系统提示词是在初始化的时候设置的。
        #   历史聊天记录也不在这里处理，messages是含历史记录的
        #       历史记忆（pass） 
        #----------------------------------------------------------
        
        # 返回生成器对象
        return self.get_llm_reply(
            messages=messages,
            stream=stream,
            delta_stream=delta_stream,
            num_llm_calls=num_llm_calls_available,
            **kwargs
        )

    #-----------------------------------------------------------------------------------------
    #                 ---** get_llm_reply 定制处理逻辑 **---
    # 这个是访问大模型，获取模型的响应接口。
    # 由于这个函数是 eq_agent 和 iq_agent定制的，所以需要重写 base_agent 的逻辑
    #   **为什么是定制的： 因为 这两个agent有个 切换agent的逻辑在里面，通过tool call来切换agent**
    #   所以 这个tool call调用不是通用逻辑，需要定制
    #-----------------------------------------------------------------------------------------
    async def get_llm_reply(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        """异步生成器函数，统一处理流式和非流式响应"""
        logger.info("get_llm_reply::get_llm_reply ---- begin!")
        
        # 调用 qwen-agent 与模型交互，历史记录啊，还有知识库啊，什么的在这里都不管了
    # 使用 await 获取 llm_client 的结果
        responses = await self.llm_client.a_generate_rsp(
            messages=messages,
            stream=stream,
            delta_stream=delta_stream,
            **kwargs
        )
        async for msg in responses:
            yield msg
       

    # 将历史记录转换成 Markdown 格式
    def format_history_as_markdown(self, history: List[Dict[str, str]]) -> str:
        # 接着这段内容下面的，所以加2个回车换行 ## **历史记录**：{history}
        markdown = "\n\n## **历史记录**：\n\n"
        for message in history:
            role = message['role']
            content = message['content']
            markdown += f"**{role}:** {content}\n\n"
        return markdown


    #------------------------------------------------
    # 同步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------        
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:

        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        # 先将  self.tools_list["abc","cdd","ddfe"] 对应的工具获取出来,里面是工具名字
        pass