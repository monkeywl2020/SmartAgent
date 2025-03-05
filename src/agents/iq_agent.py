# -*- coding: utf-8 -*-
""" Base class for Agent """
from typing import Any, Dict, Iterator, List, Literal, Optional, Union,AsyncIterator
from ..msg.message import  ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER,FUNCTION,TOOL, Message,ContentItem
from loguru import logger

from ..msg.message import  Message
from ..core.base_agent import BaseAgent,register_agent


# 智商模型代理 TaskActionAgent 继承自 BaseAgent
@register_agent('taskaction_agent')
class TaskActionAgent(BaseAgent):

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
        logger.info("BaseAgent-----------__init__:{}",name)
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
        logger.info("TaskActionAgent::a_generate_rsp begin!")
        # 先将  self.tools_list["abc","cdd","ddfe"] 对应的工具获取出来,里面是工具名字
        emotional_tools = list(self.function_map.values())
        logger.info(f"TaskActionAgent::a_generate_rsp ---- emotional_tools:{emotional_tools}!")

        # 将tool转成 模型需要的格式 
        tools = self.llm_client.convert_nuwa_tool_to_model_support(emotional_tools)
        logger.info(f"TaskActionAgent::a_generate_rsp ---- new format tools:{tools}!")

        # 添加system prompt
        # 历史记录 暂时为空
        #-----------------------------
        #  如果用openAI消息格式，qwen调用会不准确，最好的方法就是将历史记录放到system 中去
        #-----------------------------
        history_text = self.format_history_as_markdown(messages[:-1]) # 将其他消息格式化为文本历史消息，markdown格式。
        emotional_prompt = self.sys_prompt + history_text
        logger.info(f"TaskActionAgent::a_generate_rsp  ---- emotional_prompt:{emotional_prompt}!")
        sys_prompt_message = {"role":"system","content":emotional_prompt}
        #--------------------重新组装消息，openAI格式qwen支持不行-----------------------------
        new_messages = []
        # 将用户消息加入进去
        new_messages.append(messages[-1])
        # 就2个消息，system 和 user的
        new_messages.insert(0,sys_prompt_message)
        #-----------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        num_llm_calls_available = kwargs.pop('num_llm_calls', 2)
        logger.info(f"TaskActionAgent::a_generate_rsp  ---- num_llm_calls_available:{num_llm_calls_available}!")

        # 返回生成器对象
        return self.get_llm_reply(
            messages=new_messages,
            stream=stream,
            delta_stream=delta_stream,
            num_llm_calls_available=num_llm_calls_available,
            tools=tools,
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
        num_llm_calls_available: int,
        tools: List[Dict],
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        """异步生成器函数，统一处理流式和非流式响应"""
        logger.info("TaskActionAgent::get_llm_reply ---- begin!")
        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            logger.info(f"TaskActionAgent::get_llm_reply ---- num_llm_calls_available{num_llm_calls_available}!")
            responses = await self.llm_client.a_generate_rsp(
                messages=messages,
                stream=stream,
                delta_stream=delta_stream,
                tools=tools,
                **kwargs
            )

            if isinstance(responses, AsyncIterator):
                collected_responses = []
                # 遍历 迭代器中的每个消息
                #logger.info(f"get_llm_reply::get_llm_reply ---- a_generate_rsp --- AsyncIterator begin !")
                async for response in responses:
                    #logger.info(f"get_llm_reply::get_llm_reply ---ddd--- response:{response}!")
                    message = response[0]  # 假设每个 response 是 List[Message]
                    # 如果是用户消息，直接yield出去
                    if message.role == ASSISTANT and message.content:
                        yield [message]  # 实时返回用户消息
                    # 如果是 Function 消息，将所有的消息收完（模型客户端侧之后返回一个消息）
                    elif message.role == FUNCTION and message.tool_calls:
                        collected_responses.extend(response)

                # 这个是在迭代器遍历完成后。如果有 tool的话，这个不为空
                if collected_responses:
                    # 获取 tool 处理的结果。
                    # tool_call_msg：调用消息 
                    # tool_results： 调用的结果消息
                    has_tool_call, tool_call_msg, tool_results = await self._handle_tool_calls(collected_responses, **kwargs)
                    logger.info(f"TaskActionAgent::get_llm_reply ---fff--- 【has_tool_call:{has_tool_call}】 【tool_call_msg:{tool_call_msg}】 【tool_results:{tool_results}】 !")
                    if has_tool_call:
                        combined_results = [tool_call_msg] + tool_results # 组合两个消息
                        #####################################################################
                        #   combined_results 里面是 function call 的信息和
                        #       调用了tool的信息，返回的是tool的结果  
                        #       这两个消息，一个是 {role:assistant，content:function call} 
                        #                  一个是 {role:tool，content:function call result}
                        ####################################################################
                        logger.info(f"TaskActionAgent::get_llm_reply ---fff--- combined_results:{combined_results}")
                        yield combined_results  # 返回工具调用结果

                        # 增加了一个 tool 调用返回值是不是 切换agent的判断  switch_mode_to 标识是要切换agent，这个时候就不继续执行llm了
                        if (len(tool_results) == 1 
                            and tool_results[0].content == '' 
                            and isinstance(tool_results[0].extra, dict) 
                            and "switch_mode_to" in tool_results[0].extra):
                            logger.info(f"TaskActionAgent::get_llm_reply --- Switching to {tool_results[0].extra['switch_mode_to']}")
                            # 不在调用 llm
                            return # 结束生成器  不继续执行下面的内容了

                        # 如果还可以调用大模型
                        if num_llm_calls_available > 0:
                            messages.append(tool_call_msg.model_dump())   # 将 tool_call_msg 消息加入messages

                            #将调用的结果加入到message中去
                            for tool in tool_results:
                                messages.append(tool.model_dump())    # 将 tool_results 消息加入messages
                            continue
                        
                    
                return  # 结束生成器

            # 下面是非流式的消息，就是普通的消息不是异步迭代器
            elif isinstance(responses, list) and responses:
                if responses[0].role == ASSISTANT:
                    yield responses  # 返回非流式用户消息，转成 异步迭代器返回，外部统一处理

                elif responses[0].role == FUNCTION:
                    has_tool_call, tool_call_msg, tool_results = await self._handle_tool_calls(responses, **kwargs)
                    if has_tool_call:
                        combined_results = [tool_call_msg] + tool_results

                        if (len(tool_results) == 1 
                            and tool_results[0].content == '' 
                            and isinstance(tool_results[0].extra, dict) 
                            and "switch_mode_to" in tool_results[0].extra):
                            logger.info(f"BaseAgent::get_llm_reply --- Switching to {tool_results[0].extra['switch_mode_to']}")
                            # 不在调用 llm
                            yield combined_results  # 直接返回切换结果
                            return

                        if num_llm_calls_available > 0:
                            messages.append(tool_call_msg.model_dump())
                            #将调用的结果加入到message中去
                            for tool in tool_results:
                                messages.append(tool.model_dump())    # 将 tool_results 消息加入messages
                            continue
                        yield combined_results  # 返回工具调用结果，异步迭代器返回

                return

            # 如果没有响应，默认返回空列表
            yield responses if responses else [] # 异步迭代器返回
            return
        

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