# -*- coding: utf-8 -*-
import openai
import json
import time
from loguru import logger

from anthropic import AsyncAnthropic

from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence,Iterator,AsyncIterator
from .base import BaseChatModelClient, register_llm
from ..tools.base import NuwaBaseTool
from ..msg.message import  Message,ASSISTANT,FUNCTION,TOOL

#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
@register_llm('claude')
class ClaudeChatModelClient(BaseChatModelClient):
    # llm 客户端适配
    def __init__(self, 
                 config_list: Optional[Dict[str, Any]] = None, 
                 **kwargs: Any):
        logger.info("ClaudeChatModelClient-----------__init__")
        if config_list:
            # 如果有内容，直接展开作为参数传入 BaseChatModelClient 作为关键字保存
            super().__init__(config_list=config_list, **config_list, **kwargs)
        else:
            super().__init__(config_list=config_list, **kwargs)

        logger.info(f"ClaudeChatModelClient-----------__init__  api_key:{self.api_key}")
        # 初始化异步客户端
        self.client = AsyncAnthropic(
            api_key= self.api_key
        )
        # 设置 max_tokens，默认值 4096
        self.max_tokens = kwargs.get('max_tokens', 4096)

    # 根据用户输入的messages 和 Functions 进行处理了
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            tools: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """生成响应，支持流式和非流式输出
        
        Args:
            messages: 输入消息列表
            functions: 可选的功能列表（tools）
            stream: 是否使用流式输出
            delta_stream: 流式输出时是否仅返回增量内容
            **kwargs: 其他参数
        
        Returns:
            Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]: 响应内容
        """
        # 整理 OpenAI 参数
        openAIkwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "max_tokens": self.max_tokens,
        }

        # 其他参数
        if stream:
            openAIkwargs["stream_options"] = {"include_usage": True}

        # 如果有 Function 
        if tools:
            openAIkwargs["tools"] = tools

        # 其他参数从 kwargs 中更新
        openAIkwargs.update(kwargs)

        #-------------------------------------------------------------
        #   调用 OpenAI 接口
        #-------------------------------------------------------------
        logger.info(f"ClaudeChatModelClient::genrate_rsp:-------cccc---- openAIkwargs:\n{openAIkwargs}")
        response = self.client.chat.completions.create(**openAIkwargs)
        logger.info(f"ClaudeChatModelClient::genrate_rsp:-------cccc---- response:\n{response}")

        if stream:
            # 返回流式 迭代器
            abc = self.parse_stream_response(response)
            logger.info(f"ClaudeChatModelClient::genrate_rsp:-------parse_stream_response -- response:\n{response}")
            return abc
        else:
            # 返回 Message 消息
            return self.parse_non_stream_response(response)
        
    # 异步生成响应
    async def a_generate_rsp(
        self,
        messages: List[Union[Message, Dict]], # 实际处理都是dict，不是message
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any,
    ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info("ClaudeChatModelClient::a_generate_rsp --- begin!")
        """生成响应，支持流式和非流式输出（异步版本）"""
        claudekwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "top_p":self.top_p,
            "temperature":self.temperature
        }
            
        if tools:
            claudekwargs["tools"] = tools
        #------------------------------------------------------
        # 去掉 kwargs中的 user_id = user_id, user_age = user_age 这两个参数
        #------------------------------------------------------
        kwargs.pop('user_id')    # 这两个参数在openAI格式下没用，需要去掉
        kwargs.pop('user_age')   # 这两个参数在openAI格式下没用，需要去掉
        #------------------------------------------------------

        claudekwargs.update(kwargs) # 将kwargs的键和值更新到 openAIkwargs 中去，如果有相同的键直接kwargs 覆盖掉，没有的就添加

        logger.info(f"ClaudeChatModelClient::a_generate_rsp: openAIkwargs:\n{claudekwargs}")
        # 记录调用开始时间
        start_time = time.perf_counter()
        if stream:
            # 当 stream=True 时，使用异步生成器以支持流式处理
            response = self.client.messages.stream(**claudekwargs)
        else:
            # 当 stream=False 时，使用普通的创建请求
            response = await self.client.messages.create(**claudekwargs)
        #response = self.client.chat.completions.create(**openAIkwargs)
        logger.info(f"ClaudeChatModelClient::a_generate_rsp: response: {response}")

        if stream:
            # 当 stream=True 时，返回异步生成器以支持流式处理
            return self.parse_stream_response_async(response, start_time=start_time)
            #logger.info(f"ClaudeChatModelClient::a_generate_rsp: parse_stream_response_async abc: {abc}")
            #return abc
        
        else:
            # 当 stream=False 时，返回完整的消息列表
            return self.parse_non_stream_response(response)

    # 异步解析流式响应
    async def parse_stream_response_async(self, chunks: Any, start_time: float) -> AsyncIterator[List[Message]]:
        logger.info("ClaudeChatModelClient::parse_stream_response_async --- begin!")
        """解析 OpenAI 流式响应的结果（异步版本）"""
        tool_calls = []  # 存储多个工具调用
        first_packet_logged = False  # 标记是否已记录第一个报文时间

        # 使用异步迭代器遍历流式响应
        async with chunks as stream:
            async for chunk in stream:
                try:
                    logger.info(f"ClaudeChatModelClient rsp:==>{chunk}")
                    
                    # 记录第一个有效报文的时间
                    if not first_packet_logged:
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        logger.info(f"ClaudeChatModelClient =========>Time to first packet: {elapsed_time:.3f} seconds")
                        first_packet_logged = True

                    #logger.info(f"ClaudeChatModelClient::parse_stream_response_async ---ddd----chunk: {chunk} ")
                    if chunk.type == "text":
                        content = chunk.text
                        #logger.info(f"ClaudeChatModelClient::parse_stream_response_async ---yield----content: {content} ")
                        # 如果是文本直接yield出去
                        yield [Message(role=ASSISTANT, content=content)]

                    # 下面处理是 tool call部分的内容，是一次性收完再yield出去的
                    elif chunk.type == 'tool_use':
                        pass

                except (KeyError, IndexError) as e:
                    logger.error(f"Error parsing stream response: {e}, rsp: {chunk}")
                    continue

            if tool_calls:
                # 如果 role 是  FUNCTION 表示是 tool call
                logger.info(f"ClaudeChatModelClient::parse_stream_response_async ---fff----tool_calls: {tool_calls} ")
                # 注意这里的  tool_calls 返回的是整个所有被调用的 tool 集合，不是只有一个tool call调用的
                yield [Message(role=FUNCTION, content='', tool_calls=tool_calls)]
        
        #logger.info("ClaudeChatModelClient::parse_stream_response_async --- end!")

    # 解析模型的响应
    def parse_stream_response(self, chunks: Any) -> Iterator[List[Message]]:
        logger.info("ClaudeChatModelClient::parse_stream_response --- begin!")
        """解析 OpenAI 非流式响应的结果"""
        tool_calls = []  # 存储多个工具调用

        # 遍历每一个chunk 
        for chunk in chunks:
            try:
                #转成dict
                chunk = chunk.model_dump()
                delta = chunk['choices'][0]['delta']
                if delta.get('content') is not None:
                    content = delta['content']
                    # 有用户内容，将内容返回给用户
                    yield [Message(role=ASSISTANT, content=content)]
                        
                # 收集工具调用信息
                elif delta.get('tool_calls'):
                    tool_call_data = delta['tool_calls'][0]
                    index = tool_call_data.get('index', 0)

                    # 扩展 tool_calls 列表以匹配 index
                    while len(tool_calls) <= index:
                        tool_calls.append(tool_call_data)

                    # 更新工具调用字段
                    if tool_call_data['function']['arguments'] == "":
                        continue
                    else:
                        tool_calls[index]['function']['arguments']  += tool_call_data['function']['arguments']

            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing stream response: {e}, rsp: {chunk}")
                continue
        
        # 流结束后返回工具调用
        if tool_calls:
            yield [Message(role=FUNCTION, content='', tool_calls=tool_calls)]

    def parse_non_stream_response(self, chunk: Any) -> List[Message]:
        logger.info("ClaudeChatModelClient::parse_non_stream_response --- begin!")
        """解析 OpenAI 非流式响应的结果"""
        try:
            #转成dict
            chunk = chunk.model_dump()

            # 获取消息体
            message = chunk['choices'][0]['message']

            if message.get('role') == 'assistant' and message.get('content'):
                return [Message(role=ASSISTANT, content=message['content'])]
            # 如果是tool call
            elif message.get('tool_calls'):
                # 如果 role 是  FUNCTION 表示是 tool call
                return [Message(role=FUNCTION, content='', tool_calls = message['tool_calls'])]
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing non-stream response: {e}, rsp: {chunk}")
            return [Message(role=ASSISTANT, content="Error processing response")]

    # 将 NuwaBaseTool 转换为 OpenAI 支持的工具格式
    def convert_nuwa_tool_to_model_support(self, nuwa_tools: Union[Dict, List[NuwaBaseTool]]) -> List[Dict]:
        """将 NuwaBaseTool 格式转换为 OpenAI 支持的工具格式"""
        '''
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia"
                            }
                        },
                        "required": [
                            "location"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },{.....}]
        '''
        openai_tools =[]
        # 没有tool 返回空
        if not nuwa_tools:
            return openai_tools
        
        # 如果dict说明是openAI格式，直接返回
        if isinstance(nuwa_tools, dict):
            nuwa_tools = [nuwa_tools]

        # 遍历传入的 nuwa_tools
        for tool in nuwa_tools:
            tool_json = tool.openAI_format()
            # 包装成 OpenAI 标准格式
            openai_tool = {
                "type": "function",
                "function": tool_json  # 直接使用 tool.openAI_format() 的返回值作为 function 内容
            }
            openai_tools.append(openai_tool)

        return openai_tools

