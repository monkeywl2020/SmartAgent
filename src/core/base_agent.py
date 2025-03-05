import copy
import json
import os
import random
import time
from abc import ABC, abstractmethod
from pprint import pformat
import asyncio
import traceback
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, AsyncIterator

from ..llm import get_chat_model_client
from ..msg.message import  ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER,FUNCTION,TOOL, Message,ContentItem
from ..utils.common import print_traceback
from ..tools.base import NuwaBaseTool,NUWA_TOOL_REGISTRY

from loguru import logger

from .operator import Operator

#这是一个全局字典，用于存储注册的 agent 类，类的名称作为键，类的实例作为值。 AGENT_REGISTRY 用来保存已注册的所有agent。
AGENT_REGISTRY = {}

#注册agent，修饰符
def register_agent(agent_type):
    #agent不会太多，所以不用管冲突问题
    def decorator(cls):
        AGENT_REGISTRY[agent_type] = cls
        return cls

    return decorator

#这个基类，所有的 chat 类的 llm 适配的基类
class BaseAgent(Operator):
    """The base class of LLM"""

    #初始化
    def __init__(
        self,
        name: str,
        sys_prompt: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        description: str = None,
        tools_list: Optional[List[Union[str, Dict]]] = None,
        **kwargs: Any,
    ):
        logger.info("BaseAgent::__init__    --------- begin!")
        self.name = name
        self.sys_prompt = sys_prompt
        self.llm_config = llm_config or {}
        self.description = description
        self.tools_list = tools_list or []  # 记录工具
        self.llm_client = None # 挂载模型客户端
        logger.info(f"BaseAgent::__init__    --------- sys_prompt:{sys_prompt}!")
        logger.info(f"BaseAgent::__init__    --------- tools_list:{tools_list}!")
        # 初始化传入的所有 tools
        self.function_map = {} # 存储工具实例的字典
        if tools_list:
            for tool in tools_list:
                self._init_tool(tool) # 初始化工具，agent的function_map下面挂着所有注册过的工具

        # 将 kwargs 中的所有参数转换为实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)  # 无条件覆盖
            
        # 初始化 LLM 客户端
        logger.info(f"BaseAgent::__init__    ---------llm_config type:{type(llm_config)}  llm_config:{llm_config}!")
        # 模型客户端的配置
        if llm_config is not False and isinstance(llm_config, dict):
            try:
                # 获取 LLM 客户端类并实例化
                llm_client_cls = get_chat_model_client(llm_config) 
                self.llm_client = llm_client_cls(config_list=llm_config)# 传入的是 llmcfg 模型客户端的配置
                logger.info(f"Initialized LLM client for {name} with model_type: {llm_config.get('model_type')}")
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing LLM client: {e}")
                raise

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    @abstractmethod
    async def a_generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any
        ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        raise NotImplementedError

    # 这个是访问大模型，获取模型的响应接口。
    async def get_llm_reply(
        self,
        messages: List[Union[Message, Dict]],
        num_llm_calls_available: int ,
        tools: List[Dict],
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        """异步生成器函数，统一处理流式和非流式响应"""
        logger.info("BaseAgent::get_llm_reply ---- begin!")
        #--------------------------------------------------------------------------------------------------------
        # 调用大模型次数默认设置为2次，第一次调用llm，可能因为模型返回了tool call，所以在调用了 tool后，还需要调用一次获取最终结果。
        # 多次调用大模型返回多次都是tool继续调用的这种情况暂不考虑，这种情况直接修改num_llm_calls 参数即可
        #--------------------------------------------------------------------------------------------------------
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            logger.info(f"BaseAgent::get_llm_reply ---- num_llm_calls_available{num_llm_calls_available}!")
            responses = await self.llm_client.a_generate_rsp(
                messages=messages,
                stream=stream,
                delta_stream=delta_stream,
                tools=tools,
                **kwargs
            )
            #logger.info(f"BaseAgent::get_llm_reply ---- a_generate_rsp --- responses{responses}!")
            #logger.info(f"BaseAgent::get_llm_reply ---- a_generate_rsp --- type responses{type(responses)}!")

            if isinstance(responses, AsyncIterator):
                collected_responses = []
                # 遍历 迭代器中的每个消息
                #logger.info(f"BaseAgent::get_llm_reply ---- a_generate_rsp --- AsyncIterator begin !")
                async for response in responses:
                    #logger.info(f"BaseAgent::get_llm_reply ---ddd--- response:{response}!")
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
                    logger.info(f"BaseAgent::get_llm_reply ---fff--- 【has_tool_call:{has_tool_call}】 【tool_call_msg:{tool_call_msg}】 【tool_results:{tool_results}】 !")
                    if has_tool_call:
                        combined_results = [tool_call_msg] + tool_results # 组合两个消息
                        #####################################################################
                        #   combined_results 里面是 function call 的信息和
                        #       调用了tool的信息，返回的是tool的结果  
                        #       这两个消息，一个是 {role:assistant，content:function call} 
                        #                  一个是 {role:tool，content:function call result}
                        ####################################################################
                        yield combined_results  # 返回工具调用 和 工具调用的结果

                        # 如果还可以调用大模型
                        if num_llm_calls_available > 0:
                            messages.append(tool_call_msg.model_dump())   # 将 tool_call_msg 消息加入messages
                            #logger.info(f"=========1  {tool_call_msg.model_dump()} !")
                            #logger.info(f"=========2  {messages} !")
                            #将调用的结果加入到message中去
                            for tool in tool_results:
                                messages.append(tool.model_dump())    # 将 tool_results 消息加入messages
                                #logger.info(f"=========3  {tool.model_dump()} !")
                                #logger.info(f"=========4  {messages} !")
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

    #-----------------------------------------------------------------------------------------
    #                   ---** tool call的通用处理逻辑 **---
    #
    # 处理tool call的函数，解析tool call消息，这个消息是模型客户端返回的
    # 检测到后调用 tool ，并且将结果转成 Message 添加到传入的 messages 中去
     #-----------------------------------------------------------------------------------------
    async def _handle_tool_calls(
        self,
        responses: List[Message],
        **kwargs: Any
    ) -> Tuple[bool, Optional[Message], List[Message]]:
        """处理工具调用并返回工具调用消息和结果消息"""
        if not responses or responses[0].role != FUNCTION:
            return False, None, []

        use_tool, tools_info = self._detect_tool(responses[0])
        if not use_tool:
            return False, None, []

        # 创建工具调用消息
        tool_call_msg = Message(role=ASSISTANT, content='', tool_calls=responses[0].tool_calls)

        # 执行工具调用并生成结果消息
        tool_results = []
        for tool in tools_info:
            func_id, func_name, func_args, _ = tool
            tool_result = self._call_tool(func_name, func_args, **kwargs)
            # 如果返回的是dict，要放到extra中
            if isinstance(tool_result, dict):
                fn_msg = Message(
                    role=TOOL,
                    name=func_name,
                    content='',
                    extra=tool_result,
                    id=func_id
                )
            else:
                #其他的放到 content中
                fn_msg = Message(
                    role=TOOL,
                    name=func_name,
                    content=tool_result,
                    id=func_id
                )
            tool_results.append(fn_msg)

        logger.info(f"BaseAgent::_handle_tool_calls    ---------tool_results:{tool_results}!")
        return True, tool_call_msg, tool_results

    #------------------------------------------------
    # 默认使用同步接口编排
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        rsp = self.genrate_rsp(*args, **kwargs)
        return rsp

    #------------------------------------------------
    # 同步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    @abstractmethod
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:

        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        raise NotImplementedError
    
    # 将消息转成对应的格式 message 或者 dict
    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    # 初始化工具，将工具实例化，挂到  function_map 中去，作为字段的方式存储，根据名称可以直接找到该工具
    def _init_tool(self, tool: Union[str, Dict, NuwaBaseTool]):
        logger.info(f"BaseAgent::_init_tool    --------- tool{tool}!")
        if isinstance(tool, NuwaBaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                # 默认我们在 使用qwen-agent的时候，调用传入的是一个字符串
                tool_name = tool #这个函数名字
                tool_cfg = None # 这个配置是空的
            
            logger.info(f'BaseAgent::_init_tool NUWA_TOOL_REGISTRY:{NUWA_TOOL_REGISTRY}')
            if tool_name not in NUWA_TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            #这里是获取每个工具的类，并且将这个工具的类实例化，参数就是工具的配置 
            # 可以参考 retrieval.py中的 retrieval函数
            self.function_map[tool_name] = NUWA_TOOL_REGISTRY[tool_name](tool_cfg)

    #检测当前的消息是否含有工具调用，并返回 函数名，参数和 调函数的时候描述(openAI没有，有些模型是有的)
    def _detect_tool(self, message: Message) -> Tuple[bool, List[Tuple[str, str, str]]]:
        """检测消息是否为工具调用，并返回工具调用的详细信息。

        Args:
            message: 由 LLM 生成的一条消息。

        Returns:
            Tuple[bool, List[List[Union[str, None]]]]:
                - 是否需要调用工具（bool）。
                - 工具调用详情列表，每个元素为 [func_name, func_args, text]。
                如果没有工具调用，返回空列表。
        """
        # 通过 role 判断是否为工具调用
        if message.role != FUNCTION:
            return False, []
        
        # 如果 message.content 存在且不为空则使用，否则为空字符串
        text = message.content if message.content else ''

        # 如果没有 tool_calls，返回 False 和空列表
        tool_calls = message.tool_calls
        if not tool_calls:
            return False, []
        
        # 确保 tool_calls 是列表格式
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        tool_calls_info = []

        # 遍历每个 tool_call，提取详细信息
        for tool_call in tool_calls:
            # 假设 tool_call 是字典，包含 'function' 键
            if isinstance(tool_call, dict):
                # 这个是
                func_id = tool_call.get('id', None)
                function_info = tool_call.get('function', {})
                func_name = function_info.get('name')
                func_args = function_info.get('arguments')
            else:
                func_id =  getattr(tool_call, 'id', None)
                # 如果 tool_call 是对象，使用 getattr 获取属性
                func_name = getattr(tool_call, 'name', None)
                func_args = getattr(tool_call, 'arguments', None)

            # 将提取的信息添加到列表
            tool_calls_info.append([func_id, func_name, func_args, text])

        logger.info(f"BaseAgent::_detect_tool    ---------tool_calls_info:{tool_calls_info}!")
        # 返回是否有工具调用和工具调用信息列表
        return True, tool_calls_info

    # 调用函数，这个agent需要有的功能
    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            return error_message

        # 字符串 和 dict 都直接返回
        if isinstance(tool_result, str) or isinstance(tool_result, dict):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    # 将迭代器中的消息转成对应的格式 message 或者 dict
    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)

    #------------------------------------------------
    # 暂不支持多模态输入
    #
    #------------------------------------------------
    @property
    def support_multimodal_input(self) -> bool:
        # Does the model support multimodal input natively? It affects how we preprocess the input.
        return False

    #------------------------------------------------
    # 暂不支持多模态输出
    #
    #------------------------------------------------
    @property
    def support_multimodal_output(self) -> bool:
        # Does the model generate multimodal outputs beyond texts? It affects how we post-process the output.
        return False

# 获取agent 类型
def get_chat_agent_cls(agent_type_name: Optional[str] = 'emotional_agent') -> BaseAgent:
    """
    根据 agent_type_name 从 AGENT_REGISTRY 中获取对应的 Agent 类。
    
    Args:
        agent_type_name (Optional[str]): Agent 类型名称，默认值为 'emotional_agent'。
    
    Returns:
        Type[BaseAgent]: 对应的 Agent 类。
    
    Raises:
        KeyError: 如果指定的 agent_type_name 未在 AGENT_REGISTRY 中注册。
    """
    if agent_type_name not in AGENT_REGISTRY:
        raise KeyError(f"Agent type '{agent_type_name}' is not registered in AGENT_REGISTRY. "
                       f"Available agents: {list(AGENT_REGISTRY.keys())}")
    # 返回类名
    return AGENT_REGISTRY[agent_type_name]