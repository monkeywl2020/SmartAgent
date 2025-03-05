import json
import asyncio
from abc import ABC,abstractmethod
from typing import Any, Iterator, Dict, List, Optional, Generator, Tuple, Union,Sequence,Literal,AsyncIterator
import inspect
import time
import copy
from collections import namedtuple


from loguru import logger

from ..tools.base import NuwaBaseTool
from .base import BaseChatModelClient, register_llm

from qwen_agent.tools import BaseTool
from qwen_agent.agents import Assistant
from qwen_agent.utils.utils import has_chinese_messages, merge_generate_cfgs

from ..msg.message import  ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, CONTENT,USER,FUNCTION,TOOL,ROLE, Message,ContentItem,FunctionCall
from qwen_agent.llm.schema import Message as qwenMessage
from qwen_agent.llm.schema import FunctionCall as qwenFunctionCall

# 这个类是继承了qwen-agent的，使用qwen-agent机制，完全使用qwen大模型的格式
class nuwaQwenLlmClient(Assistant):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Dict] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = "nuwaQwenAgent",
                 description: Optional[str] = "用来调用qwen-agent的适配agent模块",
                 max_llm_call_num:Optional[int] = 1,
                 files: Optional[List[str]] = None):
        logger.info(f"nuwaQwenLlmClient::__init__-----------begin! max_llm_call_num:{max_llm_call_num}")
        super().__init__(
            function_list=function_list,
            llm=llm,
            system_message=system_message,
            name=name,
            description=description,
            files=files,
        )


    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.          # 返回基于接收消息的响应生成器。

        This method performs a uniform type conversion for the inputted messages, # 该方法对输入的消息进行统一的类型转换，
        and calls the _run method to generate a reply.                            # 并调用 `_run` 方法生成回复。

        Args:
            messages: A list of messages.  消息列表，类型为字典或 Message 对象的列表。
            **kwargs: 其他可选参数。

        Yields:                             
            The response generator. #响应生成器，返回的响应可能是 Message 对象列表或字典列表。
        """
        messages = copy.deepcopy(messages)
        new_messages = [] # 用于存储处理后的消息。

        # 遍历 messages，如果某条消息是字典类型，则将其转换为 Message 对象并添加到 new_messages 中。
        new_messages = [qwenMessage(**msg) if isinstance(msg, dict) else msg for msg in messages]

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        # 直接返回
        return self._run(messages=new_messages, **kwargs)

    # 重写 qwen-agent 的 Assistant 的 _run 执行函数
    def _run(self,
             messages: List[qwenMessage],
             lang: Literal['en', 'zh'] = 'en',
             knowledge: str = '',
             **kwargs) -> Iterator[List[Message]]:
        logger.info(f"nuwaQwenLlmClient::_run-----------begin!")

        # 上层带下来的
        num_llm_calls = kwargs.pop('num_llm_calls', 2) # 去掉这个参数
        start_time = kwargs.pop('start_time', 0) # 去掉这个参数
        first_packet_logged = False  # 标记是否已记录第一个报文时间
        
        # 将 知识库加入消息中去
        new_messages = self._prepend_knowledge_prompt(messages=messages, lang=lang, knowledge=knowledge, **kwargs)
        messages = new_messages # 带知识库的消息，如果知识库有的话

        messages = copy.deepcopy(messages)
        num_llm_calls_available = num_llm_calls # 最大 LLM 调用次数
        response = [] # 存储响应的列表

        # 开始执行 LLM 调用
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1 # 每次调用 LLM 时，减少可用调用次数

            extra_generate_cfg = {'lang': lang} # 设置额外的生成配置参数
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']  # 如果有种子值，传递给生成配置
            
            # 调用 LLM 获取响应流
            output_stream = self._call_llm(messages=messages,
                                           functions=[func.function for func in self.function_map.values()],
                                           extra_generate_cfg=extra_generate_cfg)
            previous_content = ""  # 用于保存之前的完整内容

            # 这里存放的是 function msg list   #Message(role = ASSISTANT, content="",tool_calls=out.function_call.model_dump())
            func_msg_list = [] # 存放 function 调用的消息 列表
            func_rst_list = [] # 存放 function result的消息 列表
            is_function_call_detected = False  # 标记是否检测到 function_call
            
            # 遍历 LLM 输出的迭代器
            for output_list in output_stream:
                usermsg_output_list = [] 
                if output_list: # 如果 LLM 返回了有效的输出,不是空列表
                    
                    ############################################################################
                    #       打印 qwen 大模型回的响应信息 ，流式响应，用来定位问题，不要删除
                    #       流式打印信息，显示从大模型那边收到的流
                    ############################################################################
                    logger.info(f"nuwaQwenLlmClient::_run=========>output_list: {output_list}")

                    # 记录第一个有效报文的时间
                    if not first_packet_logged:
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        logger.info(f"nuwaQwenLlmClient=========>Time to first packet: {elapsed_time:.3f} seconds")
                        first_packet_logged = True

                    # 遍历流式中一个报文的消息list
                    for out in output_list:
                        # 如果消息有内容，并且不为空，则yield出去给用户
                        if out.get("content") not in (None, ''):
                            current_content = out.get('content', '') # 首先获取当前content内容
                            new_content = current_content[len(previous_content):]  # 获取新增部分内容
                            if new_content:  # 只有当 new_content 不为空时才生成 Message
                                # 更新 previous_content 为当前消息的完整内容
                                previous_content = current_content

                                user_msg = Message(role = out.get("role"),content = new_content)
                                usermsg_output_list.append(user_msg)  # 添加到new_output_list 
                    
                        if 0: # 下面这段代码在 function call调用的时候能输出一段描述，可以延缓等待的空白时间
                            # 检测 function_call 并输出闪动提示
                            if hasattr(out, 'function_call') and out.function_call:
                                if not is_function_call_detected:
                                    # 首次检测到 function_call，输出完整提示
                                    is_function_call_detected = True
                                    prompt_msg = Message(role="assistant", content="正在为您获取推荐内容")
                                    usermsg_output_list.append(prompt_msg)
                                else:
                                    # 后续检测到 function_call，增量输出“.”
                                    prompt_msg = Message(role="assistant", content=".")
                                    usermsg_output_list.append(prompt_msg)

                    # 如果有 用户消息，yield出去
                    if usermsg_output_list:# 用户的消息列表不能为空
                        logger.info(f"nuwaQwenLlmClient=========> packet: {usermsg_output_list}")
                        yield usermsg_output_list

            # 当有tool调用的时候，用户结束信息的时候添加一个 回车换行
            if is_function_call_detected:
                yield [Message(role="assistant", content="\n")]

            # 下面这个是 流式输出遍历完成后执行的。
            if output_list:# 如果有有效输出
                messages.extend(output_list)  # 将迭代结束后的输出添加到消息中
                used_any_tool = False # 标记是否使用了工具

                # 遍历每个输出，检查是否需要调用工具
                for out in output_list:
                    # 检查该输出是否需要使用工具，及工具名称和参数
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        logger.info(f"nuwaQwenLlmClient::_run=========>use_tool:{use_tool} tool_name:{tool_name} tool_args:{tool_args}")
                        # 如果需要使用工具，调用工具并获取结果
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        #-------------------------------------------
                        #  需要增加一个  switch_to 函数结果的特殊处理流程。
                        #-------------------------------------------
                        if 'switch_mode_to' in tool_result:
                            try:
                                # 将字符串转换为字典
                                data_dict = json.loads(tool_result)
                                #解析成功
                                switch_msg = Message(role = TOOL,
                                        name= tool_name,
                                        content='',
                                        extra=data_dict)
                                logger.info(f"nuwaQwenLlmClient::_run ---get a switch_mode_to Msg:{switch_msg}")
                                yield [switch_msg] # 返回 switch_mode_to 切换模式结果

                                return # 结束函数
                            except json.JSONDecodeError as e:
                                logger.info(f"nuwaQwenLlmClient::_run: =====> 解析失败: {e}")
                        else:
                            # 如果是 func call ，则将当前 function加入  func_msg_list
                            function_msg = Message(role = ASSISTANT, content="",tool_calls=[out.function_call.model_dump()])
                            func_msg_list.append(function_msg)

                            # 这个是一个tool调用结果消息
                            func_rst_msg = Message(role = TOOL, name= tool_name, content=tool_result)
                            func_rst_list.append(func_rst_msg)

                        # 创建一个 Message 对象，表示工具调用的结果
                        fn_msg = qwenMessage(
                            role=FUNCTION,
                            name=tool_name,
                            content=tool_result,
                        )
                        # 将工具调用结果添加到消息和响应中
                        messages.append(fn_msg)
                        used_any_tool = True# 标记已使用工具

                ## 如果没有使用任何工具，退出循环
                if not used_any_tool:
                    break
                else:
                    # 如果有工具调用
                    yield func_msg_list + func_rst_list
        
        yield []
    
#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
@register_llm('qwen')
class QwenChatModelClient(BaseChatModelClient):

    # llm 客户端适配
    def __init__(self, 
                 config_list: Optional[Dict[str, Any]] = None, 
                 **kwargs: Any):
        logger.info("QwenChatModelClient-----------__init__")
        if config_list:
            # 如果有内容，直接展开作为参数传入 BaseChatModelClient 作为关键字保存
            super().__init__(config_list=config_list, **config_list, **kwargs)
        else:
            super().__init__(config_list=config_list, **kwargs)
        
        # 获取 qwen-agent要使用的 llm的配置
        self.qwen_llm_cfg = config_list.get('qwenllmcfg',None)
        self.tool_list = config_list.get('tools_list',[])
        # 设置 max_tokens，默认值 4096
        self.max_tokens = kwargs.get('max_tokens', 4096)
        logger.info(f"QwenChatModelClient-----------qwen_llm_cfg:{self.qwen_llm_cfg} tool_list:{self.tool_list} max_tokens{self.max_tokens}")

        #------------------------------------------------------------------------------------------------------
        # max_llm_call_num 设置为1表示如果有 tool call，则调用了tool后就返回，不再请求大模型
        #   max_llm_call_num 大于1，一旦有tool调用，调用获取到结果后，还会调用大模型对结果进行重新整理再回答用户的问题。
        #   切换模式的时候，本agent处理就结束了，不需要其他处理。
        #------------------------------------------------------------------------------------------------------
        self.client = nuwaQwenLlmClient(llm=self.qwen_llm_cfg,
                                        system_message = self.sys_prompt,
                                        function_list=self.tool_list ,
                                        max_llm_call_num = 1)

    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            tools: Optional[List[Dict]] = None,
            stream: bool = True,
            delta_stream: bool = False,
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        pass

    # 异步生成响应
    async def a_generate_rsp(
        self,
        messages: List[Union[Message, Dict]], # 实际处理都是dict，不是message
        tools: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        **kwargs: Any,
    ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        # 当前只支持流式的，非流式的后面有空再说
        logger.info(f"QwenChatModelClient::a_generate_rsp: openAIkwargs:\n{messages}")
        num_llm_calls = kwargs.pop('num_llm_calls', 2) # 去掉这个参数
        start_time = time.perf_counter()
        response = self.client.run(messages = messages, num_llm_calls = num_llm_calls, start_time = start_time)
        logger.info(f"QwenChatModelClient::a_generate_rsp: response: {response}")
        return self.parse_stream_response_async(response)
    
    # 异步解析流式响应
    async def parse_stream_response_async(self, chunk_stream: Any) -> AsyncIterator[List[Message]]:
        # 将同步迭代器 改成异步迭代器
        for chunk in chunk_stream:
            if chunk:
                yield chunk

    # 解析模型的响应，同步流式，这个代码未测试，目前没这个场景先不管
    def parse_stream_response(self, chunks: Any) -> Iterator[List[Message]]:
        logger.info("OpenAiChatModelClient::parse_stream_response --- begin!")
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

    # 目前没这个场景先不管
    def parse_non_stream_response(self, chunk: Any) -> List[Message]:
        logger.info("OpenAiChatModelClient::parse_non_stream_response --- begin!")
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

    # 将 NuwaBaseTool 转换为 OpenAI 支持的工具格式， qwen-agent模式下，这个函数没用 返回的是空，上层agent不需要处理 tool调用
    def convert_nuwa_tool_to_model_support(self, nuwa_tools: Union[Dict, List[NuwaBaseTool]]) -> List[Dict]:
        # 这个函数不需要
        pass