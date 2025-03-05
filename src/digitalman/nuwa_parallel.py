from abc import abstractmethod
import asyncio
import time
import queue
from collections import deque
from typing import Any, Dict, Iterator, List, Optional, Union,AsyncIterator
from loguru import logger

from ..tools import agent_tools
from ..tools import qwen_agent_tools
from ..core.base_digital_man import BaseDigitalMan, register_digitalman
from ..msg.message import Message,USER,ASSISTANT,TOOL
from ..localdb.history_db_interface import get_history_record,save_history_record

# 数字人类型名称  nuwa
#   挂载2个 agent，一个用于聊天的情感模型，一个用于 tool call调用的模型
@register_digitalman('nuwa_parallel')
class nuwaParallel(BaseDigitalMan):
    """The base class of LLM"""
    '''
    配置样例 
    cfg1 = {
        'agentcfg': [
            {
                'agent_type': 'emotional_agent',
                'agent_name': 'agent_abc',
                'sys_prompt': emotional_prompt,
                'tools_list': ['switch_to_taskaction'],
                'description': '专门用来情感聊天的agent',
                'extra': 'for other',
                'llmcfg': {
                    "model_type": "postapi",
                    'model': 'xiaoyun-3.5-turbo',
                    'base_url': 'http://39.105.35.180:20012/v1',  
                    'api_key': 'EMPTY',
                    "temperature": 0.95,
                    "top_p": 0.35,
                    "repetition_penalty": 1.05,
                    "top_k": -1,
                    "stream": True,
                }
            },
            {
                'agent_type': 'taskaction_agent',
                'agent_name': 'agent_wl',
                'sys_prompt': task_prompt,
                'tools_list': ['recommended_service','switch_to_emotional'],
                'description': '专门用来 tool call 调用执行任务的agent',
                'llmcfg': {
                    "model_type": "openai",
                    'model': '/data2/models/Qwen2.5-72B-Instruct-AWQ',
                    'base_url': 'http://172.21.30.230:8980/v1',  
                    'api_key': 'EMPTY',
                    'top_p': 0.8,
                    'temperature': 0.1,
                }
            }
        ],
        'workmode': 'switch'
    }   
    '''
    #  workmode 是 数字人内部 agent使用模式 ，一共 3种 ： [parallel, preprocess ,switch] 
    #  没传工作模式默认就用 switch模式
    #初始化
    def __init__(
        self,
        name: str,
        llm_config: Optional[Dict[str, Any]] = None,
        description: str = None,
        workmode: Optional[str] = 'switch',
        **kwargs: Any,
    ):
        logger.info(f"nuwa::__init__  ------begin!")
        super().__init__(name=name, llm_config=llm_config, description=description, workmode = workmode, **kwargs)

        # 获取2个 agent的名称，保存当前正使用的序号
        self.agents_name = list(self.agents.keys())  # 提取所有 agent_name 到列表中
        self.emotional_agent_index = 0  # 情感聊天 agent
        self.tool_agent_index = 1  # tool调用 agent

        # 创建两个独立的队列，这两个队列，一个用来存放  emotional agent 的消息，一个用来存放  task agent的消息。
        self.emotional_queue = asyncio.Queue() # 异步队列
        self.task_queue = asyncio.Queue()# 异步队列

        #==============================
        #   数据库 客户端 mysql ，redis
        #   需要根据用户的信息来读取这个用户的聊天记录，记忆等等内容。
        #   记忆：用户的对话历史记录，肯定是要按会话来的，每个会话都是 用户与大模型的聊天记录。 打算用纯文本来存储
        #   
        #==============================

    # 获取用户的聊天记录记忆内容
    def get_user_history(self, current_messages: List[Dict], user_id: str, session_id: str, n: int = 3) -> List[Dict]:
        """
        获取指定用户和会话的最近 N 轮历史聊天记录，并将当前 messages 追加到末尾。
        参数:
            current_messages: 当前整理后的消息列表
            user_id: 用户 ID
            session_id: 会话 ID
            n: 获取最近 N 轮历史记录，默认为 3
        返回:
            包含历史记录和当前消息的 messages 列表
        """
        if not user_id or not session_id:
            return current_messages  # 如果没有用户或会话信息，直接返回当前消息

        # 获取特定用户和会话的 取最近 N 轮历史记录 
        session_history = get_history_record(user_id=user_id,session_id=session_id,n=n)

        # 只保留 'role' 和 'content' 字段
        processed_messages = [{key: message[key] for key in ['role', 'content']} for message in session_history]

        # 确保历史记录的第一条是 role 为 user 的消息
        while processed_messages and processed_messages[0]['role'] != 'user':
            processed_messages.pop(0)  # 去掉第一条不是 user 的消息

        # 返回历史记录 + 当前消息
        return processed_messages + current_messages

    def save_chat_history(
        self,
        input_messages: List[Dict] = None,
        response: List[Message] = None,
        user_id: str = None,
        session_id: str = None
    ) -> None:
        """保存用户的输入消息和完整的响应消息（Function call 不保存）"""
        if not user_id or not session_id:
            logger.warning("缺少 user_id 或 session_id，无法保存历史记录")
            return

        if input_messages:
            for msg in input_messages:
                #session_history.append({"role": msg["role"], "content": msg["content"]})
                save_history_record(user_id, session_id, msg["role"], msg["content"])
                logger.info(f"BaseDigitalMan::save_chat_history  ------saved user msg: {msg}")
            #logger.info(f"BaseDigitalMan::save_chat_history  ------saved user input_messages: {input_messages}")

        if response:
            for msg in response:
                if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
                    save_history_record(user_id, session_id, ASSISTANT, msg.content)
                elif isinstance(msg, dict) and msg.get("content") and not msg.get("tool_calls"):
                    save_history_record(user_id, session_id, ASSISTANT, msg["content"])
            logger.info(f"BaseDigitalMan::save_chat_history  ------saved response: {response}")

    # 处理用户输入的消息，提取 user_id 和 session_id，整理为标准格式。
    def prepare_user_message(self, messages: List[Dict]) -> tuple[List[Dict], str, str]:
        """
        处理用户输入的消息，提取 user_id 和 session_id，整理为标准格式。
        参数:
            messages: 用户输入的消息列表，包含 user_id, session_id 等字段
        返回:
            (prepared_messages, user_id, session_id)
            - prepared_messages: 整理后的消息列表，仅保留 role 和 content
            - user_id: 用户 ID
            - session_id: 会话 ID
        """
        if not messages:
            raise ValueError("消息列表为空")

        # 假设所有消息都有相同的 user_id 和 session_id，取第一条消息提取
        first_msg = messages[0]
        user_id = first_msg.get("user_id")
        user_age = first_msg.get("user_age") # 星云模型需要使用
        session_id = first_msg.get("session_id")

        if not user_id or not session_id:
            raise ValueError("消息必须包含 user_id 和 session_id")

        # 整理消息，仅保留 role 和 content
        prepared_messages = [
            {"role": msg.get("role", USER), "content": msg.get("content", "")}
            for msg in messages
        ]
        return prepared_messages, user_id, user_age, session_id

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    async def a_generate_rsp(
        self,
        messages: List[Union[Message, Dict]],
        **kwargs: Any,
    ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info(f"BaseDigitalMan::a_generate_rsp  ------begin! kwargs:{kwargs}\nmessages:{messages}")

        #-------------------------------------------------------------------------
        # 1:获取当前选中的 agent 名称和实例
        #   根据当前的 current_index 获取名称，根据名称获取agent 
        #-------------------------------------------------------------------------
        emotional_agent_name = self.agents_name[self.emotional_agent_index]
        tools_agent_name = self.agents_name[self.tool_agent_index]
        emotional_agent = self.agents.get(emotional_agent_name)
        tools_agent = self.agents.get(tools_agent_name)

        logger.info(f"BaseDigitalMan::a_generate_rsp  ------begin! emotional_agent name:{emotional_agent.name} tools_agent name:{tools_agent.name}")

        if not emotional_agent or not tools_agent:
            logger.error(f"未找到所需的agent - emotional_agent:{emotional_agent}, task_agent:{tools_agent}")
            yield []
            return

        #-------------------------------------------------------------------------        
        # 2: 处理用户消息，提取 user_id 和 session_id，并整理格式
        #    解析用户的 消息，将 user_id 和 session_id获取出来，并且将消息转成标准格式
        #     user_age 是星云使用的参数，用户年龄
        #-------------------------------------------------------------------------
        try:
            prepared_messages, user_id, user_age, session_id = self.prepare_user_message(messages)
            logger.info(f"BaseDigitalMan::a_generate_rsp  ------prepared_messages: {prepared_messages}, user_id: {user_id}, session_id: {session_id}")
        except ValueError as e:
            logger.error(f"处理用户消息失败: {e}")
            yield []
            return
        #-------------------------------------------------------------------------        
        # 3: 获取用户的历史聊天记录
        #     将历史聊天记录加入到用户消息中去
        #-------------------------------------------------------------------------
        # 获取包含历史记录的 messages
        #updated_messages = prepared_messages
        updated_messages = self.get_user_history(prepared_messages, user_id, session_id, n=kwargs.get("history_n", 3))
        logger.info(f"BaseDigitalMan::a_generate_rsp  ------updated_messages with history: {updated_messages}")

        #-------------------------------------------------------------------------        
        # 4: 保存用户的输入信息
        #     将用户消息加入到聊天记忆中去
        #-------------------------------------------------------------------------
        # 获取包含历史记录的 messages
        self.save_chat_history(input_messages = prepared_messages, user_id = user_id, session_id = session_id)
        logger.info(f"BaseDigitalMan::save_chat_history  ------save user message to history")

        #-------------------------------------------------------------------------        
        # 5: 携程并发执行两个agent的请求 
        #    定义协程函数  
        #-------------------------------------------------------------------------
        # emotional_agent 请求
        async def process_emotional_agent_to_queue():
            async for response in self._process_agent_responses(
                emotional_agent, 
                updated_messages, 
                user_id=user_id, 
                user_age=user_age, 
                num_llm_calls=1, 
                **kwargs
            ):
                # 并发的协程将响应加到队列中去
                for msg in response:
                    await self.emotional_queue.put(msg) # 将响应添加到队列里面去
            await self.emotional_queue.put(None)  # 表示结束

        # task_agent 请求
        async def process_task_agent_to_queue():
            async for response in self._process_agent_responses(
                tools_agent, 
                updated_messages, 
                user_id=user_id, 
                user_age=user_age, 
                num_llm_calls=1, 
                **kwargs
            ):
                for msg in response:
                    await self.task_queue.put(msg)# 将响应添加到队列里面去
            await self.task_queue.put(None)  # 表示结束

        # 启动协程任务
        start_time = time.perf_counter()
        asyncio.create_task(process_emotional_agent_to_queue())
        asyncio.create_task(process_task_agent_to_queue())
        
        emotional_packet_logged = False
        task_packet_logged = False
        emotional_contents = []  # 用于保存emotional的完整响应

        # 主循环处理队列
        # 首先处理 emotional 队列
        #----------------------------------处理 emotional 响应 --------------------------------------------------------------
        while True:
            msg = await self.emotional_queue.get()  # 异步等待，无需 timeout
            if msg is None:  # emotional处理完成
                break

            logger.info(f"emotional agent=========>packet: {msg}")

            # 直接yield emotional消息
            if hasattr(msg, "content") and msg.content:
                if not emotional_packet_logged:
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"emotional agent=========>Time to first packet: {elapsed_time:.3f} seconds")
                    emotional_packet_logged = True

                emotional_contents.append(msg.content) # 这个是要存历史记录的
                logger.info(f"emotional agent=========>yield packet: {msg}")
                yield [msg]

        # emotional处理完成后，保存历史
        if emotional_contents:
            final_response = Message(role=ASSISTANT, content="".join(emotional_contents))
            self.save_chat_history(response=[final_response.model_dump()], user_id=user_id, session_id=session_id)

        #----------------------------------处理 task 响应 -----------------------------------------------------------------
        # 然后处理task队列
        while True:
            msg = await self.task_queue.get()# 异步等待，无需 timeout
            if msg is None:  # task处理完成
                break
            
            logger.info(f"task agent=========>packet: {msg}")

            # 处理task消息
            # 检查是否是带有tool_calls的ASSISTANT消息
            if (msg.role == ASSISTANT 
                and hasattr(msg, "content") 
                and msg.content == "" 
                and hasattr(msg, "tool_calls") 
                and msg.tool_calls):
                #-------------------------------------------------------------------------
                #   记录首包时间
                #-------------------------------------------------------------------------
                if not task_packet_logged:
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"task agent=========>Time to first packet: {elapsed_time:.3f} seconds")
                    task_packet_logged = True

                yield [msg]
            # 检查是否是TOOL角色的结果消息
            elif msg.role == TOOL and hasattr(msg, "content"):
                yield [msg]

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"a_generate_rsp=========>Total execution time: {elapsed_time:.3f} seconds")

        # 在清空队列时记录日志
        logger.debug(f"Clearing emotional_queue, remaining items: {self.emotional_queue.qsize()}")
        logger.debug(f"Clearing task_queue, remaining items: {self.task_queue.qsize()}")
        while not self.emotional_queue.empty():
            await self.emotional_queue.get()
        while not self.task_queue.empty():
            await self.task_queue.get()

        #---------------------------------  下面是废弃的代码  -----------------------------------------------------------------
        if 0:
            async def process_emotional_agent():
                response_contents = []
                async for response in self._process_agent_responses(emotional_agent, updated_messages, user_id=user_id, user_age=user_age, num_llm_calls =1, **kwargs):
                    for msg in response:
                        logger.info(f"nuwaParallel::process_emotional_agent  ---emotional---ddd---- : {response}")
                        # 只收集普通文本响应：content不为空且tool_calls不存在或为空
                        if (hasattr(msg, "content") 
                            and msg.content 
                            and (not hasattr(msg, "tool_calls") or not msg.tool_calls)):
                            response_contents.append(msg.content)
                    yield response
            
            # tool_agent 请求
            async def process_task_agent():
                task_results = []
                async for response in self._process_agent_responses(tools_agent, updated_messages, user_id=user_id, user_age=user_age, num_llm_calls =1, **kwargs):
                    logger.info(f"nuwaParallel::process_task_agent  ---task---ddd---- : {response}")
                    for msg in response:
                        # 收集所有结果，交给上层处理
                        task_results.append(msg)

                return task_results
            
            # 并发执行
            start_time = time.perf_counter() # 记录开始时间
            emotional_packet_logged = False
            task_packet_logged = False


            emotional_task = process_emotional_agent() # 执行 emotional_agent 请求
            tools_task = process_task_agent()          # 执行 tool_agent 请求

            # 实时输出emotional_agent的结果
            emotional_contents = []
            async for emotional_response in emotional_task:
                for msg in emotional_response:
                    if hasattr(msg, "content") and msg.content:
                        #-------------------------------------------------------------------------
                        #   记录首包时间
                        #-------------------------------------------------------------------------
                        if not emotional_packet_logged:
                            emotional_fist_packet_time = time.perf_counter()
                            elapsed_time = emotional_fist_packet_time - start_time
                            logger.info(f"emotional agent=========>Time to first packet: {elapsed_time:.3f} seconds")
                            emotional_packet_logged = True

                        emotional_contents.append(msg.content)
                yield emotional_response

            emotional_end_time = time.perf_counter()
            elapsed_time = emotional_end_time - start_time
            logger.info(f"emotional agent=========>end Time of packet: {elapsed_time:.3f} seconds")

            start_time = time.perf_counter() # 记录开始时间
            # 等待task_agent完成并处理结果
            task_results = await tools_task

            # 处理 emotional_agent 的完整响应，将 emotional_agent 的响应作为 回答的 历史记录保存下来。
            final_emotional_response = [Message(role=ASSISTANT, content="".join(emotional_contents))] if emotional_contents else []
            if final_emotional_response:
                self.save_chat_history(response=[final_emotional_response[0].model_dump()], user_id=user_id, session_id=session_id)

            # 处理task_agent的函数调用结果
            if task_results:
                for task_msg in task_results:
                    # 检查是否是带有tool_calls的ASSISTANT消息
                    if (task_msg.role == ASSISTANT 
                        and hasattr(task_msg, "content") 
                        and task_msg.content == "" 
                        and hasattr(task_msg, "tool_calls") 
                        and task_msg.tool_calls):
                        #-------------------------------------------------------------------------
                        #   记录首包时间
                        #-------------------------------------------------------------------------
                        if not task_packet_logged:
                            task_fist_packet_time = time.perf_counter()
                            elapsed_time = task_fist_packet_time - start_time
                            logger.info(f"task agent=========>Time to first packet: {elapsed_time:.3f} seconds")
                            task_packet_logged = True

                        yield [task_msg]
                    
                    # 检查是否是TOOL角色的结果消息
                    elif task_msg.role == TOOL and hasattr(task_msg, "content"):
                        yield [task_msg]


    # 异步 agent响应处理
    async def _process_agent_responses(
        self,
        agent,
        messages: List[Union[Message, Dict]],
        **kwargs: Any
    ) -> AsyncIterator[List[Message]]:
        """处理 agent 的响应并支持动态切换"""
        try:
            # 调用agent 获取响应
            responses = await agent.a_generate_rsp(messages = messages,**kwargs)
            logger.info(f"nuwaParallel::_process_agent_responses ---responses: {responses}, type: {type(responses)}")
            if isinstance(responses, AsyncIterator):
                # 处理流式响应
                async for response in responses:
                    logger.info(f"nuwa::a_generate_rsp  ------response:{response}")
                    yield response
                
            else:
                # 非流式响应直接返回
                logger.info(f"BaseDigitalMan::a_generate_rsp  ------type responses:{type(responses)}")
                yield responses

        except Exception as e:
            logger.exception(f"调用 agent 的 a_generate_rsp 方法失败: {str(e)}")
            yield []
  
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
    def generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:

        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        pass

    # 将消息转成对应的格式 message 或者 dict
    def _convert_messages_to_target_type(self, messages: List[Message],
                                         target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    # 将迭代器中的消息转成对应的格式 message 或者 dict
    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)
