from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union,AsyncIterator
from loguru import logger

from ..tools import agent_tools
from ..tools import qwen_agent_tools
from ..core.base_digital_man import BaseDigitalMan, register_digitalman
from ..msg.message import Message,USER,ASSISTANT
from ..localdb.history_db_interface import get_history_record,save_history_record

# 数字人类型名称  nuwa
#   挂载2个 agent，一个用于聊天的情感模型，一个用于 tool call调用的模型
@register_digitalman('nuwa')
class nuwa(BaseDigitalMan):
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
        self.current_index = 0  # 当前选中的 agent 索引

        #==============================
        #   数据库 客户端 mysql ，redis
        #   需要根据用户的信息来读取这个用户的聊天记录，记忆等等内容。
        #   记忆：用户的对话历史记录，肯定是要按会话来的，每个会话都是 用户与大模型的聊天记录。 打算用纯文本来存储
        #   
        #==============================

    # 通过索引切换agent
    def switch_agent(self, index: Optional[int] = None, name: Optional[str] = None):
        """
        切换代理，支持按索引或名称。
        
        Args:
            index (Optional[int]): 代理的索引。
            name (Optional[str]): 代理的名称。
        
        Raises:
            ValueError: 如果 index 和 name 同时提供，或均未提供，或值无效。
        """
        # 检查参数互斥性
        if (index is not None and name is not None) or (index is None and name is None):
            raise ValueError("必须且只能提供 index 或 name 中的一个参数")

        # 按索引切换
        if index is not None:
            if 0 <= index < len(self.agents_name):
                self.current_index = index
                logger.info(f"Switched to agent by index: {index} ({self.agents_name[index]})")
            else:
                logger.error(f"Invalid index: {index}, range: 0 to {len(self.agents_name) - 1}")
                raise ValueError(f"索引 {index} 超出范围")
        
        # 按名称切换
        elif name is not None:
            if name in self.agents_name:
                self.current_index = self.agents_name.index(name)
                logger.info(f"Switched to agent by name: {name}")
            else:
                logger.error(f"Invalid agent name: {name}, available: {self.agents_name}")
                raise ValueError(f"代理名称 {name} 不存在")

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
        current_agent_name = self.agents_name[self.current_index]
        current_agent = self.agents.get(current_agent_name)
        logger.info(f"BaseDigitalMan::a_generate_rsp  ------begin! current_agent_name:{current_agent_name} current_agent:{current_agent}")

        if current_agent is None:
            logger.error(f"未找到对应的代理: {current_agent_name}")
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
        # 5：使用agent处理用户的消息
        #   调用独立的处理函数，这个函数可以递归调用，用于切换agent后的场景。
        #   切换agent后需要重新调用新的agent处理当前的消息。这个处理是切换后重新执行的agent操作。agent是新agent
        #-------------------------------------------------------------------------  
        response_contents = []  # 用于拼装完整的响应内容, 将 user_id 和 user_age 作为参数传入，后面postapi模型客户端会用到。
        async for response in self._process_agent_responses(current_agent, updated_messages, user_id = user_id, user_age = user_age, **kwargs):
            # 拼装流式响应的内容
            for msg in response:
                # 只收集普通文本响应：content不为空且tool_calls不存在或为空
                if (hasattr(msg, "content") 
                    and msg.content 
                    and (not hasattr(msg, "tool_calls") or not msg.tool_calls)):
                    response_contents.append(msg.content)

                #logger.info(f"BaseDigitalMan::a_generate_rsp  ------partial response: {response}")
                yield response # 实时返回流式片段给调用方

        # 拼装完整的 final_response,拼装完整后，就只有1条消息
        final_response = [Message(role=ASSISTANT,content="".join(response_contents))] if response_contents else []

        #-------------------------------------------------------------------------        
        # 6: 保存用户的输入信息
        #     将用户消息加入到聊天记忆中去
        #-------------------------------------------------------------------------
        if final_response:
            self.save_chat_history(response=[final_response[0].model_dump()], user_id=user_id, session_id=session_id)
            logger.info(f"BaseDigitalMan::save_chat_history  ------save final response to history: {final_response}")

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
            responses = await agent.a_generate_rsp(messages = messages, num_llm_calls = 2, **kwargs)
            if isinstance(responses, AsyncIterator):
                # 处理流式响应
                async for response in responses:
                    logger.info(f"nuwa::a_generate_rsp  ------response:{response}")
                    switch_triggered = False
                    for message in response:
                        # 检查是否需要切换 agent
                        if (message.content == '' 
                            and isinstance(message.extra, dict) 
                            and "switch_mode_to" in message.extra):
                            logger.info(f"nuwa::a_generate_rsp --- Switching to {message.extra['switch_mode_to']}")
                            # 切换到新 agent
                            self.switch_agent(name=message.extra['switch_mode_to'])
                            switch_triggered = True
                            break  # 跳出当前 response 的处理
                    
                    # 如果触发了切换，则用新 agent 重新处理
                    if switch_triggered:
                        new_agent_name = self.agents_name[self.current_index]
                        new_agent = self.agents.get(new_agent_name)
                        if new_agent:
                            logger.info(f"nuwa::a_generate_rsp --- Re-processing with new agent: {new_agent_name}")
                            async for new_response in self._process_agent_responses(new_agent, messages, **kwargs):
                                yield new_response
                        else:
                            logger.error(f"切换后的 agent 未找到: {new_agent_name}")
                            yield []
                        return  # 结束当前流，避免重复处理
                    
                    # 如果没有切换，则正常转发响应
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


    # 备份的代码，这是没有加数据库历史聊天记录的代码
    async def a_generate_rsp1(
            self,
            messages: List[Union[Message, Dict]],
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        logger.info(f"BaseDigitalMan::a_genrate_rsp  ------begin! kwargs:{kwargs}\nmessages:{messages}")
        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。

        # 获取当前选中的 agent 名称
        current_agent_name = self.agents_name[self.current_index]
        # 获取当前代理的实例
        current_agent = self.agents.get(current_agent_name)
        logger.info(f"BaseDigitalMan::a_genrate_rsp  ------begin! current_agent_name:{current_agent_name} current_agent:{current_agent}")

        if current_agent is None:
            logger.error(f"未找到对应的代理: {current_agent_name}")
            yield []
        
        # 调用当前 agent 的 a_generate_rsp 方法生成响应
        try:
            responses = await current_agent.a_generate_rsp(messages, **kwargs)
            if isinstance(responses, AsyncIterator):
                # 通过 agent 的 a_generate_rsp 方法生成流式响应
                # responses 是一个迭代生成器，成员是 list[Message]
                async for response in responses:
                    logger.info(f"nuwa::a_genrate_rsp  ------response:{response}")
                    # 遍历每个list[Message]
                    for message in response:

                        #-----------------------------------------------------------------
                        #  这部分处理就是根据响应的结果来判断是否需要切换agent
                        #       切换agent后可以直接更换大模型
                        #-----------------------------------------------------------------
                        if (message.content == '' 
                            and isinstance(message.extra, dict) 
                            and "switch_mode_to" in message.extra):
                            logger.info(f"nuwa::a_generate_rsp --- Switching to {message.extra['switch_mode_to']}")
                            #切换 到对应的agent上去
                            self.switch_agent(name = message.extra['switch_mode_to'])
                        
                    # 转发消息
                    yield response  # 返回给调用方，支持流式输出
            else:
                logger.info(f"BaseDigitalMan::a_genrate_rsp  ------type responses:{type(response)}")
        except Exception as e:
            logger.exception(f"调用 {current_agent_name} 的 a_generate_rsp 方法失败: {str(e)}")
            yield []