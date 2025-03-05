from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Union,AsyncIterator
from loguru import logger

from .base_agent import get_chat_agent_cls
from .operator import Operator
from ..msg.message import Message

#这是一个全局字典，用于存储注册的 digitalman 类，类的名称作为键，类的实例作为值。 DIGITALMAN_REGISTRY 用来保存已注册的所有 digitalman
DIGITALMAN_REGISTRY = {}

#注册 digitalman ，修饰符
def register_digitalman(role_type):
    #agent不会太多，所以不用管冲突问题
    def decorator(cls):
        DIGITALMAN_REGISTRY[role_type] = cls
        return cls

    return decorator


#这个基类，所有的 chat 类的 llm 适配的基类
class BaseDigitalMan(Operator):
    """The base class of LLM"""

    #初始化
    def __init__(
        self,
        name: str,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        description: str = None,
        **kwargs: Any,
    ):
        logger.info(f"BaseDigitalMan::__init__  ------begin!")
        self.name = name                #数字人名
        self.llm_config = llm_config    #llm的配置
        self.description = description  #数字说明

        # 获取工作模式
        self.workmode = 'switch'  # 默认值
        if llm_config is not False and isinstance(llm_config, dict):
            self.workmode = llm_config.get('workmode', 'switch')
        logger.info(f"Workmode set to: {self.workmode}")

        # 验证工作模式
        valid_workmodes = {'parallel', 'preprocess', 'switch'} # 只能这3个中，选择没有默认就是 switch 模式
        if self.workmode not in valid_workmodes:
            logger.warning(f"Invalid workmode '{self.workmode}', defaulting to 'switch'")
            self.workmode = 'switch'

        # 初始化 Agents 字典
        self.agents = {}

        # 初始化 Agents，根据配置挂载 agent
        if llm_config is not False and isinstance(llm_config, dict) and 'agentcfg' in llm_config:
            for agent_config in llm_config['agentcfg']:
                # agent_config 是配置中的 agentcfg 键值内容是一个list中的成员
                self._init_agent_from_config(agent_config)

        # 将 kwargs 中的所有参数转换为实例属性
        for key, value in kwargs.items():
            setattr(self, key, value)  # 无条件覆盖
        #==============================
        #   数据库 客户端 mysql ，redis
        #   需要根据用户的信息来读取这个用户的聊天记录，记忆等等内容。
        #   记忆：用户的对话历史记录，肯定是要按会话来的，每个会话都是 用户与大模型的聊天记录。 打算用纯文本来存储
        #   继承的类来做下面的事情
        #==============================

    # 根据配置初始化agent
    def _init_agent_from_config(self, agent_config: Dict[str, Any]) -> None:
        """从配置中初始化 Agent 并添加到 self.agents"""
        try:
            agent_type = agent_config.get('agent_type')
            agent_name = agent_config.get('agent_name')
            sys_prompt = agent_config.get('sys_prompt', f"You are {agent_name}" if agent_name else "You are an assistant")
            tools_list = agent_config.get('tools_list', [])
            description = agent_config.get('description', '')
            llmcfg = agent_config.get('llmcfg')  # 是获取 agentcfg 里面的 llmcfg 键值内容，这里面放的是 llmcfg
            
            logger.info(f"Initialized agent: {agent_name} with tools_list {tools_list}")

            if not agent_type or not agent_name or not llmcfg:
                logger.warning(f"Invalid config for {agent_name or 'unknown'}: missing required fields, skipping.")
                return
            
            if agent_name in self.agents:
                logger.warning(f"Agent '{agent_name}' already exists, overwriting.")
            
            logger.info(f"Initialized agent: agent_type:{agent_type} \n with llmcfg: {llmcfg}")
            # 获取agent类，进行实例化
            agent_cls = get_chat_agent_cls(agent_type)
            # agent 实例化
            agent_instance = agent_cls(
                name=agent_name,
                sys_prompt=sys_prompt,
                llm_config=llmcfg, # agent 初始化传入的是  llmcfg，模型客户端的配置
                tools_list = tools_list,
                description=description
            )
            #将agent加入到 agents的 dict中去
            self.agents[agent_name] = agent_instance
            logger.info(f"Initialized agent: {agent_name} with type {agent_type}")
        
        except KeyError as e:
            logger.error(f"Failed to initialize agent: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing agent: {e}")

    #------------------------------------------------
    # 异步接口，生成响应
    #   接收来着用户的消息，并且给用户回响应
    #   注意：messages  这个参数是来着用户侧的，目前默认是dict格式
    #------------------------------------------------
    async def a_generate_rsp(
            self,
            messages: List[Union[Message, Dict]],
            **kwargs: Any,
        ) -> Union[List[Message], List[Dict], AsyncIterator[List[Message]], AsyncIterator[List[Dict]]]:
        # 这里面会调用 _generate_llm_reply 从大模型获取响应，并且将结果返回给调用者，这个 调用者应该是 agent。
        raise NotImplementedError

   
    #------------------------------------------------
    # 默认使用同步接口编排
    #   接收来着用户的消息，并且给用户回响应
    #------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        rsp = self.generate_rsp(*args, **kwargs)
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

    # 将迭代器中的消息转成对应的格式 message 或者 dict
    def _convert_messages_iterator_to_target_type(
            self, messages_iter: Iterator[List[Message]],
            target_type: str) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)

