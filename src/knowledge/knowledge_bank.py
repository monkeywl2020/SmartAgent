# -*- coding: utf-8 -*-
"""
Knowledge bank for making Knowledge objects easier to use
"""
import copy
import json
from typing import Optional, Union, Any
from loguru import logger
from .knowledge_llama_index import LlamaIndexKnowledge

DEFAULT_INDEX_CONFIG = {
    "knowledge_id": "",
    "data_processing": [],
}
DEFAULT_LOADER_CONFIG = {
    "load_data": {
        "loader": {
            "create_object": True,
            "module": "llama_index.core",
            "class": "SimpleDirectoryReader",
            "init_args": {},
        },
    },
}
DEFAULT_INIT_CONFIG = {
    "input_dir": "",
    "recursive": True,
    "required_exts": [],
}

# 知识库，用于存储 LlamaIndexKnowledge 对象的容器
class KnowledgeBank:
    """
    KnowledgeBank enables
    1) provide an easy and fast way to initialize the Knowledge object;
    2) make Knowledge object reusable and sharable for multiple agents.
    """

    # 初始化知识库
    def __init__(
        self,
        configs: Union[list, str],
        emb_model: Any = None,
        muti_llm_model: Any = None,
        persist_root: str ="."
    ) -> None:
        """initialize the knowledge bank"""
        if isinstance(configs, str):
            logger.info(f"Loading configs from {configs}")
            with open(configs, "r", encoding="utf-8") as fp:
                self.configs = json.loads(fp.read())
        else:
            self.configs = configs

        logger.info(f"KnowledgeBank::__init__  configs:{self.configs}")
       # 检查 emb_model 这个不能为空，如果为空则抛出异常  muti_llm_model可以为空，因为不是必须的，为空就不能访问多模态模型
        if emb_model is None:
            raise ValueError("Error: emb_model cannot be None. Please provide a valid embedding model.")
        
        # 存储知识库，是一个字典，key是知识库的id，value是知识库对象
        self.stored_knowledge: dict[str, LlamaIndexKnowledge] = {}
        self.persist_root = persist_root # 保存 存储路径
        
        self._init_knowledge(emb_model, muti_llm_model) # 初始化知识库

    # 初始化知识库，将知识库挂载到知识库bank中
    def _init_knowledge(self,emb_model,muti_llm_model) -> None:
        """initialize the knowledge bank"""
        for config in self.configs: # 遍历所有的list中的配置，每个list配置都是一份 知识。知识库是多个知识组成的
            logger.info(f"_init_knowledge: bank :{config}")
            self.add_data_as_knowledge(
                knowledge_id=config["knowledge_id"],
                knowledge_config=config,
                emb_model = emb_model,
                muti_llm_model = muti_llm_model
            )
        logger.info("knowledge bank initialization completed.\n ")

    def add_data_as_knowledge(
        self,
        knowledge_id: str,
        emb_model: Any = None,
        data_dirs_and_types: dict[str, list[str]] = None,
        muti_llm_model: Any = None,
        knowledge_config: Optional[dict] = None,
    ) -> None:
        """
        Transform data in a directory to be ready to work with RAG.
        Args:
            knowledge_id (str):
                user-defined unique id for the knowledge 知识库的id是用户定义的，用来区分不同的知识库的唯一标识
            emb_model_name (str):
                name of the embedding model
            model_name (Optional[str]):
                name of the LLM for potential post-processing or query rewrite
            data_dirs_and_types (dict[str, list[str]]):
                dictionary of data paths (keys) to the data types
                (file extensions) for knowledgebase
                (e.g., [".md", ".py", ".html"])
            knowledge_config (optional[dict]):
                complete indexing configuration, used for more advanced
                applications. Users can customize
                - loader,
                - transformations,
                - ...
                Examples can refer to../examples/conversation_with_RAG_agents/

            a simple example of importing data to Knowledge object:
            ''
                knowledge_bank.add_data_as_knowledge(
                    knowledge_id="agentscope_tutorial_rag",
                    emb_model_name="qwen_emb_config",
                    data_dirs_and_types={
                        "../../docs/sphinx_doc/en/source/tutorial": [".md"],
                    },
                    persist_dir="./rag_storage/tutorial_assist",
                )
            ''
        """
        # 知识库id已经存在
        if knowledge_id in self.stored_knowledge:
            raise ValueError(f"knowledge_id {knowledge_id} already exists.")

        assert data_dirs_and_types is not None or knowledge_config is not None

        if knowledge_config is None:
            knowledge_config = copy.deepcopy(DEFAULT_INDEX_CONFIG)
            for data_dir, types in data_dirs_and_types.items():
                loader_config = copy.deepcopy(DEFAULT_LOADER_CONFIG)
                loader_init = copy.deepcopy(DEFAULT_INIT_CONFIG)
                loader_init["input_dir"] = data_dir
                loader_init["required_exts"] = types
                loader_config["load_data"]["loader"]["init_args"] = loader_init
                knowledge_config["data_processing"].append(loader_config)


        #将知识库挂载到知识库bank中
        self.stored_knowledge[knowledge_id] = LlamaIndexKnowledge(
            knowledge_id=knowledge_id,
            emb_model= emb_model,# emb_model使用的是自定义的wlMultiModalEmbedding
            knowledge_config=knowledge_config,
            model= muti_llm_model,
            persist_root = self.persist_root,
            showprogress = True,
        )
        logger.info(f"data loaded for knowledge_id = {knowledge_id}.")

    # 根据knowledge_id获取知识库对象
    def get_knowledge(
        self,
        knowledge_id: str,
        duplicate: bool = False,
    ) -> LlamaIndexKnowledge:
        """
        Get a Knowledge object from the knowledge bank.
        Args:
            knowledge_id (str):
                unique id for the Knowledge object
            duplicate (bool):
                whether return a copy of the Knowledge object.
        Returns:
            LlamaIndexKnowledge:
                the Knowledge object defined with Llama-index
        """
        if knowledge_id not in self.stored_knowledge:
            raise ValueError(
                f"{knowledge_id} does not exist in the knowledge bank.",
            )
        knowledge = self.stored_knowledge[knowledge_id]
        #if duplicate:
        #    knowledge = copy.deepcopy(knowledge)
        logger.info(f"knowledge bank loaded: {knowledge_id}.")
        return knowledge

    def equip(
        self,
        agent: Any,
        knowledge_id_list: list[str] = None,
        duplicate: bool = False,
    ) -> None:
        """
        Equip the agent with the knowledge by knowledge ids.

        Args:
            agent (AgentBase):
                the agent to be equipped with knowledge
            knowledge_id_list:
                the list of knowledge ids to be equipped with the agent
            duplicate (bool): whether to deepcopy the knowledge object
        TODO: to accommodate with distributed setting
        """
        logger.info(f"Equipping {agent.name} knowledge {knowledge_id_list}")
        knowledge_id_list = knowledge_id_list or []

        if not hasattr(agent, "knowledge_list"):
            agent.knowledge_list = []
        for kid in knowledge_id_list:
            knowledge = self.get_knowledge(
                knowledge_id=kid,
                duplicate=duplicate,
            )

            # 将知识库挂载到agent的knowledge_list中去。则就是agent挂载了多个知识库
            agent.knowledge_list.append(knowledge)
