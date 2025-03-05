# -*- coding: utf-8 -*-
"""
Base class module for retrieval augmented generation (RAG).
To accommodate the RAG process of different packages,
we abstract the RAG process into four stages:
- data loading: loading data into memory for following processing;
- data indexing and storage: document chunking, embedding generation,
and off-load the data into VDB;
- data retrieval: taking a query and return a batch of documents or
document chunks;
- post-processing of the retrieved data: use the retrieved data to
generate an answer.

检索增强生成 (RAG) 的基类模块。
为了适应不同软件包的 RAG 过程，我们将 RAG 过程抽象为四个阶段：
    - 数据加载：将数据加载到内存中以便进行后续处理；
    - 数据索引和存储：文档分块、嵌入生成，并将数据卸载到 VDB；
    - 数据检索：进行查询并返回一批文档或文档块；（数据检索之前还可以进行数据提取，将用户问题转换成与知识库相关的问题。可以大大增加RAG成功率）
    - 检索数据的后处理：使用检索到的数据生成答案。
"""
import os
import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional
from loguru import logger
#from agentscope.models import ModelWrapperBase

#from wl_custom_embeding import wlMultiModalEmbedding

class Knowledge(ABC):
    """
    Base class for RAG, CANNOT be instantiated directly
    """

    def __init__(
        self,
        knowledge_id: str,
        emb_model: Any = None,
        knowledge_config: Optional[dict] = None,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        # pylint: disable=unused-argument
        """
        initialize the knowledge component
        Args:
        knowledge_id (str):
            The id of the knowledge unit.
        emb_model (ModelWrapperBase):
            The embedding model used for generate embeddings
        knowledge_config (dict):
            The configuration to generate or load the index.

        model 交互模型多模态的，用于回答消息结果
        """
        self.knowledge_id = knowledge_id
        self.emb_model = emb_model
        self.knowledge_config = knowledge_config or {}
        self.postprocessing_model = model

    @abstractmethod
    def _init_rag(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Initiate the RAG module.
        """

    @abstractmethod
    def retrieve(
        self,
        query: Any,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        """
        查询返回记忆的内容
        retrieve list of content from database (vector stored index) to memory
        Args:
            query (Any):
                query for retrieval
            similarity_top_k (int):
                the number of most similar data returned by the
                retriever.
            to_list_strs (bool):
                whether return a list of str

        Returns:
            return a list with retrieved documents (in strings)
        """

    def post_processing(
        self,
        retrieved_docs: list[str],
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """
        这个函数是调用大模型返回结果。使用了查询回来的内容与 prompt组合到一起请求大模型。
        A default solution for post-processing function, generates answer
        based on the retrieved documents.
        
        Args:
            retrieved_docs (list[str]):
                list of retrieved documents
            prompt (str):
                prompt for LLM generating answer with the retrieved documents

        Returns:
            Any: a synthesized answer from LLM with retrieved documents

        Example:
            self.postprocessing_model(prompt.format(retrieved_docs))
        """
        assert self.postprocessing_model
        prompt = prompt.format("\n".join(retrieved_docs))
        return self.postprocessing_model(prompt, **kwargs).text

    # 这个主要是读取配置，并根据配置读取配置文件目录指定后缀的文件，返回 class 指定的对象（使用 init_args参数初始化的实例）
    def _prepare_args_from_config(self, config: dict) -> Any:
            """
            Helper function to build objects in RAG classes.

            Args:
                config (dict): a dictionary containing configurations
            Returns:
                Any: an object that is parsed/built to be an element
                    of input to the function of RAG module.
            这个函数主要是读取配置，并根据配置读取配置文件目录指定后缀的文件，返回 class 指定的对象（使用 init_args参数初始化的实例）

            这个是配置的例子：                    
                "load_data": {
                    "loader": {
                            "create_object": true,
                            "module": "llama_index.core",
                            "class": "SimpleDirectoryReader",
                            "init_args": {
                                "input_dir": "../../src/agentscope",
                                "recursive": true,
                                "required_exts": [
                                    ".py"
                                ]
                        }
                    }
                },
                "store_and_index": {
                    "transformations": [
                        {
                            "create_object": true,
                            "module": "llama_index.core.node_parser",
                            "class": "CodeSplitter",
                            "init_args": {
                                "language": "python",
                                "chunk_lines": 100
                            }
                        }
                    ]
                }                
            """
            if not isinstance(config, dict):
                return config

            if "create_object" in config:
                # if a term in args is an object,
                # recursively create object with args from config
                module_name = config.get("module", "")
                class_name = config.get("class", "")
                init_args = config.get("init_args", {})
                # 假设配置中的路径是相对路径
                input_dir = init_args.get("input_dir", "")
                absolute_path = os.path.abspath(input_dir)  # 转换为绝对路径
                init_args["input_dir"] = absolute_path

                try:
                    cur_module = importlib.import_module(module_name)
                    cur_class = getattr(cur_module, class_name)
                    init_args = self._prepare_args_from_config(init_args)
                    logger.info(
                        f"load and build object: {class_name}",
                    )
                    logger.info(f"============>init_args:{init_args}")
                    return cur_class(**init_args) # 返回 SimpleDirectoryReader("xxxx/xxx") 的实例，路径这个参数在配中
                except ImportError as exc_inner:
                    logger.error(
                        f"Fail to load class {class_name} "
                        f"from module {module_name}",
                    )
                    raise ImportError(
                        f"Fail to load class {class_name} "
                        f"from module {module_name}",
                    ) from exc_inner
            else:
                prepared_args = {}
                for key, value in config.items():
                    if isinstance(value, list):
                        prepared_args[key] = []
                        for c in value:
                            prepared_args[key].append(
                                self._prepare_args_from_config(c),
                            )
                    elif isinstance(value, dict):
                        prepared_args[key] = self._prepare_args_from_config(value)
                    else:
                        prepared_args[key] = value
                return prepared_args