
# -*- coding: utf-8 -*-
"""
This module is an integration of the Llama index RAG
into AgentScope package
"""

import os.path
import re
import json
from tqdm import tqdm  # 进度条工具
from typing import Any, Optional, List, Union
from loguru import logger
from pydantic import BaseModel, Field

try:
    import llama_index
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.ingestion import IngestionPipeline

    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import (
        VectorStoreIndex,
        StorageContext,
        load_index_from_storage,
    )
    from llama_index.core.schema import (
        Document,
        TransformComponent,
    )
    from llama_index.core.query_engine.multi_modal import SimpleMultiModalQueryEngine
    from llama_index.core.schema import ImageNode, TextNode
    from llama_index.core.schema import QueryType,QueryBundle
    from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding
    from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
    
except ImportError:
    llama_index = None
    BaseRetriever = None
    BaseEmbedding = None
    Embedding = None
    IngestionPipeline = None
    SentenceSplitter = None
    VectorStoreIndex = None
    StorageContext = None
    load_index_from_storage = None
    PrivateAttr = None
    Document = None
    TransformComponent = None

from .knowledge import Knowledge

import llama_index
#新增的自定义 Embedding和Index
from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding
from llama_index.core.indices.multi_modal.wl_multi_vectorstoreindex import wlMultiModalVectorStoreIndex

# 文档切块配置
DEFAULT_CHUNK_SIZE = 1224
DEFAULT_CHUNK_OVERLAP = 20 # 块直接的重叠部分
#查找默认返回条数
DEFAULT_TOP_K = 5

class LlamaIndexKnowledge(Knowledge):
    """
    This class is a wrapper with the llama index RAG.
    """

    def __init__(
        self,
        knowledge_id: str,
        emb_model: Optional[wlMultiModalEmbedding] = None,
        knowledge_config: Optional[dict] = None,
        model: Any = None,
        persist_root: Optional[str] = None,
        overwrite_index: Optional[bool] = False,
        showprogress: Optional[bool] = True,
        **kwargs: Any,
    ) -> None:
        """
        initialize the knowledge component based on the
        llama-index framework: https://github.com/run-llama/llama_index

        Notes:
            In LlamaIndex, one of the most important concepts is index,
            which is a data structure composed of Document objects, designed to
            enable querying by an LLM. The core workflow of initializing RAG is
            to convert data to index, and retrieve information from index.
            For example:
            1) preprocessing documents with data loaders
            2) generate embedding by configuring pipline with embedding models
            3) store the embedding-content to vector database
                the default dir is "./rag_storage/knowledge_id"

        Args:
            knowledge_id (str):
                The id of the RAG knowledge unit.
            emb_model (ModelWrapperBase):
                The embedding model used for generate embeddings
            knowledge_config (dict):
                The configuration for llama-index to
                generate or load the index.
            model (ModelWrapperBase):
                The language model used for final synthesis
            persist_root (str):
                The root directory for index persisting
            overwrite_index (Optional[bool]):
                Whether to overwrite the index while refreshing
            showprogress (Optional[bool]):
                Whether to show the indexing progress
        """
        super().__init__(
            knowledge_id=knowledge_id,
            emb_model=emb_model,
            knowledge_config=knowledge_config,
            model=model,
            **kwargs,
        )
        if llama_index is None:
            raise ImportError(
                "LlamaIndexKnowledge require llama-index installed. "
                "Try a stable llama-index version, such as "
                "`pip install llama-index==0.10.30`",
            )

        # 以当前目录为根目录
        if persist_root is None:
            persist_root = "./"
        self.persist_dir = os.path.join(persist_root, knowledge_id) # 将传入的目录作为根目录，传入的id作为文件夹建立新的 根目录
        self.emb_model = emb_model
        self.overwrite_index = overwrite_index
        self.showprogress = showprogress
        # index是用来存放索引结构的。这个索引是 llama-index的关键结构，索引 wlMultiModalVectorStoreIndex
        self.index = None

        # 多模态知识库，中图片部分需要一个 图片的描述文件
        # 获取 img 描述文件信息，如果没有图片就不需要添加描述。有就需要给图片添加提示，传入的这个参数就是目标目录下的图片描述文件
        #self.img_description = kwargs.get('img_description',None)

        # then we can initialize the RAG
        self._init_rag()

    def _init_rag(self, **kwargs: Any) -> None:
        """
        Initialize the RAG. This includes:
            * if the persist_dir exists, load the persisted index
            * if not, convert the data to index
            * if needed, update the index
            * set the retriever to retrieve information from index

        Notes:
            * the index is persisted in the self.persist_dir
            * the refresh_index method is placed here for testing, it can be
                called externally. For example, updated the index periodically
                by calling rag.refresh_index() during the execution of the
                agent.
        """
        # 从目录中恢复index
        if os.path.exists(self.persist_dir):
            self._load_index()
            # self.refresh_index()
        else:
            # 从文档中创建index
            self._data_to_index()
        self._get_retriever()
        logger.info(
            f"RAG with knowledge ids: {self.knowledge_id} "
            f"initialization completed!\n",
        )

    def _load_index(self) -> None:
        """
        Load the persisted index from persist_dir.
        """
        # load the storage_context
        storage_context = StorageContext.from_defaults(
            persist_dir=self.persist_dir,
        )
        # construct index from
        self.index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=self.emb_model,
            image_embed_model = self.emb_model,
        )
        logger.info(f"index loaded from {self.persist_dir}")

    def _data_to_index(self) -> None:
        """
        Convert the data to index by configs. This includes:
            * load the data to documents by using information from configs
            * set the transformations associated with documents
            * convert the documents to nodes
            * convert the nodes to index

        Notes:
            As each selected file type may need to use a different loader
            and transformations, knowledge_config is a list of configs.
        """
        nodes = []
        # load data to documents and set transformations
        # using information in knowledge_config
        # 下面是遍历 data_processing 配置中所有用来 loader解析文件的配置，同时根据 分割 内容的配置，将所有文档解析成 nodes。然后将所有节点汇聚起来。
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            nodes_docs = self._docs_to_nodes(
                documents=documents,
                transformations=transformations,
            )
            nodes = nodes + nodes_docs
                
        # convert nodes to index, 将所有的nodes进行index化处理。
        self.index = wlMultiModalVectorStoreIndex(
            nodes=nodes,
            embed_model=self.emb_model,
            image_embed_model = self.emb_model,
        )
        logger.info("index calculation completed.")
        # persist the calculated index 持久化 这个 index的内容，可以作为后续快速启动时候的处理。
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("index persisted.")

    def _data_to_docs(
        self,
        query: Optional[str] = None,
        config: dict = None,
    ) -> Any:
        """
        This method set the loader as needed, or just use the default setting.
        Then use the loader to load data from dir to documents.

        Notes:
            We can use simple directory loader (SimpleDirectoryReader)
            to load general documents, including Markdown, PDFs,
            Word documents, PowerPoint decks, images, audio and video.
            Or use SQL loader (DatabaseReader) to load database.

        Args:
            query (Optional[str]):
                optional, used when the data is in a database.
            config (dict):
                optional, used when the loader config is in a config file.
        Returns:
            Any: loaded documents
        """
        loader = self._set_loader(config=config).get("loader")
        # let the doc_id be the filename for each document
        loader.filename_as_id = True
        if query is None:
            documents = loader.load_data(persist_root = self.persist_dir) # 新增了持久化路径
        else:
            # this is for querying a database,
            # does not work for loading a document directory
            documents = loader.load_data(query,persist_root = self.persist_dir) # 新增了持久化路径
        logger.info(f"loaded {len(documents)} documents")
        print(f"loaded {len(documents)} documents")
        return documents

    def _docs_to_nodes(
        self,
        documents: List[Document],
        transformations: Optional[list[Optional[TransformComponent]]] = None,
    ) -> Any:
        """
        Convert the loaded documents to nodes using transformations.

        Args:
            documents (List[Document]):
                documents to be processed, usually expected to be in
                 llama index Documents.
            transformations (Optional[list[TransformComponent]]):
                optional, specifies the transformations (operators) to
                process documents (e.g., split the documents into smaller
                chunks)
        Return:
            Any: return the index of the processed document
        """
        # nodes, or called chunks, is a presentation of the documents
        # we build nodes by using the IngestionPipeline
        # for each document with corresponding transformations
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # stack up the nodes from the pipline
        nodes = pipeline.run(
            documents=documents,
            show_progress=self.showprogress,
        )
        logger.info("nodes generated.")
        return nodes

    def _set_loader(self, config: dict) -> Any:
        """
        Set the loader as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations
        """
        if "load_data" in config:
            # we prepare the loader from the configs
            loader = self._prepare_args_from_config(
                config=config.get("load_data", {}),
            )
        else:
            # we prepare the loader by default
            try:
                from llama_index.core import SimpleDirectoryReader
            except ImportError as exc_inner:
                raise ImportError(
                    " LlamaIndexAgent requires llama-index to be install."
                    "Please run `pip install llama-index`",
                ) from exc_inner
            loader = {
                "loader": SimpleDirectoryReader(
                    input_dir="set_default_data_path",
                ),
            }
        logger.info(f"loaders are ready.loader:{loader}")
        return loader

    def _set_transformations(self, config: dict) -> Any:
        """
        Set the transformations as needed, or just use the default setting.

        Args:
            config (dict): a dictionary containing configurations.

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
        if "store_and_index" in config:
            temp = self._prepare_args_from_config(
                config=config.get("store_and_index", {}),
            )
            transformations = temp.get("transformations")
        else:
            #----------------------------------------------------------------------
            # 获取 input_dir 的值,这里面可以获取 image 的描述内容，如果没有就不需要添加这些内容
            data_processing = self.knowledge_config.get("data_processing", [])
            if data_processing:
                load_data = data_processing[0].get("load_data", {})
                image_desc_file = load_data.get("img_desc", None)
                logger.info(f"image_desc_file:{image_desc_file}")
                
                # 这个图片描述的关键格式化格式字符串模版，可配置的
                img_desc_format_str = load_data.get("img_desc_format_str", {})
            else:
                logger.info("data_processing 配置不存在")
                image_desc_file = None
            #----------------------------------------------------------------------
            # 如果没有图片的文件描述则不处理
            if image_desc_file is None:
                transformations = [
                    SentenceSplitter(
                        chunk_size=self.knowledge_config.get(
                            "chunk_size",
                            DEFAULT_CHUNK_SIZE,
                        ),
                        chunk_overlap=self.knowledge_config.get(
                            "chunk_overlap",
                            DEFAULT_CHUNK_OVERLAP,
                        ),
                    ),
                ]
            else:
                # 如果有图片的文件描述则将描述信息和图片信息合并
                transformations = [
                    SentenceSplitter(
                        chunk_size=self.knowledge_config.get(
                            "chunk_size",
                            DEFAULT_CHUNK_SIZE,
                        ),
                        chunk_overlap=self.knowledge_config.get(
                            "chunk_overlap",
                            DEFAULT_CHUNK_OVERLAP,
                        ),
                    ),
                    
                    #这个transformation中添加图片的描述
                    ImageDescriptionTransformer(
                        imgdesc_dir=image_desc_file,
                        text_format_str = img_desc_format_str
                    ),
                ]
             
        # adding embedding model as the last step of transformation
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html
        #最后做 Embedding
        transformations.append(self.emb_model)
        logger.info("transformations are ready.")
        # as the last step, we need to repackage the transformations in dict
        transformations = {"transformations": transformations}
        return transformations

    def _get_retriever(
        self,
        similarity_top_k: int = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Set the retriever as needed, or just use the default setting.

        Args:
            retriever (Optional[BaseRetriever]): passing a retriever in
             LlamaIndexKnowledge
            rag_config (dict): rag configuration, including similarity top k
            index.
        """
        # set the retriever
        logger.info(
            f"similarity_top_k" f"={similarity_top_k or DEFAULT_TOP_K}",
        )
        retriever = self.index.as_retriever(
            embed_model=self.emb_model,
            similarity_top_k=similarity_top_k or DEFAULT_TOP_K,
            **kwargs,
        )
        logger.info("retriever is ready.")
        return retriever
    
    #新增一个 多模态retrieve查询
    def mutimodel_retrieve(
        self,
        query: str,
        query_image_path:str,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> list[Any]:
        '''
        多模态的查询是指，用户可以同时输入文本和图片，然后查询与输入文本相关的图片。
        emb_model 是用来将图片和文本转换为嵌入的模型。
        '''
        # 如果查询和查询图片路径都为 None，则返回空列表
        if query is None and query_image_path is None:
            logging.error("Both query and query_image_path are None. Please provide at least one input.")
            return []
        
        if retriever is None:
            retriever = self._get_retriever(similarity_top_k)

        #print(f"============>query{query} query_image_path:{query_image_path}")
        #先用多模态大模型将 查询文本和 图片 转换成embedding 
        query_text_and_image_embedding = self.emb_model._get_query_embedding_with_img(query, query_image_path)
        # 创建一个 QueryBundle 对象，用于存储查询的 Embedding
        query_Embedding = QueryBundle(query_str="",embedding=query_text_and_image_embedding)

        retrieved = retriever.retrieve(query_Embedding)
        if to_list_strs:
            results = []
            for node in retrieved:
                results.append(node.get_text())
            return results
        
        #print("============>retrieved:\n",retrieved)
        return retrieved
    
    def retrieve(
        self,
        query: QueryType,
        similarity_top_k: int = None,
        to_list_strs: bool = False,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        This is a basic retrieve function for knowledge.
        It will build a retriever on the fly and return the
        result of the query.
        Args:
            query (str):
                query is expected to be a question in string
            similarity_top_k (int):
                the number of most similar data returned by the
                retriever.
            to_list_strs (bool):
                whether returns the list of strings;
                if False, return NodeWithScore
            retriever (BaseRetriever):
                for advanced usage, user can pass their own retriever.
        Return:
            list[Any]: list of str or NodeWithScore

        More advanced query processing can refer to
        https://docs.llamaindex.ai/en/stable/examples/query_transformations/query_transform_cookbook.html
        """
        if retriever is None:
            retriever = self._get_retriever(similarity_top_k)
        retrieved = retriever.retrieve(query)
        if to_list_strs:
            results = []
            for node in retrieved:
                results.append(node.get_text())
            return results
        
        #print("============>retrieved:\n",retrieved)
        return retrieved

    def refresh_index(self) -> None:
        """
        Refresh the index when needed.
        """
        for config in self.knowledge_config.get("data_processing"):
            documents = self._data_to_docs(config=config)
            # store and indexing for each file type
            transformations = self._set_transformations(config=config).get(
                "transformations",
            )
            self._insert_docs_to_index(
                documents=documents,
                transformations=transformations,
            )

    def _insert_docs_to_index(
        self,
        documents: List[Document],
        transformations: TransformComponent,
    ) -> None:
        """
        Add documents to the index. Given a list of documents, we first test if
        the doc_id is already in the index. If not, we add the doc to the
        list. If yes, and the over-write flag is enabled,
        we delete the old doc and add the new doc to the list.
        Lastly, we generate nodes for all documents on the list, and insert
        the nodes to the index.

        Args:
            documents (List[Document]): list of documents to be added.
            transformations (TransformComponent): transformations that
            convert the documents into nodes.
        """
        # this is the pipline that generate the nodes
        pipeline = IngestionPipeline(
            transformations=transformations,
        )
        # we need to generate nodes from this list of documents
        insert_docs_list = []
        for doc in documents:
            if doc.doc_id not in self.index.ref_doc_info.keys():
                # if the doc_id is not in the index, we add it to the list
                insert_docs_list.append(doc)
                logger.info(
                    f"add new documents to index, " f"doc_id={doc.doc_id}",
                )
            else:
                if self.overwrite_index:
                    # if we enable overwrite index, we delete the old doc
                    self.index.delete_ref_doc(
                        ref_doc_id=doc.doc_id,
                        delete_from_docstore=True,
                    )
                    # then add the same doc to the list
                    insert_docs_list.append(doc)
                    logger.info(
                        f"replace document in index, " f"doc_id={doc.doc_id}",
                    )
        logger.info("documents scan completed.")
        # we generate nodes for documents on the list
        nodes = pipeline.run(
            documents=insert_docs_list,
            show_progress=True,
        )
        logger.info("nodes generated.")
        # insert the new nodes to index
        self.index.insert_nodes(nodes=nodes)
        logger.info("nodes inserted to index.")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def _delete_docs_from_index(
        self,
        documents: List[Document],
    ) -> None:
        """
        Delete the nodes that are associated with a list of documents.

        Args:
            documents (List[Document]): list of documents to be deleted.
        """
        doc_id_list = [doc.doc_id for doc in documents]
        for key in self.index.ref_doc_info.keys():
            if key in doc_id_list:
                self.index.delete_ref_doc(
                    ref_doc_id=key,
                    delete_from_docstore=True,
                )
                logger.info(f"docs deleted from index, doc_id={key}")
        # persist the updated index
        self.index.storage_context.persist(persist_dir=self.persist_dir)
        logger.info("nodes delete completed.")

    #获取多模态查询engine
    def _get_query_engine(
        self,
        template: str,
        **kwargs: Any,
    ) -> SimpleMultiModalQueryEngine:
        """
        Set the retriever as needed, or just use the default setting.

        Args:
            retriever (Optional[BaseRetriever]): passing a retriever in
             LlamaIndexKnowledge
            rag_config (dict): rag configuration, including similarity top k
            index.
        """
        # 打印 template,如果没有传入模版就用默认的模版
        logger.info(
            f"template is {template}",
        )

        # 传入的参数就是模版，多模态大模型用初始化 knowledgebank的时候传入的多模态大模型
        query_engine = self.index.as_query_engine(
            llm = self.postprocessing_model,
            text_qa_template = template,
            **kwargs,
        )
        logger.info("query_engine is ready.")
        return query_engine

    #资料 查询,这个查询会访问多模态大模型，并返回大模型返回的结果
    def simple_query(
        self,
        query: Optional[str],
        query_image_path:Optional[str],
        similarity_top_k: int = None,
        query_template: str = None,
        **kwargs: Any,
    ) -> list[Any]:
        '''
        多模态的查询是指，用户可以同时输入文本和图片，然后查询与输入文本相关的图片。
        emb_model 是用来将图片和文本转换为嵌入的模型。
        '''
        # 如果查询和查询图片路径都为 None，则返回空列表
        if query is None and query_image_path is None:
            logger.error("Both query and query_image_path are None. Please provide at least one input.")
            print("query and query_image_path 不能同时为空")
            return []
        
        if query_template is None:
            logger.error("query_template can not None! we need template to output rst!")
            print("query_template 不能同时为空")
            return []
        
        retriever = self._get_retriever(similarity_top_k)
        #print(f"============>query{query} query_image_path:{query_image_path}")
        #先用多模态大模型将 查询文本和 图片 转换成embedding 
        query_text_and_image_embedding = self.emb_model._get_query_embedding_with_img(query, query_image_path)
        # 创建一个 QueryBundle 对象，用于存储查询的 Embedding
        query_Embedding = QueryBundle(query_str="",embedding=query_text_and_image_embedding)
        
        # retrieve 返回的节点
        nodes = retriever.retrieve(query_Embedding)
        
        #获取查询引擎
        query_engine = self._get_query_engine(query_template,**kwargs)

        # 用查询引擎处理查询结果
        mutilmodel_llm_query = QueryBundle(query_str=query, image_path = query_image_path)
        response = query_engine.custom_synthesize(query_bundle = mutilmodel_llm_query,nodes= nodes,template = query_template)
        
        #print("============>retrieved:\n",retrieved)
        return response
    
    #多模态大模型 查询,这个查询会访问多模态大模型，并返回大模型返回的结果
    def mutimodel_query(
        self,
        query: Optional[str],
        query_image_path:Optional[str],
        similarity_top_k: int = None,
        query_template: str = None,
        **kwargs: Any,
    ) -> list[Any]:
        '''
        多模态的查询是指，用户可以同时输入文本和图片，然后查询与输入文本相关的图片。
        emb_model 是用来将图片和文本转换为嵌入的模型。
        '''
        # 如果查询和查询图片路径都为 None，则返回空列表
        if query is None and query_image_path is None:
            logger.error("Both query and query_image_path are None. Please provide at least one input.")
            print("query and query_image_path 不能同时为空")
            return []
        
        if query_template is None:
            logger.error("query_template can not None! we need template to output rst!")
            print("query_template 不能同时为空")
            return []
        
        retriever = self._get_retriever(similarity_top_k)
        #print(f"============>query{query} query_image_path:{query_image_path}")
        #先用多模态大模型将 查询文本和 图片 转换成embedding 
        query_text_and_image_embedding = self.emb_model._get_query_embedding_with_img(query, query_image_path)
        # 创建一个 QueryBundle 对象，用于存储查询的 Embedding
        query_Embedding = QueryBundle(query_str="",embedding=query_text_and_image_embedding)
        
        # retrieve 返回的节点
        nodes = retriever.retrieve(query_Embedding)
        
        #获取查询引擎
        query_engine = self._get_query_engine(query_template,**kwargs)

        # 用查询引擎处理查询结果
        mutilmodel_llm_query = QueryBundle(query_str=query, image_path = query_image_path)
        response = query_engine.custom_synthesize(query_bundle = mutilmodel_llm_query,nodes= nodes,template = query_template)
        
        #print("============>retrieved:\n",retrieved)
        return response

# 1. 这个llama index的transformer主要用来将文件的的描述信息添加的 图片的Embedding 中去。提高Embedding质量
class ImageDescriptionTransformer(TransformComponent):
    # 显式声明变量
    descriptions: dict[str, str] = Field(default_factory=dict) 
    text_format_str: Optional[str] = Field(default=None)

    def __init__(self, imgdesc_dir, text_format_str = None):
        # 从image目录中获取 描述文件路径，默认为
        try:
            # 尝试打开并加载 JSON 文件
            with open(imgdesc_dir, 'r') as json_file:
                image_descriptions = json.load(json_file)
                logger.info(f"Image Descriptions:{image_descriptions}")
        except FileNotFoundError:
            # 捕获文件未找到错误
            logger.error(f"Error: The file '{imgdesc_dir}' was not found.")
        except json.JSONDecodeError as e:
            # 捕获 JSON 解码错误
            logger.error(f"Error: Failed to decode JSON in '{imgdesc_dir}'.")
            logger.error(f"Details: {e}")
        except Exception as e:
            # 捕获所有其他类型的错误
            logger.error(f"An unexpected error occurred: {e}")

        #self.descriptions = image_descriptions
        #self.text_format_str = text_format_str
        super().__init__(descriptions=image_descriptions, text_format_str=text_format_str)

    def __call__(self, nodes, show_progress=False, **kwargs):
        transformed_nodes = []

        # 如果 show_progress 为 True，使用 tqdm 包装迭代器
        iterator = tqdm(nodes, desc="Processing Nodes", unit="node") if show_progress else nodes

        for node in iterator:
            # 必须是 ImageNode
            if isinstance(node, ImageNode):
                # 提取文件名
                file_name = re.search(r"([^/]+)(?=\.\w+$)", node.image_path).group(0)
                
                # 查找并添加描述
                if file_name in self.descriptions:
                    description = self.descriptions[file_name]
                    if self.text_format_str is not None:
                        # 使用正则表达式提取花括号中的变量名, 这个是从字符串中提取出来的参数名
                        variables = re.findall(r'\{(\w+)\}', self.text_format_str)
                        # 打印提取的变量名
                        #logger.info(f"ImageDescriptionTransformer::__call__ --->self.text_format_str variables:{variables}")
                        # 动态构造参数字典
                        values = {
                            variables[0]: file_name,     # 将第一个变量名映射到 file_name
                            variables[1]: description   # 将第二个变量名映射到 description
                        }

                        formatted_description = self.text_format_str.format(**values)
                    else:
                        formatted_description  = description
                    
                    #将图片的描述添加到图片的node中去，可以一起做Embedding
                    node.text = formatted_description

                # 可以添加调试信息
                logger.info(f"Processed image: {node.image_path}")  
                #print("=================node:",node)
                logger.info(f"=================node:{node}")

            transformed_nodes.append(node)

        return transformed_nodes