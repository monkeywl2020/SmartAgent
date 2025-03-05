import sys, os
import time
from pathlib import Path
from typing import List,Tuple,Any,Union

from llama_index.core.schema import ImageNode, NodeWithScore, MetadataMode
from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding

from qwen_agent.agents import Assistant
from loguru import logger

from .knowledge_bank import KnowledgeBank
from .knowledge_llama_index import LlamaIndexKnowledge

# 获取当前目录
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
sys.path.append(os.path.join(current_path, '..'))

current_path = Path(__file__).resolve()
#picture_knowledge_path = str(current_path.parent / "image_knowledge")
nuwa_knowl_path = str(current_path.parent / "nuwa_knowledge")

# UPLOAD_ROOT_PATH = current_path
#logger.info(f"picture_knowledge_path:{picture_knowl_path}")
logger.info(f"knowledge_path:{nuwa_knowl_path}")

knowledge_id = "emotional_mutiModel_rag"

#知识库bank配置
knowledge_config =  [
{
    "knowledge_id": knowledge_id,
    "data_processing": [
        {
            "load_data": {
                "loader": {
                    "create_object": "true",
                    "module": "llama_index.core",
                    "class": "wlSimplePdfDirectoryReader",
                    "init_args": {
                        "input_dir": f"{nuwa_knowl_path}",
                        "output_dir":"./nuwaKnowledgeRag",
                        "savePdfToPng": False,
                        "page_overlap": 30,
                        "required_exts": [
                            ".txt",".jpg",".png"
                        ]
                    }
                }
            }
        }
    ]
},
]


embedding_model = wlMultiModalEmbedding(
    model_name = '/models/bge-m3',
    model_weight = '/models/bge-visualized/Visualized_m3.pth'
)

# 这个是用来回答用户问的大模型配置
llm_cfg = {
    'model': '/data2/models/Qwen2.5-72B-Instruct-AWQ',
    'model_server': 'http://172.21.30.230:8980/v1',  
    'api_key': 'EMPTY',

    # (Optional) LLM hyperparameters for generation:
    'generate_cfg': {
        'top_p': 0.8,
        'temperature':0.1,
    }
}

Refine_system_instruction = '''你是一位专业的信息提取专家。你的任务是根据用户输入的最新信息，判断其与以下关注信息的相关性，并生成简洁、准确的搜索查询。如果输入信息与关注信息无关，则返回用户原有的提问。

- **关注信息**：
    - 健成星云公司介绍，公司动态，使命愿景和价值观
    - 小星的形象认知
    - 星云星空APP使用手册，包括APP下载，安装，注册和使用流程
    - 星云星空优势介绍
    - 星云星空免费服务介绍
    - 咨询师入驻要求和流程
    - 星云星空平台介绍，平台的作用
    - 星云星空招商加盟介绍，涵盖个人和机构
    - 星云星空服务套餐介绍，用户注册，会员类型和权益，各类服务收费标准
    - 星云星空机构入驻流程介绍
    - 星云星空机构入驻要求介绍

### 优化后的搜索查询生成规则：
1. **结合上下文**：考虑用户最新输入及其上下文信息，判断是否涉及关注信息中的某一项。
2. **提取关键信息**：从最新的用户输入中提炼出核心信息，避免无关内容。
3. **生成简洁查询**：将提炼后的核心信息转化为简洁的搜索查询，用于快速获取相关答案。
4. **无关情形**：如果用户的提问与上述关注信息无关，则返回用户原有的提问。'''

#llm_client = get_llm_client(SupportedLLM.LLM_QIANWEN_VL.value)

class LocalMultimodalRAG:
    def __init__(self, 
        base_knowledge_config: Union[list, str], 
        similarity_top_k:int = 5, 
        document_path:str ="."
    ) -> None:
        logger.info("LocalMultimodalRAG::__init__============KnowledgeBank restore:")
        self.knowledge_bank = None  # 知识库实例
        self.rag_knowledge = None  # 检索知识实例
        self.similarity_top_k = similarity_top_k # 检索返回的结果数量
        self.documents_folder_path = document_path  #检索文件的路径
        self.knowledge_id = base_knowledge_config[0]["knowledge_id"] # 获取配置中 知识id

        # 配置知识库
        # 根据传入的基础配置，生成 knowledge_config
        self.base_knowledge_config = base_knowledge_config
        
        # 这个是本地知识库的持久化路径，正常情况下默认只会初始化 index化一次，后续都是直接读取文件夹中的index
        persist_folder_path = os.path.join(current_path.parent, "local_rag_index")
        os.makedirs(persist_folder_path, exist_ok=True)  # 创建 local_rag_index 目录，如果没有的话
        self.persist_folder_path = persist_folder_path # 保存持久化路径

        # 记录访问大模型的初始时间
        wlstartTime = time.time()
        # 创建知识库，用于存储 LlamaIndexKnowledge 对象的容器，根据配置初始化知识库
        wl_knowledge_bank = KnowledgeBank(configs = base_knowledge_config,
                                        emb_model = embedding_model,
                                        persist_root = persist_folder_path)
        self.knowledge_bank = wl_knowledge_bank # 将实例保存到knowledge_bank中去
        wlEndTime = time.time()
        restoreTimeCost = wlEndTime - wlstartTime

        logger.info("LocalMultimodalRAG::__init__============KnowledgeBank restore:")
        logger.info(f"LocalMultimodalRAG::__init__ 花费了实际是(s):{restoreTimeCost}")
        logger.info("LocalMultimodalRAG::__init__============KnowledgeBank restore:")

        #提炼用户提问agent
        self.Refine_questions_agent = Assistant(llm=llm_cfg,
                        system_message=Refine_system_instruction,
                        )
    # 获取知识库
    def get_rag_knowledge_bank(self):
        return self.knowledge_bank

    # 根据 知识的id获取知识库中的知识实例
    def get_rag_knowledge(self, knowledge_id: str) -> LlamaIndexKnowledge:
        return self.knowledge_bank.get_knowledge(knowledge_id,True)

    # 异步查询，获取知识库中相关的文本信息
    def query_text(self, knowledge_id: str, query: str, similarity_top_k: int = 5) -> List[NodeWithScore]:
        """文本查询，查询指定知识库"""
        knowledge = self.get_rag_knowledge(knowledge_id)
        if not knowledge:
            logger.error("LocalMultimodalRAG::query_text============knowledge null!")
            return "Knowledge not found."

        # 使用 retrieve 方法进行文本查询
        wlstartTime = time.time()
        # 使用知识库进行检索，返回相似度最高的前 similarity_top_k 个节点
        nodes_with_score = knowledge.retrieve(query = query,similarity_top_k = similarity_top_k)
        wlEndTime = time.time()
        TimeCost = wlEndTime - wlstartTime

        logger.info("LocalMultimodalRAG::query_text============KnowledgeBank restore:")
        logger.info(f"LocalMultimodalRAG::query_text 花费了实际是(s):{TimeCost}")
        logger.info("LocalMultimodalRAG::query_text============KnowledgeBank restore:")

        return nodes_with_score

    # 异步查询，获取知识库中相关的文本信息
    def query_image_text(self, knowledge_id: str, query: str, query_image_path: str, similarity_top_k: int = 5) -> List[NodeWithScore]:
        """文本查询，查询指定知识库"""
        knowledge = self.get_rag_knowledge(knowledge_id)
        if not knowledge:
            logger.error("LocalMultimodalRAG::query_image_text============knowledge null!")
            return "Knowledge not found."

        # 使用 retrieve 方法进行文本查询
        wlstartTime = time.time()
        # 使用知识库进行检索，返回相似度最高的前 similarity_top_k 个节点
        nodes_with_score = knowledge.mutimodel_retrieve(query = query, query_image_path = query_image_path, similarity_top_k = similarity_top_k)
        wlEndTime = time.time()
        TimeCost = wlEndTime - wlstartTime

        logger.info("LocalMultimodalRAG::query_image_text============KnowledgeBank restore:")
        logger.info(f"LocalMultimodalRAG::query_image_text 花费了实际是(s):{TimeCost}")
        logger.info("LocalMultimodalRAG::query_image_text============KnowledgeBank restore:")

        return nodes_with_score
    
    # 从 NodeWithScore 获取 图像和文本节点
    def _get_image_and_text_nodes(
        nodes: List[NodeWithScore],
    ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
        image_nodes = []
        text_nodes = []
        for res_node in nodes:
            if isinstance(res_node.node, ImageNode):
                image_nodes.append(res_node)
            else:
                text_nodes.append(res_node)
        return image_nodes, text_nodes
    
    def Refine_user_query(self, question: str, history:str = None)-> str:
        logger.info(f"LocalMultimodalRAG::Refine_user_query: ============用户查询 question:{question}")
        user_input_formatted_message = {'role': 'user', 'content': question}

        rsps = self.Refine_questions_agent.run(messages=[user_input_formatted_message])
        for res in rsps:
            #logger.info(f"====>res:\n {res}")
            new_qury = res
        
        new_qury_content = new_qury[0]["content"]
        logger.info(f"LocalMultimodalRAG::Refine_user_query: ============提炼后的用户查询 new_qury:{new_qury_content}")
        return new_qury_content
        

    # 根据查询的内容 获取召回的知识库知识，返回值是 二元组， (分数，召回的内容)
    def query_knowledge(self, question: str, query_image_path:str = None, similarity_top_k: int=2, score_threshold: float = 0.5) -> str:
        logger.info("LocalMultimodalRAG::query_knowledeg -------begin!")
        knowledge = self.get_rag_knowledge(knowledge_id)

        wlstartTime = time.time()
        nodes_with_score = knowledge.mutimodel_retrieve(
            query=question, 
            query_image_path=query_image_path, 
            similarity_top_k=similarity_top_k
        )
        wlEndTime = time.time()
        queryTimeCost = wlEndTime - wlstartTime
        logger.info("============wl_pdf_rag_knowledge retrieve:")
        logger.info(f"花费了实际是(s):{queryTimeCost}")
        logger.info("============wl_pdf_rag_knowledge retrieve:")
        #logger.info("========================>nodes_with_score:",nodes_with_score)

        result = []
        for idx, node_with_score in enumerate(nodes_with_score):
            score = node_with_score.get_score()
            if score >= score_threshold:
                text_content = node_with_score.get_text()
                result.append(f"知识{idx + 1}:\n- 得分: {score}\n- 内容: {text_content}\n")

        # 将所有格式化的内容合并为一条文本
        formatted_result = "\n".join(result)
        return formatted_result

    def nuwa_getRagknowledge(self, query: str, similarity_top_k: int=2, score_threshold: float = 0.5):
        # 将用户的查询格式进行分析并提炼，如果与本地知识库相关就会修改 用户查询语句。否则返回原有的用户查询语句
        new_query = nuwa_local_rag.Refine_user_query(question = query)
        knowledge_info = nuwa_local_rag.query_knowledge(question=new_query, similarity_top_k=similarity_top_k,score_threshold = score_threshold)
        logger.info(f"nuwa_getRagknowledge============ retrieve:{knowledge_info}")
        return knowledge_info


# 创建 LocalMultimodalRAG 实例并初始化
nuwa_local_rag = LocalMultimodalRAG(
    base_knowledge_config=knowledge_config, 
    similarity_top_k=5,
    document_path=nuwa_knowl_path
)

if __name__ == "__main__":

   
    print("========================================")
    print("请输入查询内容 (输入'quit'结束)：")

    while True:
        query = input("请输入查询：")
        if query.lower() == "quit":
            print("程序结束！")
            break
        
        new_query = nuwa_local_rag.Refine_user_query(question = query)
        print(f"提炼后的查询：\n{new_query}")
        # 查询文本和图像信息
        response = nuwa_local_rag.query_knowledge(question=new_query, similarity_top_k=3)
        
        # 输出查询结果
        if response:
            print(f"查询结果：\n{response}")
        else:
            print("没有找到相关内容，请重新尝试！")
        