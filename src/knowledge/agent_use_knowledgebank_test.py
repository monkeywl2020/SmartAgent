import json
import os
import sys
import re
import time

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
#from llama_index.llms.openai import OpenAI

from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding

import logging

#from knowledge_llama_index import LlamaIndexKnowledge
from knowledge_bank import KnowledgeBank

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from qwen_agent.llm import get_chat_model

# 设置全局日志级别为 WARNING（忽略 DEBUG 和 INFO）
#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
#)

#logging.getLogger("llama_index").setLevel(logging.INFO)
#logging.getLogger("llama-index-core").setLevel(logging.INFO)
#logging.getLogger("llama-index-readers-file").setLevel(logging.INFO)

'''
a = os.path.abspath(__file__)
print(a)
b = os.path.dirname(a)
print(b)

sys.path.append(b)
print(sys.path)

        {
          "load_data": {
            "img_desc":"../ai_challenger_caption_test_a_20180103/output/1_imgdesc.json",
            "img_desc_format_str":"{imagename}：这张图片展示了以下内容：{desc}。",
            "loader": {
              "create_object": "true",
              "module": "llama_index.core",
              "class": "SimpleDirectoryReader",
              "init_args": {
                "input_dir": "../ai_challenger_caption_test_a_20180103/output/1",
                "required_exts": [
                  ".jpg"
                ]
              }
            }
          }
        },

'''
if 1:
  # 知识库knowledge_config 配置参说明：
  ''' 
      input_dir  参数:  解析为知识库的文档路径 下面例子为 /home/wl/wlwork/llama_index_mutimodeRag/xyxk
      output_dir 参数:  输出的路径，会输出png截图还有pdf解析后的txt文件
      savePdfToPng 参数: 是否要将pdf的内容截图保存为png文件。 Ture 保存， False 不保存
      page_overlap 参数: 页和页直接的 重叠大小，页和页直接回保存重叠的内容 单位是200个字符
      excluded_content 参数: 是指的是需要在document的txt文档中排除掉的内容。(pdf有些有规律的内容是没用的，可以用这个将其清除掉，同时可以用正则表达式)
          "excluded_content":[
                   ("str","→_→\n https://github.com/datawhalechina/leedl-tutorial\n ←_←"),
                   ("regex",r"[A-Z][a-z]*")
                ],
    
    上面都是新增参数，用于pdf文档特殊处理的。如果不用特殊处理直接使用 SimpleDirectoryReader 上的注释里的配置即可。设置 好 input_dir 就可以了

  '''
  # 如果是普通pdf处理直接设置  SimpleDirectoryReader 作为 解析器。
  # 如果是需要将pdf的页面信息，还留有将pdf每一页的信息整理出来就是用 wlSimplePdfDirectoryReader， 将pdf解析成markdown格式，并且获取文档中的页面信息和截图信息
  # 两者区别是pdf处理的时候的区别。wlSimplePdfDirectoryReader是继承了SimpleDirectoryReader的，做了特殊的pdf文档处理。

  #知识库bank配置
  knowledge_config =  [
    {
      "knowledge_id": "wl_pdf_rag",
      "data_processing": [
        {
          "load_data": {
            "loader": {
              "create_object": True,
              "module": "llama_index.core",
              "class": "wlSimplePdfDirectoryReader",
              "init_args": {
                "input_dir": "/home/ubuntu/wlwork/llama_index_mutimodeRag/xyxk",
                "output_dir":"./wlKnowledgeRag",
                "savePdfToPng": True,
                "page_overlap": 200,
                "excluded_content":[
                   ("str","→_→\n https://github.com/datawhalechina/leedl-tutorial\n ←_←")
                ],
                "required_exts": [
                  ".pdf",".txt",".docx"
                ]
              }
            }
          },
        }
      ]
    },
  ]

  #--------------------------------------------------------------
  # 多模态 Embedding
  myMutilModalEmbed_model = wlMultiModalEmbedding(
      model_name = '/models/bge-m3',
      model_weight = '/models/bge-visualized/Visualized_m3.pth'
  )

  # 多模态 llm,用来回答多模态问题的，当前pdf没使用
  local_chatLLm_cfg = {
      "model":"/data2/models/Qwen2.5-72B-Instruct-AWQ",  # 使用您的模型名称
      "api_base":"http://172.21.30.230:8980/v1/",  # 您的 vllm 服务器地址
      "api_key":"EMPTY",  # 如果需要的话
  }
  # define our lmms ,采用本地 qwen2-vl大模型
  openai_mm_llm = OpenAIMultiModal(**local_chatLLm_cfg)
#--------------------------------------------------------------
#  首先，创建知识库，这个会使用llama-index对文档进行解析，形成索引，
#  首次加载会比较慢，后续会使用这个已经建立的索引文件恢复索引
#--------------------------------------------------------------#
  # 记录访问大模型的初始时间
  wlstartTime = time.time()
  # 知识库，用于存储 LlamaIndexKnowledge 对象的容器，根据配置初始化知识库
  wlknowledge_bank = KnowledgeBank(configs = knowledge_config,
                                  emb_model = myMutilModalEmbed_model,
                                  muti_llm_model = openai_mm_llm)
  wlEndTime = time.time()
  restoreTimeCost = wlEndTime - wlstartTime

  print("============KnowledgeBank restore:")
  print("花费了实际是(s):",restoreTimeCost,flush=True)
  print("============KnowledgeBank restore:")

  if 1:
    #获取知识库
    wl_pdf_rag_knowledge =  wlknowledge_bank.get_knowledge("wl_pdf_rag")

    #上面是1个知识库

    # 查询

    #query = "武汉有什么特点"
    query = "我想加入星云星空，我是个人"
    #query = "有哪些服务"
    #query = "chatgpt是什么智力水平"

    print(f"============KnowledgeBank query:{query}")

    wlstartTime = time.time()
    nodes_with_score = wl_pdf_rag_knowledge.retrieve(query = query,similarity_top_k = 5)
    wlEndTime = time.time()
    queryTimeCost = wlEndTime - wlstartTime
    print("============wl_pdf_rag_knowledge retrieve:")
    print("花费了实际是(s):",queryTimeCost,flush=True)
    print("============wl_pdf_rag_knowledge retrieve:")
    print("========================>nodes_with_score:",nodes_with_score)
    print("\n\n")

    new_nodes_with_score = []
    for node_with_score in nodes_with_score:
       score = node_with_score.get_score()
       # 只有分数大于 0.5的才获取
       if score >= 0.5:
          new_nodes_with_score.append(node_with_score)
          textcontent = node_with_score.get_text()
          retrievescore = node_with_score.get_score()
          print(f"========================>nodes_with_score score{retrievescore} content:{textcontent}",)
          
    print("========================>nodes_with_score:",new_nodes_with_score)
  #----------------------------------------------------
  #  返回的 nodes_with_score 是一个类的实例
  #  - nodes_with_score.get_text() 是这个节点的文本内容。
  #  - nodes_with_score.get_score() 是这个节点的相似度分数。



#----------------将查询的 文本和 图片 进行Embedding处理，然后使用Embedding进行查询---------------------------------------
if 0:
  query = "我想找一张这张图片类似的图片"
  query_image = "../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg"

  wlstartTime = time.time()
  nodes_with_score = wl_asl_rag_knowledge.mutimodel_retrieve(query = query, query_image_path = query_image, similarity_top_k = 2)
  wlEndTime = time.time()
  queryTimeCost = wlEndTime - wlstartTime

  print("============wl_asl_rag_knowledge retrieve:")
  print("花费了实际是(s):",queryTimeCost,flush=True)
  print("============wl_asl_rag_knowledge retrieve:")
  print("========================>nodes_with_score:",nodes_with_score)


  query = "我想知道健成星云是家什么公司？"
  #query_image = "../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg"

  wlstartTime = time.time()
  nodes_with_score = wl_asl_rag_knowledge.mutimodel_retrieve(query = query, query_image_path = None, similarity_top_k = 2)
  wlEndTime = time.time()
  queryTimeCost = wlEndTime - wlstartTime

  print("============wl_asl_rag_knowledge retrieve:")
  print("花费了实际是(s):",queryTimeCost,flush=True)
  print("============wl_asl_rag_knowledge retrieve:")
  print("========================>nodes_with_score:",nodes_with_score)


#----------------使用query engine---------------------------------------
if 0:
  print("============使用query engine 进行查询")
  # 查询模板 其中 imgdesc_str ，imgpath_str，context_str，query_str 是需要替换的变量,名字不能变
  qa_tmpl_str_cn = (
      "以下内容包含与用户查询相关的上下文，包括图片的描述信息、图片路径，以及相关的文字内容。\n"
      "请严格根据以下上下文回答问题，而不是基于其他知识。\n\n"

      "---------------------\n"
      "[图片描述信息]\n"
      "{imgdesc_str}\n"
      "---------------------\n"

      "[文字内容]\n"
      "{context_str}\n"
      "---------------------\n"

      "回答规则：\n"
      "1. 如果回答需要引用图片，请选择最相关的一张图片，并在答案中明确提供对应的图片路径。\n"
      "2. 如果回答不需要图片，仅根据文字内容回答即可。\n"
      "3. 如果无法根据提供的上下文回答问题，请明确回答“无法回答查询”。\n"
      "4. 在任何情况下，请仅根据上述内容回答问题，避免引用先验知识。\n\n"

      "用户的查询如下：\n"
      "---------------------\n"
      "查询: {query_str}\n"
      "---------------------\n"

      "答案（如涉及图片，请包括图片路径）: "
  )



  query = "我想找一张与这张图片类似的图片,找到后请描述下该图片的内容"
  query_image = "../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg"

  print(f"============wl_asl_rag_knowledge mutimodel_query begin:\nquery:{query} and query_image:{query_image}")
  wlstartTime = time.time()
  response = wl_asl_rag_knowledge.mutimodel_query(query = query, 
                                                  query_image_path = query_image, 
                                                  similarity_top_k = 2,
                                                  query_template = qa_tmpl_str_cn)
  wlEndTime = time.time()
  queryTimeCost = wlEndTime - wlstartTime
  print("============wl_asl_rag_knowledge mutimodel_query:\n",response)

#----------------使用qwen-agent查询---------------------------------------
if 0:
  # Config for the model
  llm_cfg_oai = {
      # Using Qwen2-VL deployed at any openai-compatible service such as vLLM:
      'model_type': 'qwenvl_oai',
      'model': '/work/wl/wlwork/my_models/Qwen2-VL-72B-Instruct-GPTQ-Int4',
      'model_server': 'http://172.21.30.230:8980/v1/',  # api_base
      'api_key': 'EMPTY',
  }
  llm = get_chat_model(llm_cfg_oai)

  query = "我想找一张与这张图片类似的图片,找到后请描述下该图片的内容，并呈现给用户"
  query_image = "../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg"

  # Initial conversation
  messages = [{
      'role':
          'user',
      'content': [{
          'image': 'file://../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg'
      }, {
          'text': '这是一张本地路径是：../ai_challenger_caption_test_a_20180103/caption_test_a_images_20180103/0ccf2673cd78469f4fad140311fcf48e5e406d2c.jpg 的图片，\
          请描述下该图片的内容，并将图片显示给用户'
      }]
  }]

  def image_show(image_path: str):
      print("===================这个函数模拟显示图片功能===================")
  
  functions = [
      {
          'name': 'image_show',
          'description': '在用户界面显示图片内容，输入完整的本地路径，会在用户界面显示对应的图片',
          'parameters': {
              'name': 'image_path',
              'type': 'string',
              'description': '本地图片的绝对路径',
              'required': True
          }
      },
  ]

  print('# Assistant Response 1:')
  responses = []
  for responses in llm.chat(messages=messages, functions=functions, stream=True):
      print(responses)
  messages.extend(responses)