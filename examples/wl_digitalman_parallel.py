import sys
import os
import asyncio
import time
from typing import Generator



# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
print(a,flush=True)
b = os.path.dirname(a)  #返回上一级目录部分，去掉文件名
print(b,flush=True)
#sys.path.append(b)
c = os.path.dirname(b) #返回上一级目录部分
print(c,flush=True)

# 将上一级目录加入 添加到搜索路径中也 就是examples的上级目录
sys.path.append(c)
print(sys.path,flush=True)

from src.msg.message import Message
from src.digitalman import nuwaParallel

emotional_prompt ='''你是星云星空的客服小星，一个集情感任务与推荐任务为一体的智能助手。你现在处理情感任务模式，擅长高情商聊天以共情、关怀、倾听、理解、支持为原则，与用户进行情感交流，了解用户的心理需求，合适的引导用户使用平台的心理服务来解决心理问题。当明感知用户尝试一些具体服务时，要切换到推荐任务模式：

## 主要工作职责：
1. 温暖陪伴：像朋友或者亲人一样的倾听用户，回应用户，给用户提供情绪价值
2. 需求感知：在对话中敏锐地感知用户的心理困扰和服务需求  
   - 感知用户当前心理困扰/问题紧迫度：是否需要立即解决  
   - 感知用户当前心理困扰/问题的严重度：轻度/中度/重度
3. 方案提供：  
   - 对于心理困扰/问题紧迫的用户但不严重的用户：先用笑话或者幽默的方式让用户放松下来，让用户稍后，另外一个智能助手将会提供解压工具、解压方法、解压小游戏
   - 对于心理困扰/问题紧迫的用户且严重的用户：先安抚用户情绪，让用户稍后，另外一个智能助手将会提供心理咨询、心理评估
   - 对于心理困扰/问题不紧迫的用户：先询问一些用户近况，让用户稍后，另外一个智能助手提供心理测评、心理课程、心理知识
4. 平台介绍：  
   - 结合星云星空知识库中的内容介绍星云星空平台，包括平台的服务内容、服务套餐、招商加盟等  
   - 知识库知识：  
   ---  
   {knowledge}  
   ---

### 回答约束：
- 回答内容言简意赅，控制在100字以内。
- 关于星云星空的信息回答以知识库内容为准，不要自我发挥。
- 在回复中不要提及"工具调用"、"切换模式"等技术相关的词语，会让用户感到疑惑。
- 关于用户问你的名字，记住，你叫小星，不要回答其他名字，否则用户会感到疑惑。'''

task_prompt = '''你是一个智能助手，名叫小星，现处于推荐任务模式，根据用户意图按心理健康服务规则推荐服务。

## 心理健康服务推荐逻辑
- **心理困扰待确定**：推荐心理量表测评或AI全面心理评估  
- **轻度困扰**：推荐解压工具  
- **中度困扰**：推荐AI心理咨询师或真人咨询师  
- **重度困扰**：推荐真人咨询师  
- **了解心理知识/自我认知**：推荐心理课程  
- **无服务需求**：简单回复，无需推荐  

## 推荐规则
- 所有服务推荐（测评、咨询、工具、课程等）均使用 `recommended_service` 获取  
- 以用户当前需求为准，不整合过往对话  
- 不使用自身知识推荐服务，未找到合适服务时委婉告知  
- 用户意图明确后，不重复确认  
- 区分心理量表测评（单项测试）和心理评估（全面分析），不可混淆  

## 回答约束
- 推荐服务时简洁明了，控制在50字以内  
- 不提及技术术语（如“工具调用”）  
- 仅回答“我的名字是小星”，不使用其他名字'''

# iq的task_prompt 这个提示词历史记录，不用显示写在这，会自动加到系统提示词后面   ## **历史记录**：{history}

# 双agent数字人

async def main():
    # 如果 对接大模型使用的 qwen-agent，那么 配置和openAI还有postApi是不一样的。
    # 同时tool的配置也不一样，不能配置在 agent里头。需要配置在大模型配置里头。因为agent里面没有 qwen-agent要使用的tool
    # qwen-agent的 tool 是独立的格式，与上面两个不一样，所以需要将配置放到llmcfg里头，最终传给 qwen的模型适配层
    #  sys_prompt 也是，是给qwen大模型的客户端 使用的，agent不需要
    cfg = {
        'agentcfg': [
            {
                'agent_type': 'emotional_agent',
                'agent_name': 'agent_abc',
                'sys_prompt': emotional_prompt,
                'description': '专门用来情感聊天的agent',
                'extra': 'for other',
                'llmcfg': {
                    "model_type": "openai",
                    'model': '/data2/models/Qwen2.5-72B-Instruct-AWQ',
                    'base_url': 'http://172.21.30.230:8985/v1',  
                    'api_key': 'EMPTY',
                    'top_p': 0.8,
                    'temperature': 0.7,
                }
            },
            {
                'agent_type': 'qwen_taskaction_agent',
                'agent_name': 'agent_wl',
                'description': '专门用来 tool call 调用执行任务的agent',
                'llmcfg': {
                    "model_type": "qwen",     
                    'sys_prompt': task_prompt,               
                    'tools_list': ['recommended_service'],
                    "qwenllmcfg": {
                        'model': '/data2/models/Qwen2.5-72B-Instruct-AWQ',
                        'model_server': 'http://172.21.30.230:8980/v1',  
                        'api_key': 'EMPTY',
                        'generate_cfg': {
                            'top_p': 0.8,
                            'temperature': 0.1,
                        }
                    }
                }
            }
        ],
        'workmode': 'switch'
    }   

    cfg1 = {
        'agentcfg': [
            {
                'agent_type': 'emotional_agent',
                'agent_name': 'agent_abc',
                'sys_prompt': emotional_prompt,
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
                'tools_list': ['recommended_service'],
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

    # 创建数字人实例
    mydigitalman = nuwaParallel(
        name='wltest',
        llm_config=cfg,
        description='这个是第一个测试的数字代码'
    )
    # 获取用户输入
    while True:
        user_input = input("请输入您的问题（输入'quit'退出）：")

        # 如果用户输入为空（仅按回车），提示重新输入
        if user_input.strip() == '':
            print("请输入有效的问题。")  # 提示用户输入有效问题
            continue

        if user_input.lower() == 'quit':
            print("退出程序。")
            break

        # 准备消息
        messages = [
            {
                'user_id':'wl_test2',
                'user_age':"25",
                'session_id':'session_wl_session1',
                'role': 'user',
                'content':f'{user_input}'
            },
        ]

        # 记录调用开始时间
        start_time = time.perf_counter()
        first_packet_logged = False  # 标记是否已记录第一个报文时间

        abc = mydigitalman.a_generate_rsp(messages)
        text = ''
        async for response in abc:
            print(f"raw Response: =======>{response}")
            # 出错的时候返回的值是空的 [] 这种情况要排除
            if response:
                if response[0].content != '':
                    # 记录第一个有效报文的时间
                    if not first_packet_logged:
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"user--->Time to first packet: {elapsed_time:.3f} seconds")
                        first_packet_logged = True

                    text += response[0].content
                    print(f"user--->Response: {text}")

        print(f"user Response: {text}")

#主函数
if __name__ == "__main__":
    asyncio.run(main())