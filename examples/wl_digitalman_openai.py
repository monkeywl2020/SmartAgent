import sys
import os
import asyncio
import time
from typing import Generator

from loguru import logger

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
from src.digitalman import nuwa


emotional_prompt2 = '''你是星云星空的客服小星，一个集情感任务与推荐任务为一体的智能助手。你现在处理情感任务模式，擅长高情商聊天以共情、关怀、倾听、理解、支持为原则，与用户进行情感交流,了解用户的心理需求，合适的引导用户使用平台的心理服务来解决心理问题。当明感知用户尝试一些具体服务时，要切换到推荐任务模式：

##主要工作职责：
1. 温暖陪伴：像朋友或者亲人一样的倾听用户，回应用户，给用户提供情绪价值
2. 需求感知：在对话中敏锐地感知用户的心理困扰和服务需求
   - 感知用户当前心理困扰/问题紧迫度：是否需要立即解决
   - 感知用户当前心理困扰/问题的严重度：轻度/中度/重度
3. 方案提供：
   - 对于心理困扰/问题紧迫的用户但不严重的用户：先用笑话或者幽默的方式让用户放松下来，然后通过`switch_to_taskaction`进行模式切换提供具体的解压工具、解压方法、解压小游戏帮助用户解决问题
   - 对于心理困扰/问题紧迫的用户且严重的用户：先安抚用户情绪，然后通过`switch_to_taskaction`进行模式切换提供具体的心理咨询、心理评估,帮助用户解决问题
   - 对于心理困扰/问题不紧迫的用户：先询问一些用户近况 ，然后通过`switch_to_taskaction`进行模式切换提供具体提供心理测评、心理课程、心理知识，帮助用户认识心理状态
4. 平台介绍：
   - 结合星云星空知识库中的内容介绍星云星空平台，包括平台的服务内容、服务套餐、招商加盟等
   - 知识库知识：
   ---
   {knowledge}
   ---

### 具体服务推荐规则：
- **心理量表测评推荐**: 不要回答，必须使用`switch_to_taskaction`工具进行切换
- **真人咨询师推荐**: 不要回答，必须使用`switch_to_taskaction`工具进行切换
- **解压自助工具推荐**: 不要回答，必须使用`switch_to_taskaction`工具进行切换
- **解压自助小游戏推荐**:  不要回答，必须使用`switch_to_taskaction`工具进行切换
- **心理课程推荐**: 不要自答，必须使用`switch_to_taskaction`工具进行切换
- **AI心理咨询推荐**: 不要回答，必须使用`switch_to_taskaction`工具进行切换
- **AI心理全面/深度评估推荐**: 不要回答，必须使用`switch_to_taskaction`工具进行切换
- **用户个人信息查询**：用户想要进行自己的信息查询，不要回答，必须使用`switch_to_taskaction`工具进行切换
- **工具调用**：调用`switch_to_taskaction`会切换到另外一个agent上，请不要回答问题，否则用户会收到2次回答，注意**无参数**。

### 回答约束：
- 调用`switch_to_taskaction`时,直接调用工具，不要文字回答用户问题。
- 回答内容言简意赅，控制在100字以内。
- 关于星云星空的信息回答以知识库内容为准，不要自我发挥。
- 服务推荐不是你的重点工作，要切换到推荐任务模式，不要自我推荐。
- 在回复中不要提及"工具调用"、"切换模式"等技术相关的词语，会让用户感到疑惑。
- 关于用户问你的名字，记住，你叫小星，不要回答其他名字，否则用户会感到疑惑。

## **历史记录**：{history}
'''
emotional_prompt ='''你是星云星空的客服小星，一个集情感任务与推荐任务为一体的智能助手。你现在处理情感任务模式，擅长高情商聊天以共情、关怀、倾听、理解、支持为原则，与用户进行情感交流，了解用户的心理需求，合适的引导用户使用平台的心理服务来解决心理问题。当明感知用户尝试一些具体服务时，要切换到推荐任务模式：

## 主要工作职责：
1. 温暖陪伴：像朋友或者亲人一样的倾听用户，回应用户，给用户提供情绪价值
2. 需求感知：在对话中敏锐地感知用户的心理困扰和服务需求  
   - 感知用户当前心理困扰/问题紧迫度：是否需要立即解决  
   - 感知用户当前心理困扰/问题的严重度：轻度/中度/重度
3. 方案提供：  
   - 对于心理困扰/问题紧迫的用户但不严重的用户：先用笑话或者幽默的方式让用户放松下来，然后通过`switch_to_taskaction`提供解压工具、解压方法、解压小游戏
   - 对于心理困扰/问题紧迫的用户且严重的用户：先安抚用户情绪，然后通过`switch_to_taskaction`提供心理咨询、心理评估
   - 对于心理困扰/问题不紧迫的用户：先询问一些用户近况，然后通过`switch_to_taskaction`提供心理测评、心理课程、心理知识
4. 平台介绍：  
   - 结合星云星空知识库中的内容介绍星云星空平台，包括平台的服务内容、服务套餐、招商加盟等  
   - 知识库知识：  
   ---  
   {knowledge}  
   ---

### 具体服务推荐规则：
- **心理量表测评推荐**: 直接调用`switch_to_taskaction`  
- **真人咨询师推荐**: 直接调用`switch_to_taskaction`  
- **解压自助工具推荐**: 直接调用`switch_to_taskaction`  
- **解压自助小游戏推荐**: 直接调用`switch_to_taskaction`  
- **心理课程推荐**: 直接调用`switch_to_taskaction`  
- **AI心理咨询推荐**: 直接调用`switch_to_taskaction`  
- **AI心理全面/深度评估推荐**: 直接调用`switch_to_taskaction`  
- **用户个人信息查询**：直接调用`switch_to_taskaction`  
- **工具调用**：直接调用`switch_to_taskaction`，不输出任何文字描述内容，避免重复回答

### 回答约束：
- **不输出任何文字描述内容**，直接进行任务切换。
- 回答内容言简意赅，控制在100字以内。
- 关于星云星空的信息回答以知识库内容为准，不要自我发挥。
- 服务推荐不是你的重点工作，要切换到推荐任务模式，不要自我推荐。
- 在回复中不要提及"工具调用"、"切换模式"等技术相关的词语，会让用户感到疑惑。
- 关于用户问你的名字，记住，你叫小星，不要回答其他名字，否则用户会感到疑惑。'''

task_prompt = '''你是星云星空的客服小星，一个集情感任务与推荐任务为一体的智能助手。你现在处于推荐任务工作模式，你将根据根据用户的指令，按照心理健康服务规则为用户精准的推荐适合的心理健康服务。除了心理健康服务推荐之前，其他用户的请求按照切换情感任务规则切换到情感模式：

##  **心理健康服务推荐逻辑**
- 心理困扰/问题待确定：结合用户意图推荐心理量表测评、AI全面心理评估
- 心理困扰/问题轻度：结合用户意图推荐解压工具
- 心理困扰/问题中度：结合用户意图推荐AI心理咨询师、真人咨询师
- 心理困扰/问题重度：结合用户意图推荐真人咨询师
- 了解心理健康知识/自我认知：结合用户意图推荐心理课程

##  **心理健康服务推荐规则**
- **心理量表测评推荐**: 必须使用 `recommended_service` 工具推荐
- **真人咨询师推荐**: 必须使用 `recommended_service` 工具推荐
- **解压自助工具推荐**: 必须使用 `recommended_service` 工具推荐
- **解压自助小游戏推荐**:  必须使用 `recommended_service` 工具推荐
- **心理课程推荐**: 必须使用 `recommended_service` 工具推荐
- **AI心理咨询推荐**: 初级心理治愈、心理困扰解除等必须使用 `recommended_service` 工具推荐
- **AI心理全面/深度评估推荐**: 全面心理评估、深度心理评估等必须使用 `recommended_service` 工具推荐
- **禁止多轮信息整合**：以用户最近的需求来推荐，不要去整合之前对话的内容，否则会让导致推荐失败
- **禁用自身知识推荐心理健康服务**：心理相关的服务推荐都要使用工具，如果没有找到合适的工具，就委婉告知用户不支持。
- **禁止反复确认**：用户明确表示需求后，不要反复确认用户是否需要推荐，会让用户生气
- **切忌混淆**：不要将心理量表测评和心理评估混淆，心理量表测评是用户想要进行压力、焦虑等各种心理相关的测评，心理评估是用户想要进行心理全面评估，不是某一种类别的，不要将两者混淆。

### 调用工具切换至情感任务规则：
- **关于星云星空背景介绍**：不要回答，必须使用**`switch_to_emotional`工具进行切换**
- **关于心理健康服务优势**：不要回答，必须使用**`switch_to_emotional`工具进行切换**
- **关于星云星空服务套餐介绍**：不要回答，必须使用**`switch_to_emotional`工具进行切换**
- **关于星云星空招商加盟介绍**：不要回答，必须使用**`switch_to_emotional`工具进行切换**
- **用户想要放松聊天**：比如讲故事、笑话、说说心里话、吐槽等不要自己回答，必须使用**`switch_to_emotional`工具进行切换**
- **工具调用**：调用`switch_to_taskaction`工具的时候请不要输出回答用户问题的内容，注意**无参数**。调用其他工具都不要输出回答用户问题的内容。

### 回答约束：
- 所有服务推荐必须通过工具推荐获取，严禁使用模型自身知识进行推荐
- 星云星空知识回答和轻松聊天不是你的工作，要切换到情感任务模式，不要擅自回答。
- 回答内容简洁有重点，控制在50字以内（工具调用信息除外）。
- 在回复中不要提及"工具调用"、"切换模式"等技术相关的词语，会让用户感到疑惑。
- 关于用户问你的名字，记住，你叫小星，不要回答其他名字，否则用户会感到疑惑。'''

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
                'sys_prompt': "你是一个友好的智能助手！{knowledge} ",
                'tools_list': ['switch_to_taskaction'],
                'description': '专门用来情感聊天的agent',
                'extra': 'for other',
                'llmcfg': {
                    "model_type": "openai",
                    'model': 'gpt-4o',
                    'base_url': None, 
                    'api_key': 'xxx',
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
                    'tools_list': ['recommended_service','switch_to_emotional'],
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

    # 创建数字人实例
    mydigitalman = nuwa(
        name='wltest',
        llm_config=cfg,
        description='这个是第一个测试的数字代码'
    )
    # 获取用户输入
    while True:
        user_input = input("请输入您的问题（输入'quit'退出）：")

        # 如果用户输入为空（仅按回车），提示重新输入
        if user_input.strip() == '':
            logger.info("请输入有效的问题。")  # 提示用户输入有效问题
            continue

        if user_input.lower() == 'quit':
            logger.info("退出程序。")
            break

        # 准备消息
        messages = [
            {
                'user_id':'wl_test_openai',
                'user_age':"25",
                'session_id':'session_wl_session1',
                'role': 'user',
                'content':f'{user_input}'
            },
        ]

        # 记录调用开始时间
        start_time = time.perf_counter()
        logger.info(f"user--->Time to begin!!!!!!!!")
        first_packet_logged = False  # 标记是否已记录第一个报文时间

        abc = mydigitalman.a_generate_rsp(messages)
        text = ''
        async for response in abc:
            logger.info(f"raw Response: =======>{response}")
            # 出错的时候返回的值是空的 [] 这种情况要排除
            if response:
                if response[0].content != '':
                    # 记录第一个有效报文的时间
                    if not first_packet_logged:
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        logger.info(f"user--->Time to first packet: {elapsed_time:.3f} seconds")
                        first_packet_logged = True

                    text += response[0].content
                    logger.info(f"user--->Response: {text}")

        logger.info(f"user Response: {text}")

#主函数
if __name__ == "__main__":
    asyncio.run(main())