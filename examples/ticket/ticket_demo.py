import asyncio
from typing import Any, List, Dict
import json
import sys
import os
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.core.base_agent import (
    BaseAgent, 
    PersonalityTrait, 
    SpeakingStyle, 
    Task, 
    Tool,
    GLOBAL_AGENT_REGISTRY
)
from examples.ticket.refund_agent import RefundAgent
from examples.ticket.booking_agent import BookingAgent

async def chat_loop(agent: BookingAgent):
    """处理多轮对话的循环"""
    print(f"\n=== 开始与{agent.name}对话 ===")
    print("提示：输入'quit'或'exit'结束对话")
    
    dialogue_count = 0
    recent_messages = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\nUser> ").strip()
            
            # 检查是否退出
            if user_input.lower() in ['quit', 'exit']:
                print(f"\n=== 结束与{agent.active_agent.name}的对话 ===")
                break
            
            if not user_input:
                continue
            
            # 记录用户消息
            recent_messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"\nAssistant ({agent.active_agent.name})> ", end="", flush=True)
            
            # 使用run_agent处理对话
            assistant_response = ""
            async for response in agent.run_agent(user_input):
                response_obj = json.loads(response)
                if response_obj["type"] == "content":
                    content = response_obj["content"]
                    assistant_response += content
                    print(content, end="", flush=True)
                elif response_obj["type"] == "tool_call":
                    print(f"\n[工具调用] {response_obj['content']}")
                elif response_obj["type"] == "error":
                    print(f"\n[错误] {response_obj['content']}")
            
            # 记录助手回复
            recent_messages.append({
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat()
            })
            
            print()  # 换行
            
            # 增加对话计数
            dialogue_count += 1
            
            # 每5轮对话存储一次长期记忆
            if dialogue_count % 5 == 0:
                try:
                    agent.active_agent.memorize({
                        "dialogue_round": dialogue_count,
                        "messages": recent_messages,
                        "timestamp": datetime.now().isoformat()
                    }, "LONG_TERM")
                    recent_messages = []
                except Exception as e:
                    print(f"\n[记忆处理错误] {str(e)}")
            
        except KeyboardInterrupt:
            print("\n\n=== 对话被用户中断 ===")
            break
        except Exception as e:
            print(f"\n[系统错误] {str(e)}")
            print("继续对话...\n")

async def main():
    # 先创建客服智能体实例
    refund_agent = RefundAgent(
        name="客服小美",
        personality=[
            PersonalityTrait.EMPATHETIC,
            PersonalityTrait.PROFESSIONAL
        ],
        speaking_style=SpeakingStyle.FORMAL,
        skills_description=[
            "专业的退票和改签服务",
            "投诉处理和问题解决",
            "紧急情况应对",
            "客户满意度跟进"
        ]
    )
    
    # 然后创建搜索智能体实例
    booking_agent = BookingAgent(
        name="小鱼儿",
        personality=[
            PersonalityTrait.FRIENDLY,
            PersonalityTrait.PROFESSIONAL
        ],
        speaking_style=SpeakingStyle.CASUAL,
        skills_description=[
            "专业的火车票查询和预订服务",
            "实时列车时刻表查询",
            "智能座位推荐",
            "多站点路线规划"
        ]
    )

    # 在创建完所有智能体后，手动更新它们的协调工具
    #refund_agent._update_coordination_tool()
    #booking_agent._update_coordination_tool()
    
    print("\n=== 验证Agent互相发现 ===")
    
    def print_agent_info(agent):
        # 基本信息
        print(json.dumps({
            "name": agent.name,
            "personality": [trait.value for trait in agent.personality],
            "speaking_style": agent.speaking_style.value,
            "skills_description": agent.skills_description,
        }, indent=2, ensure_ascii=False))
        
        # 工具详细信息
        print("\nAgent Tools:")
        for tool in agent.tools:
            print(json.dumps({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "required": tool.required
            }, indent=2, ensure_ascii=False))
            print()  # 工具之间添加空行
        
        # Prompt信息
        print("\nAgent Prompt:")
        print(agent._generate_prompt())
        print()  # 添加空行分隔

    # 验证搜索智能体是否能找到客服智能体
    agent1 = booking_agent.get_agent("客服小美")
    if agent1:
        print("\n搜索智能体成功找到客服智能体:")
        #print_agent_info(agent1)
    else:
        print("\n搜索智能体未能找到客服智能体")
    
    # 验证客服智能体是否能找到搜索智能体
    agent2 = refund_agent.get_agent("小鱼儿")
    if agent2:
        print("\n客服智能体成功找到搜索智能体:")
        #print_agent_info(agent2)
    else:
        print("\n客服智能体未能找到搜索智能体")

    # 配置API密钥
    api_key = "sk-55f401e819934acd87ca0a1310ace72f"
    booking_agent.client.api_key = api_key
    refund_agent.client.api_key = api_key
    
    print("\n=== 搜索智能体信息 ===")
    print("\nPrompt Template:")
    print(booking_agent._generate_prompt())
    
    print("\nTools:")
    for tool in booking_agent.tools:
        print(f"\n工具名称: {tool.name}")
        print(f"描述: {tool.description}")
        print("参数:")
        print(json.dumps(tool.parameters, indent=2, ensure_ascii=False))
        print(f"必需参数: {tool.required}")

    print("\n=== 客服智能体信息 ===")
    print("\nPrompt Template:")
    print(refund_agent._generate_prompt())
    
    print("\nTools:")
    for tool in refund_agent.tools:
        print(f"\n工具名称: {tool.name}")
        print(f"描述: {tool.description}")
        print("参数:")
        print(json.dumps(tool.parameters, indent=2, ensure_ascii=False))
        print(f"必需参数: {tool.required}")

    # 智能体发现同伴
    refund_agent.discover_peers()
    booking_agent.discover_peers()

    try:
        # 启动对话循环
        await chat_loop(booking_agent)
        
        # 打印完整对话历史
        print("\n=== 对话历史 ===")
        for msg in booking_agent.messages:
            role = msg['role']
            content = msg.get('content', '')
            if role == 'user':
                print(f"\nUser: {content}")
            elif role == 'assistant':
                print(f"Assistant: {content}")
                if 'tool_calls' in msg:
                    print(f"Tool Calls: {msg['tool_calls']}")
            elif role == 'tool':
                print(f"Tool Response: {content}")
                
    finally:
        # 清理资源
        booking_agent.cleanup()
        refund_agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main()) 