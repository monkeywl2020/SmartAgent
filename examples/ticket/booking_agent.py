import asyncio
from typing import Any, List, Dict
import json
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# 添加项目根目录到Python路径
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.core.base_agent import (
    BaseAgent, 
    PersonalityTrait, 
    SpeakingStyle, 
    Task, 
    Tool,
    GLOBAL_AGENT_REGISTRY
)

class BookingAgent(BaseAgent):
    """搜索智能体，专注于处理订票和查询相关问题"""
    
    def __init__(self, name: str, personality: List[PersonalityTrait], speaking_style: SpeakingStyle, skills_description: List[str] = None):
        super().__init__(name, personality, speaking_style, skills_description)
        
        # 覆盖search和execute_task工具定义
        search_tool = Tool(
            name="search",
            description="Search for information about tickets, schedules, etc.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            required=["query"]
        )
        
        execute_task_tool = Tool(
            name="execute_task",
            description="Execute tasks like booking tickets, making reservations etc.",
            parameters={
                "task": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["BOOK_TICKET"],
                            "description": "Type of task to execute"
                        },
                        "params": {
                            "type": "object",
                            "properties": {
                                "departure": {"type": "string"},
                                "destination": {"type": "string"},
                                "date": {"type": "string"},
                                "train_type": {"type": "string"},
                                "seat_preference": {"type": "string"},
                                "round_trip": {"type": "boolean"}
                            },
                            "required": ["departure", "destination", "date"]
                        }
                    },
                    "required": ["type", "params"]
                }
            },
            required=["task"]
        )
        
        # 更新工具列表中的特定工具
        self.tools = [tool for tool in self.tools if tool.name not in ["search", "execute_task"]]
        self.tools.extend([search_tool, execute_task_tool])

    async def search(self, query: str) -> Any:
        """实现订票相关的搜索"""
        return {
            "result": f"这是关于 '{query}' 的订票相关信息",
            "source": "订票系统"
        }

    async def execute_task(self, task: Task) -> None:
        """实现订票任务执行"""
        try:
            if task.type == "BOOK_TICKET":
                # 验证必要参数
                required_params = ["departure", "destination", "date"]
                for param in required_params:
                    if param not in task.params:
                        raise ValueError(f"缺少必要参数: {param}")
                
                print(f"正在订票: 从{task.params['departure']}到{task.params['destination']}")
                print(f"日期: {task.params['date']}")
                print(f"偏好: {task.params.get('seat_preference', '无特殊要求')}")
                
                # 模拟订票过程
                task.status = "COMPLETED"
                task.result = {
                    "status": "success",
                    "ticket_info": {
                        "order_id": str(uuid4()),
                        "departure": task.params["departure"],
                        "destination": task.params["destination"],
                        "date": task.params["date"],
                        "seat_type": task.params.get("seat_preference", "二等座"),
                        "price": "￥553.5"
                    },
                    "message": "订票成功"
                }
            else:
                raise ValueError(f"不支持的任务类型: {task.type}")
                
        except Exception as e:
            task.status = "FAILED"
            task.result = {
                "status": "error",
                "error": str(e)
            }
            raise
