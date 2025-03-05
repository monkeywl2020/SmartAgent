from typing import Any, List, Dict
from datetime import datetime
from uuid import uuid4

from src.core.base_agent import (
    BaseAgent, 
    PersonalityTrait, 
    SpeakingStyle, 
    Tool,
    Task
)

class RefundAgent(BaseAgent):
    """客服智能体，专注于处理用户服务相关问题"""
    
    def __init__(self, name: str, personality: List[PersonalityTrait], speaking_style: SpeakingStyle, skills_description: List[str] = None):
        super().__init__(name, personality, speaking_style, skills_description)
        
        # 覆盖search和execute_task工具定义
        search_tool = Tool(
            name="search",
            description="Search for customer service related information",
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
            description="Execute customer service tasks",
            parameters={
                "task": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["HANDLE_COMPLAINT", "HANDLE_REFUND", "HANDLE_CHANGE"],  # 添加HANDLE_CHANGE
                            "description": "Type of task to execute"
                        },
                        "params": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["REFUND", "CHANGE", "SERVICE", "PRODUCT", "OTHER"]  # 添加CHANGE
                                },
                                "description": {"type": "string"},
                                "urgency": {
                                    "type": "string",
                                    "enum": ["LOW", "MEDIUM", "HIGH"]
                                },
                                "order_id": {"type": "string"},
                                "reason": {"type": "string"},
                                "original_train": {"type": "string"},  # 新增：原车次
                                "target_train": {"type": "string"},    # 新增：目标车次
                                "seat_preference": {"type": "string"}  # 新增：座位偏好
                            },
                            "required": ["type", "description"]
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
        """实现客服知识库搜索"""
        try:
            print(f"\n[DEBUG] 客服知识库搜索: {query}")
            
            # 模拟客服知识库搜索结果
            if "退款" in query or "退票" in query:
                return {
                    "type": "refund_policy",
                    "content": {
                        "policy": "根据购票时间和车次类型，退票手续费比例为5%-20%不等",
                        "procedure": ["登录12306", "在订单中心选择需要退票的订单", "点击退票按钮"],
                        "timeline": "退款将在3-7个工作日内退回原支付账户"
                    }
                }
            elif "改签" in query:
                return {
                    "type": "change_policy",
                    "content": {
                        "policy": "改签需要支付票价差额，改签手续费按照退票规则计算",
                        "procedure": ["登录12306", "在订单中心选择需要改签的订单", "选择目标车次完成改签"],
                        "timeline": "改签成功后立即生效"
                    }
                }
            elif "投诉" in query:
                return {
                    "type": "complaint_guide",
                    "content": {
                        "channels": ["12306在线客服", "95105105铁路客服电话", "车站服务台"],
                        "process": "我们会在24小时内处理您的投诉并给予回复",
                        "timeline": "紧急情况2小时内响应"
                    }
                }
            else:
                # 如果不是客服相关的查询，返回None表示无法处理
                return None
                
        except Exception as e:
            print(f"[DEBUG] 搜索错误: {str(e)}")
            return {
                "type": "error",
                "error": str(e)
            }

    async def execute_task(self, task: Task) -> None:
        """实现客服任务处理"""
        try:
            if task.type == "HANDLE_COMPLAINT":
                print(f"处理投诉: {task.params}")
                task.status = "COMPLETED"
                task.result = {
                    "status": "已受理",
                    "case_id": str(uuid4()),
                    "next_steps": ["客服将在2小时内联系您", "请保持电话畅通"],
                    "estimated_time": "2小时内"
                }
            elif task.type == "HANDLE_REFUND":
                print(f"处理退票: {task.params}")
                # 验证必要参数
                order_id = task.params.get("order_id")
                if not order_id:
                    task.status = "FAILED"
                    task.result = {
                        "status": "error",
                        "message": "退票处理需要订单号，请提供订单号",
                        "required_info": ["order_id"]
                    }
                    return

                task.status = "COMPLETED"
                task.result = {
                    "status": "success",
                    "refund_info": {
                        "order_id": order_id,
                        "refund_id": str(uuid4()),
                        "reason": task.params.get("reason", "未提供原因"),
                        "refund_amount": "￥553.5",
                        "estimated_time": "3-7个工作日",
                        "next_steps": [
                            "退款申请已受理",
                            "退款将在3-7个工作日内退回原支付账户",
                            "您可以通过订单号查询退款进度"
                        ]
                    },
                    "message": "退票申请已成功提交"
                }
            elif task.type == "HANDLE_CHANGE":
                print(f"处理改签: {task.params}")
                # 验证必要参数
                order_id = task.params.get("order_id")
                target_train = task.params.get("target_train")
                if not order_id or not target_train:
                    task.status = "FAILED"
                    task.result = {
                        "status": "error",
                        "message": "改签处理需要订单号和目标车次",
                        "required_info": [
                            param for param in ["order_id", "target_train"] 
                            if not task.params.get(param)
                        ]
                    }
                    return

                task.status = "COMPLETED"
                task.result = {
                    "status": "success",
                    "change_info": {
                        "order_id": order_id,
                        "change_id": str(uuid4()),
                        "original_train": task.params.get("original_train", "未知"),
                        "target_train": target_train,
                        "reason": task.params.get("reason", "未提供原因"),
                        "price_difference": "￥0.0",  # 实际应该计算差价
                        "estimated_time": "处理中",
                        "next_steps": [
                            "改签申请已受理",
                            "正在为您确认新车次的座位",
                            "确认成功后将发送改签结果通知",
                            "如有差价将自动退还或补收"
                        ]
                    },
                    "message": "改签申请已成功提交"
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