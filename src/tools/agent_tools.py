import json5
from typing import List, Dict

from .base import nuwa_register_tool,NuwaBaseTool

# 注册工具到 nuwa agent系统中去
@nuwa_register_tool('recommended_service')
class recommended_service(NuwaBaseTool):
    description = '推荐心理服务。'
    parameters = [{
        'name': 'recommend_type',
        'type': 'string',
        'description': (
            '推荐类型，可用选项：'
            'psychological_assessment: 心理量表测评或者心理测评，提供专业的心理量表测试'
            'ai_psychological_evaluation: 心理评估，进行深度的AI心理状态评估'
            'ai_psychological_consultation: 心理咨询，提供AI心理咨询服务'
            'psychological_course: 心理课程，提供心理健康教育课程'
            'psychological_tool: 心理工具，提供自助式心理调节工具'
            'psychological_counselor: 心理咨询师，推荐专业的人工心理咨询师'
        ),
        'enum':[
            'psychological_assessment',
            'ai_psychological_evaluation',
            'ai_psychological_consultation',
            'psychological_course',
            'psychological_tool',
            'psychological_counselor'
        ],
        'required': True
    },{
        'name':'recommend_content', 
        'type':'string', 
        'description':'推荐内容，比如量表名称或者类型、课程名称或者类型、工具名称或者类型、咨询师姓名或者类型等。',
        'optional': False
    }
    ]

    def call(self, params: str, **kwargs) -> str:
        # 获取参数
        recommend_type = json5.loads(params)['recommend_type']
        recommend_content = json5.loads(params)['recommend_content']
        print(f"接收到的类别: {recommend_type} 内容{recommend_content}")  # 调试信息

        # 验证推荐类型是否有效
        # 下面是模拟函数被调用后的结果
        return f"recommend_type: {recommend_type}, recommend_content: 这是 {recommend_content} 推荐内容名称，例如量表名称或者类型、课程名称或者类型、工具名称或者类型、咨询师姓名或者类型相关的操作。"
    

# 注册工具到 nuwa agent系统中去
@nuwa_register_tool('switch_to_emotional')
class switch_to_emotional(NuwaBaseTool):
    description = '工作模式切换，将任务执行模式切换为情感聊天模式'
    parameters = [
    ]

    def call(self, params: str, **kwargs) -> str:
        # 获取参数
        print(f"切换到情感模型 使用 emotional_agent ")  # 调试信息

        # 验证推荐类型是否有效
        # 下面是模拟函数被调用后的结果
        return {'switch_mode_to':'agent_abc'}
    
# 注册工具到 nuwa agent系统中去
@nuwa_register_tool('switch_to_taskaction')
class switch_to_taskaction(NuwaBaseTool):
    description = '工作模式切换，将情感聊天模式切换为任务执行模式'
    parameters = [
    ]

    def call(self, params: str, **kwargs) -> str:
        # 获取参数
        print(f"切换到任务模型 使用 taskaction_agent ")  # 调试信息

        # 验证推荐类型是否有效
        # 下面是模拟函数被调用后的结果
        return {'switch_mode_to':'agent_wl'}