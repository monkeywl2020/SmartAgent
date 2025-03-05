import sys
import os
import asyncio
from typing import Generator,List, Dict

from .history_db_control import HistoryDBControl,db_control
from loguru import logger

# 获取历史记录。
def get_history_record(user_id: str, session_id: str, n: int = 3) -> List[Dict]:
    logger.info(f"get_history_record===========begin1")
    try:
        # 先检查会话是否存在
        session_exist = db_control.check_session_exist(user_id, session_id)
        logger.info(f"Session {session_id} exists: {session_exist}")
        
        if not session_exist:  # 判断会话是否不存在
            return []  # 会话不存在，返回空列表

        # 获取最近 3 轮历史记录
        history = db_control.get_history_record(user_id, session_id, n)
        logger.info(f"get_history_record: [userid:{user_id}, sessionid:{session_id}, n:{n}] history: {history}")
        
        return history
    except Exception as e:
        logger.error(f"Error occurred while getting history for user {user_id}, session {session_id}: {str(e)}")
        return []  # 返回空列表，如果发生异常

# 保存历史记录消息 ,保存之前要查找 会话是否存在
def save_history_record(user_id: str, session_id: str, role: str, content: str) -> bool:
    logger.info(f"save_history_record===========begin1")

    #先检查会话是否存在
    session_exist = db_control.check_session_exist(user_id, session_id)
    logger.info(f"Session {session_id} is exist [{session_exist}]")

    if session_exist:
        # 如果会话已经存在，直接添加聊天记录
        rst = db_control.append_history_record(user_id, session_id, role, content)
        logger.info(f"save_history_record (session exists): [userid:{user_id}, sessionid:{session_id}, role:{role}, content:{content}] rst:{rst}")
    else:
        # 如果会话不存在，先创建会话
        title = content[:20]  # 假设前 20 个字符作为标题
        rst = db_control.create_session(user_id, session_id, title)

        if rst:
            # 如果会话创建成功，添加聊天记录
            rst = db_control.append_history_record(user_id, session_id, role, content)
            logger.info(f"save_history_record (session created): [userid:{user_id}, sessionid:{session_id}, role:{role}, content:{content}] rst:{rst}")
    
    return rst
