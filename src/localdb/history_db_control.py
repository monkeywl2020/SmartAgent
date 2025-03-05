import sys,os
import json
import ujson
from datetime import datetime
from typing import List, Dict

from .dbclient.mysql_client import MysqlManagerMain
from .dbclient.redis_client import RedisManager
from .dbclient.history_content_mysql_table import HistoryContentTable
from loguru import logger

class HistoryDBControl:
    def __init__(self):
        # 初始化 MySQL 管理器
        self.mysql_manager = MysqlManagerMain()
        if not self.mysql_manager.is_connected():
            raise Exception("MySQL connection failed")

        # 初始化 Redis 管理器
        self.redis_manager = RedisManager()
        if not self.redis_manager.is_connected():
            raise Exception("Redis connection failed")

        # 初始化 MySQL 历史表,创建mysql表，并加载表操控的处理
        self.history_table = HistoryContentTable(self.mysql_manager.execute_query_)

    # 创建会话
    def create_session(self, user_id: str, session_id: str, session_name: str = "新会话", session_type: int = 0) -> bool:
        """创建新会话，同时存储到 MySQL 和 Redis"""
        try:
            # 存储到 MySQL
            mysql_success = self.history_table.add_new_session(user_id, session_id, session_name, session_type)
            if not mysql_success:
                logger.error(f"Failed to create session in MySQL for {user_id}/{session_id}")
                return False

            # 存储到 Redis 会话元数据
            session_data = {
                "session_name": session_name,
                "session_type": str(session_type),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            redis_success = self.redis_manager.set_session(user_id, session_id, session_data)
            if not redis_success:
                logger.warning(f"Failed to create session in Redis for {user_id}/{session_id}, proceeding with MySQL only")

            logger.info(f"Session {session_id} created for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating session for {user_id}/{session_id}: {str(e)}")
            return False
        
    # 删除会话
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除会话及其聊天记录，同时从 MySQL 和 Redis 删除"""
        try:
            # 删除 MySQL 数据
            mysql_success = self.history_table.delete_session(user_id, session_id)
            if not mysql_success:
                logger.error(f"Failed to delete session in MySQL for {user_id}/{session_id}")
                return False

            # 删除 Redis 数据
            redis_success = self.redis_manager.delete_session(user_id, session_id)
            if not redis_success:
                logger.warning(f"Failed to delete session in Redis for {user_id}/{session_id}, proceeding with MySQL only")

            logger.info(f"Session {session_id} deleted for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting session for {user_id}/{session_id}: {str(e)}")
            return False

    # 获取会话列表
    def get_session_list(self, user_id: str) -> List[Dict]:
        """获取用户的会话列表，优先从 Redis 获取，失败则从 MySQL 获取并更新 Redis"""
        try:
            # 从 Redis 获取会话 ID 列表
            session_ids = self.redis_manager.get_sessions(user_id)
            if session_ids:
                result = []
                for session_id in session_ids:
                    session_data = self.redis_manager.get_session(user_id, session_id)
                    if session_data:
                        result.append({
                            "session_id": session_id,
                            "session_name": session_data.get("session_name", "新会话"),
                            "session_type": int(session_data.get("session_type", 0)),
                            "created_at": session_data.get("created_at"),
                            "updated_at": session_data.get("updated_at")
                        })
                if result:
                    logger.debug(f"Loaded session list from Redis for {user_id}")
                    return result

            # Redis 失败或为空，从 MySQL 获取并更新 Redis
            logger.debug(f"Loading session list from MySQL for {user_id}")
            sessions = self.history_table.get_session_list(user_id)
            if sessions:
                for session in sessions:
                    session_data = {
                        "session_name": session["session_name"],
                        "session_type": str(session["session_type"]),
                        "created_at": session["created_at"].isoformat(),
                        "updated_at": session["updated_at"].isoformat()
                    }
                    self.redis_manager.set_session(user_id, session["session_id"], session_data)
                logger.info(f"Updated Redis with session list for {user_id}")
            return sessions
        except Exception as e:
            logger.error(f"Error getting session list for {user_id}: {str(e)}")
            return []

    # 检查会话是否存在
    def check_session_exist(self, user_id: str, session_id: str) -> bool:
        """检查会话是否存在，优先查 Redis，失败则查 MySQL"""
        rst = self.redis_manager.get_session(user_id, session_id)
        logger.info(f"check_session_exist: redis  rst {rst}")
        if rst:  # 优先检查 Redis 中的会话元数据
            logger.info(f"check_session_exist: session exists in Redis")
            return True

        return self.history_table.check_session_exist(user_id, session_id)
        
    # 添加聊天记录
    def append_history_record(self, user_id: str, session_id: str, role: str, content: str) -> bool:
        """添加聊天记录，同时存储到 MySQL 和 Redis"""
        try:
            # 存储到 MySQL
            mysql_success = self.history_table.append_history_record(user_id, session_id, role, content)
            if not mysql_success:
                logger.error(f"Failed to append history to MySQL for {user_id}/{session_id}")
                return False

            # 存储到 Redis
            item = json.dumps({"role": role, "content": content, "created_at": datetime.now().isoformat()})
            redis_success = self.redis_manager.append_to_list(user_id, session_id, item)
            if not redis_success:
                logger.warning(f"Failed to append history to Redis for {user_id}/{session_id}, proceeding with MySQL only")

            logger.info(f"History record added for {user_id}/{session_id}")
            return True
        except Exception as e:
            logger.error(f"Error appending history for {user_id}/{session_id}: {str(e)}")
            return False
        
    # 获取最近 N 轮历史记录
    def get_history_record(self, user_id: str, session_id: str, n: int = 3) -> List[Dict]:
        """获取最近 N 轮历史记录，优先从 Redis 获取，失败则从 MySQL 获取并更新 Redis"""
        try:
            # 从 Redis 获取
            history_items = self.redis_manager.get_list_all(user_id, session_id)
            if history_items:
                history = [json.loads(item) for item in history_items]
                result = history[-n * 2:]  # 取最近 N 轮（假设一轮包含用户和助手两条消息）
                logger.debug(f"Loaded history from Redis for {user_id}/{session_id}")
                return result #获取最后 2*n 条数据，顺序是正常顺序

            # Redis 失败或为空，从 MySQL 获取并更新 Redis
            logger.debug(f"Loading history from MySQL for {user_id}/{session_id}")
            history = self.history_table.get_history_record(user_id, session_id, n)
            if history:
                for record in history:
                    item = json.dumps({
                        "role": record["role"],
                        "content": record["content"],
                        "created_at": record["created_at"].isoformat()
                    })
                    self.redis_manager.append_to_list(user_id, session_id, item)
                logger.info(f"Updated Redis with history for {user_id}/{session_id}")
            return history
        except Exception as e:
            logger.error(f"Error getting history for {user_id}/{session_id}: {str(e)}")
            return []
        
    # 获取会话的所有历史记录
    def get_session_all_history_record(self, user_id: str, session_id: str) -> List[Dict]:
        """获取会话的所有历史记录，优先从 Redis 获取，失败则从 MySQL 获取并更新 Redis"""
        try:
            # 从 Redis 获取
            history_items = self.redis_manager.get_list_all(user_id, session_id)
            if history_items:
                history = [json.loads(item) for item in history_items]
                logger.debug(f"Loaded all history from Redis for {user_id}/{session_id}")
                return history

            # Redis 失败或为空，从 MySQL 获取并更新 Redis
            logger.debug(f"Loading all history from MySQL for {user_id}/{session_id}")
            history = self.history_table.get_session_all_history_record(user_id, session_id)
            if history:
                for record in history:
                    item = json.dumps({
                        "role": record["role"],
                        "content": record["content"],
                        "created_at": record["created_at"].isoformat()
                    })
                    self.redis_manager.append_to_list(user_id, session_id, item)
                logger.info(f"Updated Redis with all history for {user_id}/{session_id}")
            return history
        except Exception as e:
            logger.error(f"Error getting all history for {user_id}/{session_id}: {str(e)}")
            return []
        
    # 修改会话名称
    def update_session_name(self, user_id: str, session_id: str, session_name: str) -> bool:
        """修改会话名称，同时更新 MySQL 和 Redis"""
        try:
            # 更新 MySQL
            mysql_success = self.history_table.update_session_name(user_id, session_id, session_name)
            if not mysql_success:
                logger.error(f"Failed to update session name in MySQL for {user_id}/{session_id}")
                return False

            # 更新 Redis
            session_data = self.redis_manager.get_session(user_id, session_id)
            if session_data:
                session_data["session_name"] = session_name
                session_data["updated_at"] = datetime.now().isoformat()
                redis_success = self.redis_manager.set_session(user_id, session_id, session_data)
                if not redis_success:
                    logger.warning(f"Failed to update session name in Redis for {user_id}/{session_id}, proceeding with MySQL only")

            logger.info(f"Session name updated to '{session_name}' for {user_id}/{session_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating session name for {user_id}/{session_id}: {str(e)}")
            return False
        

logger.debug("===================start up step to HistoryDBControl")
# 对数据库连接处理进行实例化。        
db_control = HistoryDBControl()