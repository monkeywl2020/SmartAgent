import uuid
import ujson
import mmh3
import hashlib
from datetime import datetime
from typing import List, Dict

from loguru import logger

class HistoryContentTable:
    def __init__(self, execute_query):
        self.execute_query = execute_query
        self.table_count = 100 # 分表数量，可根据需求调整
        self.create_tables()

    def create_tables(self):
        """创建 chat_sessions 和 chat_history_{index} 表"""
        #----------------------------------------------------
        # 创建 chat_sessions 表：
        #      这个表根据 user_id 和 session_id 可以展示该session的概览
        #      可以根据 user_id 索引快速找到所有的表项 ，创建的时候需要传入时间，记录表格创建时间
        #----------------------------------------------------
        sessions_query = """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                user_id VARCHAR(255) NOT NULL,
                session_id VARCHAR(255) NOT NULL,
                session_name VARCHAR(255),
                session_type INT DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, session_id),
                INDEX idx_user (user_id)
            );
        """
        self.execute_query(sessions_query, (), commit=True)
        logger.info("Created table: chat_sessions")

        #----------------------------------------------------
        # 创建 chat_history_{index} 分表
        #   下面是分表，主要表多了mysql性能不够，所以表的数量要限制，初始化设置为50个表。
        #   根据  get_table_name 的时候 根据 table_index 来查找所使用的表名 chat_history_{i}
        #----------------------------------------------------
        for i in range(self.table_count):
            table_name = f"chat_history_{i}"
            history_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    session_id VARCHAR(255) NOT NULL,
                    msg_id INT AUTO_INCREMENT,
                    user_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    PRIMARY KEY (session_id, msg_id),
                    UNIQUE (msg_id),
                    INDEX idx_user_session (user_id, session_id),
                    INDEX idx_time (created_at)
                );
            """
            self.execute_query(history_query, (), commit=True)
            logger.info(f"Created table: {table_name}")

    def get_table_name(self, user_id):
        """根据 user_id 计算分表名"""
        hash_value = mmh3.hash(user_id)
        table_index = abs(hash_value) % self.table_count
        return f"chat_history_{table_index}"

    # 获取用户会话列表
    def get_session_list(self, user_id: str) -> List[Dict]:
        """获取用户的会话列表"""
        query = """
            SELECT session_id, session_name, session_type, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
        """
        result = self.execute_query(query, (user_id,), fetch=True)
        return [
            {
                "session_id": row[0],
                "session_name": row[1],
                "session_type": row[2],
                "created_at": row[3],
                "updated_at": row[4]
            } for row in result
        ] if result else []

    # 添加新会话
    def add_new_session(self, user_id: str, session_id: str, session_name: str = "新会话", session_type: int = 0) -> bool:
        """添加新会话"""
        created_at = datetime.now()
        
        #为了性能和简洁性，INSERT IGNORE 替代了显式检查。
        query = """
            INSERT IGNORE INTO chat_sessions (user_id, session_id, session_name, session_type, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            self.execute_query(query, (user_id, session_id, session_name, session_type, created_at), commit=True)
            logger.info(f"Added new session {session_id} for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add session {session_id} for {user_id}: {str(e)}")
            return False

    # 删除会话及其聊天记录
    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除会话及其聊天记录"""
        try:
            # 删除 chat_sessions 中的会话
            session_query = "DELETE FROM chat_sessions WHERE user_id = %s AND session_id = %s"
            self.execute_query(session_query, (user_id, session_id), commit=True)

            # 删除 chat_history_{index} 中的记录
            table_name = self.get_table_name(user_id)
            history_query = f"DELETE FROM {table_name} WHERE user_id = %s AND session_id = %s"
            self.execute_query(history_query, (user_id, session_id), commit=True)

            logger.info(f"Deleted session {session_id} for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id} for {user_id}: {str(e)}")
            return False

    # 检查会话是否存在
    def check_session_exist(self, user_id: str, session_id: str) -> bool:
        """检查会话是否存在"""
        query = "SELECT 1 FROM chat_sessions WHERE user_id = %s AND session_id = %s LIMIT 1"
        result = self.execute_query(query, (user_id, session_id), fetch=True)
        return bool(result)

    #添加会话的第一条消息（或任何消息）
    def add_session_msg(self, user_id: str, session_id: str, role: str, content: str) -> bool:
        """添加会话的第一条消息（或任何消息）"""
        if not self.check_session_exist(user_id, session_id):
            logger.warning(f"Session {session_id} for {user_id} does not exist, cannot add message")
            return False

        table_name = self.get_table_name(user_id)
        query = f"""
            INSERT INTO {table_name} (session_id, user_id, role, content, created_at)
            VALUES (%s, %s, %s, %s, NOW())
        """
        try:
            self.execute_query(query, (session_id, user_id, role, content), commit=True)
            logger.info(f"Added message to {table_name} for {user_id}/{session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add message for {user_id}/{session_id}: {str(e)}")
            return False

    #修改会话名称
    def update_session_name(self, user_id: str, session_id: str, session_name: str) -> bool:
        """修改会话名称"""
        query = """
            UPDATE chat_sessions
            SET session_name = %s
            WHERE user_id = %s AND session_id = %s
        """
        try:
            self.execute_query(query, (session_name, user_id, session_id), commit=True)
            logger.info(f"Updated session name to '{session_name}' for {user_id}/{session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update session name for {user_id}/{session_id}: {str(e)}")
            return False

    # 获取最近 N 轮历史记录
    def get_history_record(self, user_id: str, session_id: str, n: int = 3) -> List[Dict]:
        """获取最近 N 轮历史记录"""
        table_name = self.get_table_name(user_id)
        query = f"""
            SELECT role, content, created_at
            FROM {table_name}
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """
        result = self.execute_query(query, (user_id, session_id, n * 2), fetch=True)
        return [
            {"role": row[0], "content": row[1], "created_at": row[2]}
            for row in result[::-1]  # 反转恢复正序
        ] if result else []

    # 添加聊天记录（与 add_session_msg 相同，保留兼容性）
    def append_history_record(self, user_id: str, session_id: str, role: str, content: str) -> bool:
        """添加聊天记录（与 add_session_msg 相同，保留兼容性）"""
        return self.add_session_msg(user_id, session_id, role, content)

    # 获取会话的所有历史记录
    def get_session_all_history_record(self, user_id: str, session_id: str) -> List[Dict]:
        """获取会话的所有历史记录"""
        table_name = self.get_table_name(user_id)
        query = f"""
            SELECT role, content, created_at
            FROM {table_name}
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at ASC
        """
        result = self.execute_query(query, (user_id, session_id), fetch=True)
        return [
            {"role": row[0], "content": row[1], "created_at": row[2]}
            for row in result
        ] if result else []
