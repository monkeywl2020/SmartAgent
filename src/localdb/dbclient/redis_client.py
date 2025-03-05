
import json
from datetime import datetime
from redis import StrictRedis, ConnectionPool
from typing import List, Dict, Optional
import os

from loguru import logger

# my_redis_nodes = [
#     {"host": "39.105.154.18", "port": "7001"},
#     {"host": "39.105.154.18", "port": "7002"},
#     {"host": "39.105.154.18", "port": "7003"},
#     {"host": "39.105.154.18", "port": "7004"},
#     {"host": "39.105.154.18", "port": "7005"},
#     {"host": "39.105.154.18", "port": "7006"},
# ]

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
#print(a,flush=True)
b = os.path.dirname(a)  #返回上一级目录部分，去掉文件名  dbclient目录
#print(b,flush=True)
#sys.path.append(b)
c = os.path.dirname(b) #返回上一级目录部分 localdb目录
#print(c,flush=True)

d = os.path.dirname(c) #返回上一级目录部分  src目录
print(d,flush=True)


project_root = d
config_path = os.path.join(project_root, "configs", "db_conf.json")
print(f"redis_client -->  Config path: {config_path}")

class RedisManager:
    def __init__(self):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                db_config = json.load(f)
            logger.debug(f"Loaded config from {config_path}")
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            raise FileNotFoundError(f"Config file {config_path} not found")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {config_path} as JSON")
            raise json.JSONDecodeError(f"Invalid JSON format in {config_path}")
        except Exception as e:
            logger.error(f"Unexpected error reading {config_path}: {str(e)}")
            raise Exception(f"Error reading {config_path}: {str(e)}")

        try:
            host = db_config["redis_host"]
            port = db_config["redis_port"]
            password = db_config["redis_pwd"] if db_config["redis_pwd"] else None
        except KeyError as e:
            logger.error(f"Missing required key in config: {str(e)}")
            raise KeyError(f"Config missing required key: {str(e)}")

        try:
            logger.debug("===================start up step to RedisClient")
            logger.debug(f"redis--------connecting to {host}:{port}")
            pool = ConnectionPool(host=host, port=port, db=0, password=password)
            self.redisClient = StrictRedis(connection_pool=pool, decode_responses=True)

            # self.redisClient = RedisCluster(startup_nodes=my_redis_nodes, password=password, decode_responses=True)

            self.redisClient.ping()
            logger.debug(f"[Success] redis {host}:{port} 连接成功")
            self.connected = True
        except Exception as e:
            logger.warning(f"[Failed] redis {host}:{port} 连接失败")
            logger.error(e)
            self.connected = False


    def is_connected(self):
        # print(f"redis--------is_connected: {self.connected}")
        return self.connected
    
    # 生成聊天记录的 key
    def _make_history_key(self, user_id: str, session_id: str) -> str:
        """生成聊天记录的键"""
        logger.info(f"nuwa:history:{user_id}:{session_id}")
        return f"nuwa:history:{user_id}:{session_id}"

    # 生成会话元数据的键
    def _make_session_key(self, user_id: str, session_id: str) -> str:
        """生成会话元数据的键"""
        logger.info(f"nuwa:session:{user_id}:{session_id}")
        return f"nuwa:session:{user_id}:{session_id}"

    # 生成会话列表的键
    def _make_sessions_key(self, user_id: str) -> str:
        """生成会话列表的键"""
        logger.info(f"nuwa:sessions:{user_id}")
        return f"nuwa:sessions:{user_id}"

    # 列表是否存在，list_id 是整个表的键值字符串 nuwa:history:user_id:session_id
    # redis_manager.list_exist("nuwa:history:user123:session456")
    def list_exist(self, user_id: str, session_id: str) -> bool:
        """检查聊天记录是否存在"""
        if not self.connected:
            return False
        key = self._make_history_key(user_id, session_id)
        try:
            # 检查列表的长度是否为 0
            if self.redisClient.llen(key) == 0:
                return False  # 如果列表为空，认为没有有效数据
            
            return True
        except Exception as e:
            logger.error(f"Failed to check list existence {key}: {str(e)}")
            return False
    
    def list_delete(self, user_id: str, session_id: str) -> bool:
        """删除聊天记录"""
        if not self.connected:
            return False
        key = self._make_history_key(user_id, session_id)
        try:
            return self.redisClient.delete(key) > 0
        except Exception as e:
            logger.error(f"Failed to delete list {key}: {str(e)}")
            return False
        
    # 添加到列表
    def append_to_list(self, user_id: str, session_id: str, item: str) -> bool:
        """添加一条聊天记录"""
        if not self.connected:
            return False
        key = self._make_history_key(user_id, session_id)
        try:
            with self.redisClient.pipeline() as pipe:
                pipe.rpush(key, item)
                pipe.expire(key, 604800)  # 7 天过期
                pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Failed to append to list {key}: {str(e)}")
            return False
            
    # 获取列表所有item
    def get_list_all(self, user_id: str, session_id: str) -> List[str]:
        """获取所有聊天记录"""
        if not self.connected:
            return []
        key = self._make_history_key(user_id, session_id)
        try:
            return self.redisClient.lrange(key, 0, -1)
        except Exception as e:
            logger.error(f"Failed to get list {key}: {str(e)}")
            return []

    #存储会话元数据到 Redis ，根据 user_id ，session_id 存储 session_data
    def set_session(self, user_id: str, session_id: str, session_data: Dict) -> bool:
        """存储会话元数据到 Redis"""
        if not self.connected:
            return False
        session_key = self._make_session_key(user_id, session_id) # 存放会话元数据，例如会话名称。
        sessions_key = self._make_sessions_key(user_id)           # 存放会话的所有的 session id
        logger.info(f"RedisManager::set_session-------- session_key:{session_key} sessions_key:{sessions_key}")
        try:
            with self.redisClient.pipeline() as pipe:
                pipe.hset(session_key, mapping=session_data)
                pipe.expire(session_key, 604800)
                pipe.sadd(sessions_key, session_id)
                pipe.expire(sessions_key, 604800)
                pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Failed to set session {session_key}: {str(e)}")
            return False

    def get_session(self, user_id: str, session_id: str) -> Dict:
        """从 Redis 获取会话元数据"""
        if not self.connected:
            return {}
        key = self._make_session_key(user_id, session_id)
        try:
            return self.redisClient.hgetall(key) or {}
        except Exception as e:
            logger.error(f"Failed to get session {key}: {str(e)}")
            return {}

    def get_sessions(self, user_id: str) -> List[str]:
        """从 Redis 获取用户的所有会话 ID"""
        if not self.connected:
            return []
        key = self._make_sessions_key(user_id)
        try:
            return list(self.redisClient.smembers(key)) or []
        except Exception as e:
            logger.error(f"Failed to get sessions {key}: {str(e)}")
            return []

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除 Redis 中的会话数据"""
        if not self.connected:
            return False
        session_key = self._make_session_key(user_id, session_id)
        sessions_key = self._make_sessions_key(user_id)
        history_key = self._make_history_key(user_id, session_id)
        try:
            with self.redisClient.pipeline() as pipe:
                pipe.srem(sessions_key, session_id)
                if not self.redisClient.scard(sessions_key):
                    pipe.delete(sessions_key)
                pipe.delete(session_key)
                pipe.delete(history_key)
                pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_key}: {str(e)}")
            return False