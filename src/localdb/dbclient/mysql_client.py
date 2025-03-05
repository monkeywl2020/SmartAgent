import pymysql
import json
import os

from dbutils.pooled_db import PooledDB
from loguru import logger

from .history_content_mysql_table import HistoryContentTable

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
print(f"mysql_client -->  Config path: {config_path}")

class MysqlManagerMain:
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

        # 读取配置获取配置信息
        #------------------------------------
        # 从配置中提取 MySQL 参数
        try:
            host = db_config["mysql_host"]
            port = db_config["mysql_port"]
            user = db_config["mysql_user"]
            password = db_config["mysql_pwd"]
            database = db_config["mysql_database"]
        except KeyError as e:
            logger.error(f"Missing required key in config: {str(e)}")
            raise KeyError(f"Config missing required key: {str(e)}")
        #------------------------------------

        try:
            logger.debug("===================start up step to MysqlClient")
            logger.debug(f"mysql--------connecting to {host}:{port}")
            self.check_database_(host, port, user, password, database)
            dbconfig = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "db": database,
                "connect_timeout": 3,
                "read_timeout": 10,
                "write_timeout": 10
            }
            # maxcached=20 最大缓存连接数，表示可以缓存的连接数量。
            # maxshared=0: 最大共享连接数，通常为 0 表示没有共享连接。
            # maxconnections=100: 最大连接数，表示最大可以同时打开的连接数。
            # maxusage=20: 每个连接最大可以使用的次数。
            # 通过 PooledDB 创建连接池，使得多个数据库操作可以复用连接，减少创建和销毁连接的开销。
            # dbconfig 中存储了连接 MySQL 所需的配置，包括数据库的主机、端口、用户名、密码、数据库名以及超时配置。
            self.cnxpool = PooledDB(pymysql, maxcached=20, maxshared=0, maxconnections=100, maxusage=20, **dbconfig)
            
            logger.debug(f"[Success] mysql数据库{database} {host}:{port}连接成功")
            self.connected = True
        except Exception as e:
            logger.warning(f"[Failed] mysql数据库{database} {host}:{port}连接失败")
            logger.error(e)
            self.connected = False

    def is_connected(self):
        return self.connected

    def check_database_(self, host, port, user, password, database_name):
        print(f"mysql server: host={host} port={port} user={user} passwd={password}")
        cnx = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        cursor = cnx.cursor()
        cursor.execute('SHOW DATABASES')
        databases = [database[0] for database in cursor]
        
        # 如果数据集不存在，则创建数据库数据集
        if database_name not in databases:
            cursor.execute(f'CREATE DATABASE IF NOT EXISTS {database_name}')
            logger.debug(f"数据库{database_name}新建成功或已存在")
        logger.debug(f"[SUCCESS] 数据库{database_name}检查通过")
        cursor.close()
        cnx.database = database_name
        cnx.close()

    def execute_query_(self, query, params, commit=False, fetch=False):
        conn = None
        cursor = None
        try:
            conn = self.cnxpool.connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            if commit:
                conn.commit()
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
        except Exception as err:
            logger.error(f"Error: {err}")
            result = None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return result

    
    # 其他的公用方法可以保留在这里...
