import sys
import os
import asyncio
from typing import Generator

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
print(a,flush=True)
b = os.path.dirname(a)  #返回上一级目录部分，去掉文件名
print(b,flush=True)
#sys.path.append(b)
c = os.path.dirname(b) #返回上一级目录部分
print(c,flush=True)

d = os.path.dirname(c) #返回上一级目录部分
print(d,flush=True)

# 将上一级目录加入 添加到搜索路径中也 就是examples的上级目录
sys.path.append(d)
print(sys.path,flush=True)

from src.localdb.history_db_control import HistoryDBControl

# 主函数
def main():
    # 创建实例
    db_control1 = HistoryDBControl()

    # 创建会话
    db_control1.create_session("wl_test", "session_wl_session1", "测试会话")

    # 添加聊天记录
    db_control1.append_history_record("wl_test", "session_wl_session1", "user", "你好")
    db_control1.append_history_record("wl_test", "session_wl_session1", "assistant", "你好！有什么我可以帮助你的？")

    # 获取最近 2 轮历史记录
    history = db_control1.get_history_record("wl_test", "session_wl_session1", 2)
    print("Recent history:", history)

    # 获取会话列表
    rst = db_control1.check_session_exist("wl_test","session_wl_session1")
    print (f"check_session_exist:wl_test, session_wl_session1 =  rst {rst}")
    sessions = db_control1.get_session_list("wl_test")
    print("Session list:", sessions)

    # 删除会话
    #db_control.delete_session("user123", "session456")

if __name__ == "__main__":
    main()