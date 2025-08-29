# database.py
import sqlite3
from datetime import datetime
import json
# import bcrypt  # 用于密码哈希

# 初始化数据库
def init_db():
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 创建用户表
    c.execute('''CREATE TABLE IF NOT EXISTS Users (
                 user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL)''')

    # 创建quiz表
    c.execute('''CREATE TABLE IF NOT EXISTS LearningRecords (
                 record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 learning_mode TEXT NOT NULL,
                 initial_assessment TEXT NOT NULL,
                 learning_session TEXT NOT NULL,
                 learning_order TEXT NOT NULL,
                 final_assessment TEXT NOT NULL,
                 initial_correct INTEGER NOT NULL,
                 final_correct INTEGER NOT NULL,
                 improvement INTEGER NOT NULL,
                 FOREIGN KEY (user_id) REFERENCES Users (user_id))''')

    # 创建游戏记录表
    c.execute('''CREATE TABLE IF NOT EXISTS GameRecords (
                 record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 score INTEGER NOT NULL,
                 mode TEXT NOT NULL,
                 FOREIGN KEY (user_id) REFERENCES Users (user_id))''')

    conn.commit()
    conn.close()

# 注册用户
def register_user(username, password):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 检查用户名是否已存在
    c.execute("SELECT username FROM Users WHERE username = ?", (username,))
    if c.fetchone():
        conn.close()
        return False  # 用户名已存在

    # 对密码进行哈希处理
    hashed_password = password #bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # 插入新用户
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO Users (username, password) VALUES (?, ?)",
              (username, hashed_password, created_at))
    conn.commit()
    conn.close()
    return True

# 验证用户登录
def verify_user(username, password):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 查询用户信息
    c.execute("SELECT password FROM Users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        # 验证密码
        hashed_password = result[0]
        return password == hashed_password #bcrypt.checkpw(password.encode("utf-8"), hashed_password)
    return False

# 获取用户 ID
def get_user_id(username):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 查询用户 ID
    c.execute("SELECT user_id FROM Users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        return result[0]
    return None

# 保存学习记录
def save_learning_record(user_id, learning_mode, initial_assessment, learning_session, learning_order, final_assessment,initial_correct,final_correct,improvement):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()
    data = {
        "user_id": user_id,  # 必须是整数
        "learning_mode": learning_mode,  # 必须是字符串
        "initial_assessment": initial_assessment,
        "learning_session": learning_session,
        "learning_order": learning_order,
        "final_assessment": final_assessment,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "improvement": improvement,
        # "played_at": datetime.now().isoformat()  # 转换为ISO格式字符串
    }
    # 插入游戏记录
    # played_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO LearningRecords (user_id, learning_mode, initial_assessment, learning_session, learning_order, final_assessment,initial_correct,final_correct,improvement) VALUES (?, ?, ?, ?,?,?,?,?,?)",
              (
                  data["user_id"],
                  data["learning_mode"],
                  json.dumps(data["initial_assessment"]),
                  json.dumps(data["learning_session"]),
                  json.dumps(data["learning_order"]),  # 列表转JSON字符串
                  json.dumps(data["final_assessment"]),
                  data["initial_correct"],
                  data["final_correct"],
                  data["improvement"]
                  # data["played_at"]
              ))
    conn.commit()
    conn.close()

# 保存游戏记录
def save_game_record(user_id, score, mode):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 插入游戏记录
    played_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO GameRecords (user_id, score, mode) VALUES (?, ?, ?)",
              (user_id, score, mode))
    conn.commit()
    conn.close()

# 查询用户游戏记录
def get_game_records(user_id):
    conn = sqlite3.connect("game_data.db")
    c = conn.cursor()

    # 查询游戏记录
    c.execute("SELECT score FROM GameRecords WHERE user_id = ?", (user_id,))
    records = c.fetchall()
    conn.close()
    return records

# def save_initial_quiz_record(user_id,):
#     conn = sqlite3.connect("game_data.db")
#     c = conn.cursor()
#     played_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     c.execute("INSERT INTO GameRecords (user_id, score, played_at) VALUES (?, ?, ?)",
#               (user_id, score, played_at))
#     conn.commit()
#     conn.close()