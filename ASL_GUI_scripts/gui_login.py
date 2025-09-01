import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
# import bcrypt  # 用于密码哈希
# import gui_game as gui  # 导入游戏界面的模块
import gui_selection as gui

from gui_database import *

class LoginApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("400x420")

        # 初始化数据库
        init_db()

        # 将窗口居中
        self.center_window()

        # 创建界面
        self.setup_ui()

    def center_window(self):
        """将窗口居中"""
        # 获取屏幕尺寸
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # 计算窗口位置
        x = (screen_width - 400) // 2
        y = (screen_height - 420) // 2

        # 设置窗口位置和尺寸
        self.window.geometry(f"400x420+{x}+{y}")

    def setup_ui(self):
        """设置登录界面布局"""
        # 登录界面
        self.label_username = ttk.Label(self.window, text="Username:", font=("Arial", 12))
        self.label_username.pack(pady=10)

        self.entry_username = ttk.Entry(self.window, width=30)
        self.entry_username.pack(pady=5)

        self.label_password = ttk.Label(self.window, text="Password:", font=("Arial", 12))
        self.label_password.pack(pady=10)

        self.entry_password = ttk.Entry(self.window, width=30, show="*")
        self.entry_password.pack(pady=5)

        self.btn_login = ttk.Button(self.window, text="Login", command=self.login)
        self.btn_login.pack(pady=20)

        # 注册界面
        self.label_new_username = ttk.Label(self.window, text="New Username:", font=("Arial", 12))
        self.label_new_username.pack(pady=10)

        self.entry_new_username = ttk.Entry(self.window, width=30)
        self.entry_new_username.pack(pady=5)

        self.label_new_password = ttk.Label(self.window, text="New Password:", font=("Arial", 12))
        self.label_new_password.pack(pady=10)

        self.entry_new_password = ttk.Entry(self.window, width=30, show="*")
        self.entry_new_password.pack(pady=5)

        self.btn_register = ttk.Button(self.window, text="Register", command=self.register)
        self.btn_register.pack(pady=10)

    def login(self):
        """登录逻辑"""
        username = self.entry_username.get()
        password = self.entry_password.get()

        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password.")
            return

        if verify_user(username, password):
            messagebox.showinfo("Login Success", "Login successful! Redirecting to the game...")
            self.window.destroy()  # 关闭登录窗口
            self.open_game(username)  # 打开游戏界面
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def register(self):
        """注册逻辑"""
        username = self.entry_new_username.get()
        password = self.entry_new_password.get()

        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password.")
            return

        if register_user(username, password):
            messagebox.showinfo("Register Success", "Registration successful! You can now login.")
        else:
            messagebox.showerror("Register Failed", "Username already exists.")

    def open_game(self,username):
        game_window = tk.Tk()
        app = gui.MainApp(game_window, "Main Menu",username)
        game_window.mainloop()

# 创建主窗口
root = tk.Tk()
app = LoginApp(root, "Login and Register")

# 运行主循环
root.mainloop()