import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import random
import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import gui_database as database  # 导入数据库模块
# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # 用于绘制手部关键点
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

model_digit = tf.keras.models.load_model('model/hand_gesture_model_10_digits_update.h5')
model_word = tf.keras.models.load_model('model/hand_gesture_model_10_words_update.h5')
digit_gesture_labels = ['0','1', '2', '3', '4', '5','6', '7', '8','9']
digit_label_encoder = LabelEncoder()
digit_label_encoder.fit(digit_gesture_labels)
word_gesture_labels = ["airplane", "father", "hello", "I'm", "love", "mother", "ok", "sorry", "water", "yes"]
word_label_encoder = LabelEncoder()
word_label_encoder.fit(word_gesture_labels)

def extract_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    return np.array(landmarks).flatten()

def predict_gesture(frame, type="digit"):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if type == "digit":
        model = model_digit
        label_encoder = digit_label_encoder
    else:
        model = model_word
        label_encoder = word_label_encoder

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            landmarks = landmarks.reshape(1, -1)

            # Measure latency
            start_time = time.time()
            prediction = model.predict(landmarks)
            latency = time.time() - start_time

            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
            label_index = np.argmax(prediction)
            return predicted_label[0], latency, label_index

    return "No hand detected", 0, 0
class WhackAMoleGameApp:
    def __init__(self, window, window_title,width,height,username):
        self.window = window
        self.window.title(window_title)

        # 设置窗口尺寸
        self.width = width
        self.height = height
        self.user_id = database.get_user_id(username)  # 获取用户 ID
        self.username = username
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Cannot open camera")

        # 加载图片资源
        self.mole_image = Image.open("animal/mouse.png")  # 地鼠图片
        self.mole_image = self.mole_image.resize((100, 100), Image.Resampling.LANCZOS)
        self.mole_photo = ImageTk.PhotoImage(self.mole_image)

        self.cat_images = []
        for i in range(1, 10):
            cat_image = Image.open("animal/cat.png")  # 猫的图片
            cat_image = cat_image.resize((100, 100), Image.Resampling.LANCZOS)
            self.cat_images.append(ImageTk.PhotoImage(cat_image))

        # 游戏变量
        self.whack_mole_positions = [(i % 5 * 120 + 60, i // 5 * 120 + 60) for i in range(10)]  # 10 个位置
        self.whack_mole_labels = digit_gesture_labels #[i for i in range(10)]  # 地鼠标签（0 到 9）
        self.current_mole_index = None  # 当前地鼠的位置索引
        self.current_mole_label = None  # 当前地鼠的标签
        self.score = 0
        self.game_start_time = None
        self.game_duration = 60  # 游戏时间（秒）
        self.cursor_pos = (0, 0)  # 手势光标位置
        self.gesture_label = None  # 手势识别结果

        # 将窗口居中
        self.center_window()

        # 布局
        self.setup_ui()

        # 更新视频帧
        self.update()

        # 窗口关闭事件
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def center_window(self):
        """将窗口居中"""
        # 获取屏幕尺寸
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # 计算窗口位置
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2

        # 设置窗口位置和尺寸
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def setup_ui(self):
        """设置界面布局"""
        self.middle_left_frame = ttk.Frame(self.window, padding="10")
        self.middle_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.game_canvas = tk.Canvas(self.middle_left_frame, width=600, height=280, bg="white")
        self.game_canvas.pack()

        self.username_label = ttk.Label(self.middle_left_frame, text="Username: "+self.username, font=("Arial", 16))
        self.username_label.pack(pady=10)

        self.score_label = ttk.Label(self.middle_left_frame, text="Score: 0", font=("Arial", 16))
        self.score_label.pack(pady=10)

        self.timer_label = ttk.Label(self.middle_left_frame, text="Time: 60", font=("Arial", 16))
        self.timer_label.pack(pady=10)

        self.btn_start = ttk.Button(self.middle_left_frame, text="Start Game", width=30, command=self.start_game)
        self.btn_start.pack(pady=10)

        # 中间右：视频手势识别界面
        self.middle_right_frame = ttk.Frame(self.window, padding="10")
        self.middle_right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(self.middle_right_frame, width=600, height=430)
        self.video_canvas.pack()

    def start_game(self):
        """开始打地鼠游戏"""
        self.score = 0
        self.game_start_time = time.time()
        self.update_game()
        self.btn_start.config(state=tk.DISABLED)

    def update_game(self):
        """更新打地鼠游戏状态"""
        if self.game_start_time is None:
            return

        # 计算剩余时间
        elapsed_time = time.time() - self.game_start_time
        remaining_time = max(0, self.game_duration - int(elapsed_time))
        self.timer_label.config(text=f"Time: {remaining_time}")

        # 检查游戏是否结束
        if remaining_time <= 0:
            messagebox.showinfo("Game End", f"End！Your score: {self.score}")
            self.btn_start.config(state=tk.NORMAL)
            database.save_game_record(self.user_id, self.score)
            return

        # 每 3 秒刷新地鼠位置和标签
        if self.current_mole_index is None or elapsed_time % 3 < 0.1:
            self.current_mole_index = random.randint(0, 9)
            self.current_mole_label = self.whack_mole_labels[self.current_mole_index]
            # print(self.current_mole_index)

        # 绘制地鼠和猫
        self.game_canvas.delete("all")
        for i, pos in enumerate(self.whack_mole_positions):
            if i == self.current_mole_index:
                self.game_canvas.create_image(pos[0], pos[1], image=self.mole_photo)  # 绘制地鼠
                self.game_canvas.create_text(pos[0], pos[1] + 60, text=f"Label: {self.current_mole_label}", font=("Arial", 12))
            else:
                self.game_canvas.create_image(pos[0], pos[1], image=self.cat_images[i % 9])  # 绘制猫

        # 检查是否击中地鼠
        if self.gesture_label is not None and self.gesture_label == self.current_mole_index:
            self.score += 1
            self.score_label.config(text=f"Score: {self.score}")
            self.current_mole_index = random.randint(0, 9)  # 立即刷新地鼠位置
            self.current_mole_label = self.whack_mole_labels[self.current_mole_index]

        # 每隔 100ms 更新一次游戏
        self.window.after(10, self.update_game)

    def detect_and_draw_hand(self, frame):
        """检测手部关键点并绘制手的轮廓"""
        # 将帧转换为 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 使用 MediaPipe 检测手部关键点
        results = hands.process(rgb_frame)
        predicted_label, latency, index = predict_gesture(frame)
        self.gesture_label = index
        print("Predicted:", str(index))
        print("current model index:", str(self.current_mole_index))
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 如果检测到手
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点和连接线
                mp_drawing.draw_landmarks(
                    frame,  # 图像帧
                    hand_landmarks,  # 手部关键点
                    mp_hands.HAND_CONNECTIONS,  # 手部连接线
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 关键点样式
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # 连接线样式
                )

                # 获取手腕位置（用于光标）
                wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                self.cursor_pos = (wrist_x, wrist_y)


    def update(self):
        """更新视频帧"""
        ret, frame = self.cap.read()
        if ret:
            # 检测并绘制手的轮廓
            self.detect_and_draw_hand(frame)


            # 显示视频帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.image = imgtk

        # 每隔 10ms 更新一次
        self.window.after(10, self.update)

    def on_close(self):
        """关闭窗口时释放资源"""
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


# 创建主窗口
# root = tk.Tk()
# app = WhackAMoleGameApp(root, "Whack-A-Mole Game", 1200, 500,'jindi')
# #
# # # 运行主循环
# root.mainloop()