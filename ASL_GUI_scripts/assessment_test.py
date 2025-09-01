import tkinter as tk
from tkinter import ttk, messagebox
import random
import threading
from openai import OpenAI
from IPython.display import Image, display, Audio, Markdown
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import time
import subprocess
import mediapipe as mp
from PIL.ImageDraw import ImageDraw
from PIL.ImageFont import ImageFont
from sklearn.preprocessing import LabelEncoder
import gui_database as database  # 假设这是你的数据库模块
from gui_game import WhackAMoleGameApp

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

digit_json_file_path = 'pairs_data/pair_digit.json'
word_json_file_path = 'pairs_data/pair_word.json'

with open(digit_json_file_path, 'r', encoding='utf-8') as json_file:
    digit_pair_data = json.load(json_file)
with open(word_json_file_path, 'r', encoding='utf-8') as json_file:
  word_pair_data = json.load(json_file)

def search_pair(gesture_pair, gesture_type="digit"):
    # Loop through all gesture pairs and search for the matching pair
    if gesture_type == "digit":
        gesture_data = digit_pair_data
    else:
        gesture_data = word_pair_data
    comparison = gesture_data.get(gesture_pair)
    if not comparison:
        return 'Pair not found'
    return {
      'differences': comparison['differences'],
      'correction_tips': comparison['correction_tips']
    }

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

    return "No hand detected", 0, -1
class ASLLearningApp:
    def __init__(self, root,username='jindi'):
        self.username = username
        self.root = root
        self.root.title("ASL Learning Experiment")
        self.root.geometry("1200x800")
        self.width = 1200
        self.height = 800
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", padding=6, font=("Arial", 12))

        # Learning content
        self.digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.words = ["airplane", "father", "hello", "I'm", "love",
                      "mother", "ok", "sorry", "water", "yes"]
        self.all_items = self.digits + self.words

        # Timer variable to handle the 2-second gesture prediction window
        self.gesture_prediction_start_time = None
        self.correct_gesture = None  # Store the correct gesture for comparison

        # Colors
        self.colors = {
            "primary": "#4a6fa5",
            "secondary": "#166088",
            "accent": "#4fc3f7",
            "background": "#f0f0f0",
            "text": "#333333",
            "correct": "#4caf50",
            "incorrect": "#f44336"
        }

        # Show mnemonic
        self.mnemonics = {
            "0": "Looks like a zero - a closed circle with your hand",
            "1": "One finger up for number one",
            "2": "Two fingers make the shape of a 'V'",
            "3": "Three fingers up like the letter 'W'",
            "4": "Four fingers up, thumb tucked - like showing '4'",
            "5": "All five fingers spread out - high five!",
            "6": "Thumb touches pinky - think '6' as a phone hang-up gesture",
            "7": "Thumb touches ring finger - like the 'call me' hand sign",
            "8": "Thumb touches middle finger - like a gun shape",
            "9": "Thumb touches index finger - like making an 'O'",
            "airplane": "Hand looks like an airplane taking off",
            "father": "Think of a dad tipping his hat (hand at forehead)",
            "hello": "Universal waving gesture",
            "I'm": "Pointing to yourself - 'I am'",
            "love": "Hands crossed over heart - universal love symbol",
            "mother": "Think of a mom touching her chin (like a kiss)",
            "ok": "Standard 'OK' gesture",
            "sorry": "Rubbing chest - showing you feel it inside",
            "water": "Tapping chin then down - like water dripping",
            "yes": "Nodding fist - like a head nodding 'yes'"
        }

        # Experiment parameters
        self.learning_mode = None
        self.current_item = None
        self.learning_index = 0
        self.learning_order = []
        self.user_performance = {}
        self.initial_assessment_results = {}
        self.start_time = None
        self.assessment_items = []
        self.current_assessment_index = 0

        # practice parameters
        self.start_time = None  # Track when the timer starts
        self.hold_time = 2  # Hold for 2 seconds
        self.capture_frame_at = None  # Time when the frame should be captured
        self.is_recording = False  # To check if the sign is being held
        self.pre_pair = None  # To store the pair of signs for practice
        # Create image directory if not exists
        # if not os.path.exists("asl_images"):
        #     os.makedirs("asl_images")
        #     for item in self.all_items:
        #         os.makedirs(f"asl_images/{item}")
        self.center_window()
        # Setup GUI
        self.setup_welcome_screen()

    def center_window(self):
        """将窗口居中"""
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 计算窗口位置
        x = (screen_width - self.width) // 2
        y = (screen_height - self.height) // 2

        # 设置窗口位置和尺寸
        self.root.geometry(f"{self.width}x{self.height}+{x}+{y}")

    def setup_welcome_screen(self):
        """Initial screen explaining the experiment"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        welcome_frame = ttk.Frame(self.root, style="TFrame")
        welcome_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        tk.Label(welcome_frame,
                 text="ASL Learning Experiment",
                 font=("Arial", 28, "bold"),
                 foreground=self.colors["primary"],
                 bg=self.colors["background"]).pack(pady=20)

        tk.Label(welcome_frame,
                 text="You will first take a brief assessment test,\nthen learn ASL signs for digits 0-9 and 10 common words.",
                 font=("Arial", 16),
                 bg=self.colors["background"],
                 justify=tk.CENTER).pack(pady=20)

        tk.Button(welcome_frame,
                  text="Begin Assessment",
                  command=self.start_initial_assessment,
                  font=("Arial", 14),
                  bg=self.colors["accent"],
                  fg="white",
                  relief=tk.FLAT,
                  padx=20,
                  pady=10).pack(pady=40)

    def start_initial_assessment(self):
        """Begin the initial assessment test"""
        self.clear_screen()
        self.start_time = datetime.now()

        # Shuffle items for assessment
        self.assessment_items = random.sample(self.all_items, len(self.all_items))
        self.current_assessment_index = 0
        self.initial_assessment_results = {}

        # Initialize performance tracking
        for item in self.all_items:
            self.initial_assessment_results[item] = {
                "correct": False,
                "response_time": None,
                "attempts": 0
            }

        self.show_assessment_item()

    def show_assessment_item(self):
        """Display an assessment item with uniformly sized ASL images"""
        if self.current_assessment_index >= len(self.assessment_items):
            self.show_assessment_complete()
            return

        self.clear_screen()
        self.current_item = self.assessment_items[self.current_assessment_index]

        # Header
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame,
                 text="Assessment Test",
                 font=("Arial", 20, "bold"),
                 foreground=self.colors["primary"],
                 background=self.colors["background"]).pack(side=tk.LEFT)

        progress = f"Item {self.current_assessment_index + 1} of {len(self.assessment_items)}"
        tk.Label(header_frame,
                 text=progress,
                 font=("Arial", 12),
                 background=self.colors["background"]).pack(side=tk.RIGHT)

        # Question
        tk.Label(self.root,
                 text=f"Which image shows the ASL sign for: {self.current_item}? Please click it.",
                 font=("Arial", 18),
                 bg=self.colors["background"]).pack(pady=30)

        # Image options
        options_frame = tk.Frame(self.root, bg=self.colors["background"])
        options_frame.pack(pady=20)

        self.option_images = []
        self.option_buttons = []

        # Target display size for all images
        DISPLAY_SIZE = (250, 250)

        # Generate options - 1 correct, 3 distractors
        folder = "digit" if self.current_item.isdigit() else "word"
        extension = ".jpg"
        option = self.current_item
        self.options = [str(option),
                   f"assessment/{folder}/{option}/{option}-2{extension}",
                   f"assessment/{folder}/{option}/{option}-3{extension}",
                   f"assessment/{folder}/{option}/{option}-4{extension}"]
        random.shuffle(self.options)
        self.correct_option = self.options.index(self.current_item)
        print(f"Correct option index: {self.current_item}")  # Debugging line

        for i, option in enumerate(self.options):
            frame = tk.Frame(options_frame, bg=self.colors["background"])
            frame.pack(side=tk.LEFT, padx=15)

            try:
                # Load image with correct extension
                if option == str(self.current_item):
                    option = f"assessment/{folder}/{option}/{option}-1{extension}"

                img_path = option

                img = Image.open(img_path)

                # Resize to display size while maintaining aspect ratio
                img.thumbnail(DISPLAY_SIZE, Image.LANCZOS)

                # Create blank canvas of target size
                canvas = Image.new('RGB', DISPLAY_SIZE, (255, 255, 255))

                # Paste centered image
                x_offset = (DISPLAY_SIZE[0] - img.width) // 2
                y_offset = (DISPLAY_SIZE[1] - img.height) // 2
                canvas.paste(img, (x_offset, y_offset))

                img_tk = ImageTk.PhotoImage(canvas)

            except Exception as e:
                print(f"Error loading image {option}: {e}")
                # Create blank placeholder
                canvas = Image.new('RGB', DISPLAY_SIZE, (240, 240, 240))
                draw = ImageDraw.Draw(canvas)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                    text = f"{option}"
                    text_width = draw.textlength(text, font=font)
                    draw.text(
                        ((DISPLAY_SIZE[0] - text_width) // 2, DISPLAY_SIZE[1] // 2 - 10),
                        text,
                        fill="black",
                        font=font
                    )
                except:
                    pass
                img_tk = ImageTk.PhotoImage(canvas)

            self.option_images.append(img_tk)

            btn = tk.Button(frame,
                            image=img_tk,
                            # text=option,
                            compound=tk.TOP,
                            font=("Arial", 12),
                            bg="white",
                            relief=tk.FLAT,
                            command=lambda idx=i: self.process_assessment_response(idx))
            btn.pack()
            self.option_buttons.append(btn)

        # Start response timer
        self.item_start_time = time.time()

    def process_assessment_response(self, selected_index):
        """Handle user's response to assessment item"""
        response_time = time.time() - self.item_start_time
        is_correct = (selected_index == self.correct_option)

        # Record results
        self.initial_assessment_results[self.current_item] = {
            "correct": is_correct,
            "selected_option": self.options[selected_index],
            "response_time": response_time,
            "attempts": 1
        }

        # Provide feedback
        feedback = "✓ Correct!" if is_correct else "✗ Incorrect"
        # messagebox.showinfo("Feedback", feedback)

        # Move to next item
        self.current_assessment_index += 1
        self.show_assessment_item()

    def show_assessment_complete(self):
        """Show results of initial assessment and let user select learning mode"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        # Calculate basic stats
        correct_count = sum(1 for item in self.initial_assessment_results.values() if item["correct"])
        percent_correct = (correct_count / len(self.initial_assessment_results)) * 100

        # Main frame
        complete_frame = ttk.Frame(self.root, style="TFrame")
        complete_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        tk.Label(complete_frame,
                 text="Assessment Complete!",
                 font=("Arial", 24, "bold"),
                 foreground=self.colors["primary"],
                 background=self.colors["background"]).pack(pady=20)

        result_text = f"You correctly identified {correct_count} out of {len(self.initial_assessment_results)} signs ({percent_correct:.1f}%)"
        tk.Label(complete_frame,
                 text=result_text,
                 font=("Arial", 16),
                 background=self.colors["background"]).pack(pady=10)

        # Determine which items need more practice
        self.weak_items = [item for item, result in self.initial_assessment_results.items()
                           if not result["correct"]]

        if self.weak_items:
            weak_text = f"These signs need practice: {', '.join(self.weak_items)}"
        else:
            weak_text = "You knew all the signs! You'll review all items."
            self.weak_items = self.all_items

        tk.Label(complete_frame,
                 text=weak_text,
                 font=("Arial", 14),
                 wraplength=600,
                 background=self.colors["background"]).pack(pady=20)

        # Learning mode selection
        tk.Label(complete_frame,
                 text="Select Learning Mode:",
                 font=("Arial", 16, "bold"),
                 background=self.colors["background"]).pack(pady=20)

        mode_frame = tk.Frame(complete_frame, bg=self.colors["background"])
        mode_frame.pack(pady=20)

        predefined_btn = tk.Button(mode_frame,
                                   text="Structured Learning",
                                   command=lambda: self.set_mode("predefined"),
                                   font=("Arial", 14),
                                   bg="#5d9cec",
                                   fg="white",
                                   relief=tk.FLAT,
                                   padx=20,
                                   pady=10)
        predefined_btn.pack(side=tk.LEFT, padx=20)

        llm_btn = tk.Button(mode_frame,
                            text="AI-Assisted Learning",
                            command=lambda: self.set_mode("llm"),
                            font=("Arial", 14),
                            bg="#48cfad",
                            fg="white",
                            relief=tk.FLAT,
                            padx=20,
                            pady=10)
        llm_btn.pack(side=tk.LEFT, padx=20)

    def set_mode(self, mode):
        """Set learning mode and initialize the learning session"""
        self.learning_mode = mode
        self.initialize_learning_order()
        if mode == "predefined":
            self.setup_structured_learning_screen()
        else:
            self.setup_llm_learning_screen()

    def initialize_learning_order(self):
        """Create personalized learning order based on assessment results"""
        # Initialize performance tracking for all items (used in both modes)
        self.user_performance = {}
        for item in self.all_items:
            self.user_performance[item] = {
                "correct": 0,
                "incorrect": 0,
                "response_times": [],
                "learning time": 0,
                "last_shown": None
            }

        if self.learning_mode == "predefined":
            # Structured path: focus on weak items first, with spaced repetition
            strong_items = [item for item in self.all_items if item not in self.weak_items]

            self.learning_order = (
                    self.weak_items[:] +  # First pass through weak items
                    random.sample(strong_items, min(3, len(strong_items))) +  # Add some strong items
                    random.sample(self.weak_items, len(self.weak_items))  # Second pass through weak items
            )
        else:
            # LLM mode starts with weak items then the rest
            self.learning_order = self.weak_items + [
                item for item in self.all_items if item not in self.weak_items
            ]

    def setup_structured_learning_screen(self):
        """Structured learning interface with systematic progression"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        # Header frame
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        # Progress indicator
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(header_frame,
                                       variable=self.progress_var,
                                       maximum=len(self.learning_order),
                                       style="TProgressbar")
        progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        # Progress label
        self.progress_label = ttk.Label(header_frame,
                                        text="0/0",
                                        font=("Arial", 12),
                                        background=self.colors["background"])
        self.progress_label.pack(side=tk.RIGHT)

        # Main content frame
        content_frame = ttk.Frame(self.root, style="TFrame")
        content_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)

        # ASL image display
        self.image_frame = ttk.Frame(content_frame, style="TFrame")
        self.image_frame.pack(pady=20)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()

        # Sign information
        info_frame = ttk.Frame(content_frame, style="TFrame")
        info_frame.pack(fill=tk.X, pady=10)

        self.sign_label = ttk.Label(info_frame,
                                    text="",
                                    font=("Arial", 28, "bold"),
                                    foreground=self.colors["primary"],
                                    background=self.colors["background"])
        self.sign_label.pack()

        self.tip_label = ttk.Label(info_frame,
                                   text="",
                                   font=("Arial", 14),
                                   wraplength=600,
                                   background=self.colors["background"])
        self.tip_label.pack(pady=10)

        # Mnemonic section
        mnemonic_frame = ttk.Frame(content_frame, style="TFrame")
        mnemonic_frame.pack(fill=tk.X, pady=10)

        ttk.Label(mnemonic_frame,
                  text="Memory Tip:",
                  font=("Arial", 12, "bold"),
                  background=self.colors["background"]).pack(anchor=tk.W)

        self.mnemonic_label = ttk.Label(mnemonic_frame,
                                        text="",
                                        font=("Arial", 12, "italic"),
                                        wraplength=600,
                                        background=self.colors["background"])
        self.mnemonic_label.pack(anchor=tk.W)

        # Practice buttons
        practice_frame = ttk.Frame(content_frame, style="TFrame")
        practice_frame.pack(pady=20)

        ttk.Button(practice_frame,
                   text="Show Video",
                   command=self.show_video_demo,
                   style="TButton").pack(side=tk.LEFT, padx=5)

        # ttk.Button(practice_frame,
        #            text="Practice Now",
        #            command=self.practice_sign,
        #            style="TButton").pack(side=tk.LEFT, padx=5)

        # Navigation controls
        nav_frame = ttk.Frame(self.root, style="TFrame")
        nav_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(nav_frame,
                   text="Previous",
                   command=self.prev_item,
                   style="TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(nav_frame,
                   text="Next",
                   command=self.next_item,
                   style="TButton").pack(side=tk.RIGHT, padx=5)

        # Start the learning session
        self.learning_index = 0
        self.start_time = datetime.now()
        self.show_item()

    def show_item(self):
        """Display item in structured learning mode"""
        if self.learning_index >= len(self.learning_order):
            self.show_completion_screen()
            return

        self.current_item = self.learning_order[self.learning_index]

        # Update progress
        self.progress_var.set(self.learning_index + 1)
        self.progress_label.config(text=f"{self.learning_index + 1}/{len(self.learning_order)}")

        # Update content
        self.sign_label.config(text=self.current_item)
        self.show_placeholder_image()

        # Show learning tips
        tips = {
            "0": "Form a fist with all fingers curled into the palm.",
            "1": "Extend only your index finger upward.",
            "2": "Extend both your index and middle fingers.",
            "3": "Extend your thumb, index, and middle fingers.",
            "4": "Extend all four fingers with thumb tucked in.",
            "5": "Extend all five fingers apart.",
            "6": "Touch your thumb to your pinky with other fingers extended.",
            "7": "Touch your thumb to your ring finger with others extended.",
            "8": "Touch your thumb to your middle finger with others extended.",
            "9": "Touch your thumb to your index finger with others extended.",
            "airplane": "Extend your thumb, pinky, and index finger while tilting hand forward.",
            "father": "Place your thumb on your forehead with fingers spread.",
            "hello": "Wave your hand side to side with fingers together.",
            "I'm": "Point to yourself with your index finger.",
            "love": "Cross both hands over your heart.",
            "mother": "Place your thumb on your chin with fingers spread.",
            "ok": "Make a circle with thumb and index finger, others extended.",
            "sorry": "Make a fist and rub it in a circular motion on your chest.",
            "water": "Tap your chin with your index finger then move it down.",
            "yes": "Make a fist and nod it up and down like a head nodding."
        }
        self.tip_label.config(text=tips.get(self.current_item, "Learn this sign"))


        self.mnemonic_label.config(text=self.mnemonics.get(self.current_item, ""))

        # Record when this item was shown
        self.user_performance[self.current_item]["last_shown"] = datetime.now()
        self.item_start_time = time.time()

    def setup_llm_learning_screen(self):
        """Interactive LLM-assisted learning interface with integrated webcam"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        # Header frame
        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        # Progress indicator
        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(header_frame,
                                      variable=self.progress_var,
                                      maximum=len(self.learning_order),
                                      style="TProgressbar")
        progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)

        # AI indicator
        ttk.Label(header_frame,
                  text="AI Assistant",
                  font=("Arial", 12, "bold"),
                  foreground=self.colors["secondary"],
                  background=self.colors["background"]).pack(side=tk.RIGHT)

        # Main content frame - split into left (chat) and right (webcam)
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Left side - Chat display (now smaller)
        left_frame = ttk.Frame(main_frame, style="TFrame", width=450)  # Fixed width
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_frame.pack_propagate(False)  # Prevent frame from resizing to contents

        # Chat display area
        chat_frame = ttk.Frame(left_frame, style="TFrame")
        chat_frame.pack(expand=True, fill=tk.BOTH)

        self.chat_display = tk.Text(chat_frame,
                                   font=("Arial", 12),
                                   wrap=tk.WORD,
                                   padx=10,
                                   pady=10,
                                   bg="white",
                                   relief=tk.SUNKEN)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.pack(expand=True, fill=tk.BOTH)

        ttk.Button(left_frame,
                   text="Previous",
                   command=self.prev_item,
                   style="TButton").pack(side=tk.LEFT, padx=5)

        ttk.Button(left_frame,
                   text="Next",
                   command=self.next_item,
                   style="TButton").pack(side=tk.RIGHT, padx=5)
        # Right side - Webcam display (now larger)
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Webcam label - make larger
        self.webcam_label = ttk.Label(right_frame)
        self.webcam_label.pack(expand=True, fill=tk.BOTH)

        ttk.Button(right_frame,
                   text="Show Video",
                   command=self.show_video_demo,
                   style="TButton").pack(side=tk.LEFT, padx=5)


        # Initialize webcam variables
        self.cap = None
        self.is_practicing = False
        self.predicted_label = ""
        self.latency = 0
        self.label_index = 0


        # Start the learning session
        self.learning_index = 0
        self.start_time = datetime.now()
        self.show_llm_item()

        self.start_webcam()

    def start_webcam(self):
        """Start or stop the webcam feed"""
        if not self.is_practicing:
            # Start webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return

            self.is_practicing = True
            self.update_webcam()
        else:
            # Stop webcam
            self.is_practicing = False
            if self.cap:
                self.cap.release()
            self.webcam_label.config(image='')

    def put_text_with_wrap(self,frame, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 10  # Margin from frame edges

        # Get frame dimensions
        height, width = frame.shape[:2]
        x, y = position

        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Check if text exceeds frame width (with margin)
        max_text_width = width - x - margin
        if text_width > max_text_width:
            # Reduce font scale proportionally
            font_scale = font_scale * (max_text_width / text_width) * 0.9  # 10% safety margin
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Check if text exceeds frame height (with margin)
        if y + text_height + baseline > height - margin:
            y = height - text_height - baseline - margin  # Move text up

        # Draw the text
        cv2.putText(frame, text, (x, y + text_height), font, font_scale, color, thickness)
        return text_height + baseline + 5  # Return height used (for next line position)

    def update_webcam(self):
        """Update the webcam feed in the GUI"""
        if self.is_practicing and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for prediction
                type = "digit" if self.current_item.isdigit() else "word"
                self.predicted_label, self.latency, self.label_index = predict_gesture(frame, type)
                pair_result = search_pair(str([str(self.predicted_label), str(self.current_item)]), type)
                # print(found_pair)
                # if pair_result:
                #     pair_result = pair_result
                # else:
                #     # print("Pair not found")
                #     pair_result = "Pair not found"
                # Draw prediction on frame
                y_position = 30

                # Draw texts with automatic positioning
                y_position += self.put_text_with_wrap(frame,
                                                      f"Learning: {self.current_item}",
                                                      (10, y_position),
                                                      color=(255, 0, 0))

                y_position += self.put_text_with_wrap(frame,
                                                 f"Predicted: {self.predicted_label}",
                                                 (10, y_position),
                                                 color=(0, 255, 0))
                if self.predicted_label == self.current_item:
                    y_position += self.put_text_with_wrap(frame,
                                                         f"Correct!",
                                                         (10, y_position),
                                                         color=(0, 255, 0))
                if pair_result != "Pair not found" and self.predicted_label != self.current_item and pair_result != self.pre_pair:
                    self.pre_pair = pair_result
                    self.insert_chat_message('AI', '\n'+'thumb:'+pair_result['differences']['thumb']+'\n'
                                             +'index:'+pair_result['differences']['index']+'\n'
                                             +'middle:'+pair_result['differences']['middle']+'\n'
                                             +'ring:'+pair_result['differences']['ring']+'\n'
                                             +'pinky:'+pair_result['differences']['pinky']+'\n'
                                             ,True)
                    for i, tip in enumerate(pair_result['correction_tips'], 1):
                        self.insert_chat_message('Tip', f"{i}: {tip}",False,color='red')


                # Convert to PhotoImage
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((800, 600))  # Resize to fit
                imgtk = ImageTk.PhotoImage(image=img)

                # Update label
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)

            # Schedule next update
            self.root.after(10, self.update_webcam)

    def show_llm_item(self):
        """Initiate interaction for current item in LLM mode"""
        if self.learning_index >= len(self.learning_order):
            self.show_completion_screen()
            return

        self.current_item = self.learning_order[self.learning_index]

        # Update progress
        self.progress_var.set(self.learning_index + 1)

        # Generate initial prompt based on performance
        if self.user_performance[self.current_item]["correct"] > 0:
            prompt = f"Let's review the sign for '{self.current_item}'. Can you demonstrate it?"
        else:
            prompt = f"Let's learn the sign for '{self.current_item}'.\nMemory tip:'{self.mnemonics[self.current_item]}'\n" \

        self.insert_chat_message("AI", prompt, True)
        # self.user_input.focus()

        # Record when this item was shown
        self.user_performance[self.current_item]["last_shown"] = datetime.now()
        self.item_start_time = time.time()

    def show_placeholder_image(self):
        """Load and display the actual ASL image for the current item"""
        try:
            # Determine if current item is a digit or word
            if self.current_item.isdigit():
                folder = "digits"
                end = '.jpg'
            else:
                folder = "words"
                end = '.png'

            # Construct image path - assuming .png format
            img_path = f"{folder}/{self.current_item}{end}"
            # print(self.current_item)
            # Load and resize the image while maintaining aspect ratio

            img = Image.open(img_path)
            img.thumbnail((300, 400))  # Resize to fit display area

            # Convert to PhotoImage for Tkinter
            img_tk = ImageTk.PhotoImage(img)

            # Update the image label
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Keep reference

        except FileNotFoundError:
            # Fallback to placeholder if image not found
            img = Image.new('RGB', (300, 400), color='lightgray')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
                text = f"Image not found\nfor {self.current_item}"
                draw.text((50, 150), text, fill="black", font=font)
            except:
                pass

            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create error placeholder
            img = Image.new('RGB', (300, 400), color='pink')
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.load_default()
                text = f"Error loading\n{self.current_item}"
                draw.text((50, 150), text, fill="black", font=font)
            except:
                pass

            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

    def show_video_demo(self):
        """Show video demonstration for the current item"""
        try:
            folder = "digits_video" if self.current_item.isdigit() else "words_video"
            video_path = f"{folder}/{self.current_item}.mp4"

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"No video found for {self.current_item}")

            # Open with system's default video player
            subprocess.Popen(['start', video_path], shell=True)  # Windows
            # Use 'open' on macOS, 'xdg-open' on Linux if needed

        except Exception as e:
            messagebox.showerror("Video Error", str(e))

    def practice_sign(self):
        """Open webcam for practicing sign recognition"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not access the webcam.")
            return

        # messagebox.showinfo("Practice", "Webcam started. Press 'q' to exit practice mode.")
        type = "digit" if self.current_item.isdigit() else "word"
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用 MediaPipe 检测手部关键点
            results = hands.process(rgb_frame)
            predicted_label, latency, index = predict_gesture(frame,type)
            cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(f"Practice: {self.current_item}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def generate_and_insert_tips(self, sign):
        """Generate tips and insert them into the chatbox in real-time"""
        # Simulate AI-generated tips
        tips = self.generate_tips_for_sign(sign)
        # Insert the generated tips into the chatbox immediately
        self.insert_chat_message("AI", tips, False, color='red')

    def generate_tips_for_sign(self, sign):
        """Simulate generating tips using AI for the given ASL sign"""
        tips = {
            "0": "Form a fist with all fingers curled into the palm.",
            "1": "Extend only your index finger upward.",
            "2": "Extend both your index and middle fingers.",
            "3": "Extend your thumb, index, and middle fingers.",
            "4": "Extend all four fingers with thumb tucked in.",
            "5": "Extend all five fingers apart.",
            "6": "Touch your thumb to your pinky with other fingers extended.",
            "7": "Touch your thumb to your ring finger with others extended.",
            "8": "Touch your thumb to your middle finger with others extended.",
            "9": "Touch your thumb to your index finger with others extended.",
            "airplane": "Extend your thumb, pinky, and index finger while tilting hand forward.",
            "father": "Place your thumb on your forehead with fingers spread.",
            "hello": "Wave your hand side to side with fingers together.",
            "I'm": "Point to yourself with your index finger.",
            "love": "Cross both hands over your heart.",
            "mother": "Place your thumb on your chin with fingers spread.",
            "ok": "Make a circle with thumb and index finger, others extended.",
            "sorry": "Make a fist and rub it in a circular motion on your chest.",
            "water": "Tap your chin with your index finger then move it down.",
            "yes": "Make a fist and nod it up and down like a head nodding."
        }

        # Return the corresponding tip based on the sign
        return tips.get(sign, "Tip for this sign is not available.")

    def process_captured_frame(self, frame):
        """Process and store the frame once the 2-second hold is completed"""
        # Here you can process the captured frame for further steps
        predicted_label, latency, index = predict_gesture(frame,
                                                          type="digit" if self.current_item.isdigit() else "word")
        # Show the final result in the chat
        self.insert_chat_message("AI", f"Gesture recognized: {predicted_label}",False)

        # You can add more logic here for further actions, e.g., comparing with the correct gesture, updating stats, etc.
        # print(f"Captured frame with label: {predicted_label}")

    def insert_chat_message(self, sender, message, clear_before_insert=False, color="black"):
        """Insert a message into the chat display with specified color"""
        self.chat_display.config(state=tk.NORMAL)

        # Configure a tag with the specified color
        self.chat_display.tag_config(color, foreground=color)

        if clear_before_insert:
            self.chat_display.delete(1.0, tk.END)

        # Insert text with the color tag
        self.chat_display.insert(tk.END, f"{sender}: {message}\n", color)

        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def next_item(self):
        """Move to the next learning item"""
        learning_time = time.time() - self.item_start_time
        self.user_performance[self.current_item]['learning time'] = learning_time
        self.learning_index += 1
        if self.learning_mode == "predefined":
            self.show_item()
        else:
            self.show_llm_item()

    def prev_item(self):
        """Return to the previous learning item"""
        if self.learning_index > 0:
            self.learning_index -= 1
            if self.learning_mode == "predefined":
                self.show_item()
            else:
                self.show_llm_item()

    def show_completion_screen(self):
        """Show completion screen with results"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        time_spent = datetime.now() - self.start_time
        mins, secs = divmod(time_spent.total_seconds(), 60)

        # Calculate improvement from assessment
        initial_correct = sum(1 for item in self.initial_assessment_results.values() if item["correct"])
        final_correct = sum(1 for item in self.user_performance.values() if item["correct"] > 0)
        improvement = final_correct - initial_correct

        # Main frame
        complete_frame = ttk.Frame(self.root, style="TFrame")
        complete_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        tk.Label(complete_frame,
                 text="Learning Session Complete!",
                 font=("Arial", 24, "bold"),
                 foreground=self.colors["primary"],
                 background=self.colors["background"]).pack(pady=20)

        # Statistics
        stats = [
            f"Time spent: {int(mins)} minutes {int(secs)} seconds",
            f"Initial score: {initial_correct}/{len(self.all_items)}",
        ]

        for stat in stats:
            tk.Label(complete_frame,
                     text=stat,
                     font=("Arial", 14),
                     background=self.colors["background"]).pack(pady=5)

        tk.Button(complete_frame,
                  text="Start Final Assessment",
                  command=self.start_final_assessment,
                  font=("Arial", 14),
                  bg=self.colors["accent"],
                  fg="white",
                  relief=tk.FLAT,
                  padx=20,
                  pady=10).pack(pady=30)

    def start_final_assessment(self):
        """Start the post-learning assessment"""
        self.clear_screen()
        self.start_time = datetime.now()

        # Reuse the same logic as initial assessment
        self.assessment_items = random.sample(self.all_items, len(self.all_items))
        self.current_assessment_index = 0
        self.final_assessment_results = {}

        for item in self.all_items:
            self.final_assessment_results[item] = {
                "correct": False,
                "response_time": None,
                "attempts": 0
            }

        self.show_final_assessment_item()

    def show_final_assessment_item(self):
        """Display a post-learning assessment item"""
        if self.current_assessment_index >= len(self.assessment_items):
            self.show_final_results()
            return

        self.clear_screen()
        self.current_item = self.assessment_items[self.current_assessment_index]

        header_frame = ttk.Frame(self.root, style="TFrame")
        header_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame,
                 text="Final Assessment",
                 font=("Arial", 20, "bold"),
                 foreground=self.colors["primary"],
                 background=self.colors["background"]).pack(side=tk.LEFT)
        # Question
        tk.Label(self.root,
                 text=f"Which image shows the ASL sign for: {self.current_item}?",
                 font=("Arial", 18),
                 bg=self.colors["background"]).pack(pady=30)

        progress = f"Item {self.current_assessment_index + 1} of {len(self.assessment_items)}"
        tk.Label(header_frame,
                 text=progress,
                 font=("Arial", 12),
                 background=self.colors["background"]).pack(side=tk.RIGHT)

        options_frame = tk.Frame(self.root, bg=self.colors["background"])
        options_frame.pack(pady=20)

        self.option_images = []
        self.option_buttons = []

        DISPLAY_SIZE = (250, 250)

        # Use category-matched distractors
        folder = "digit" if self.current_item.isdigit() else "word"
        extension = ".jpg"
        option = self.current_item
        self.options = [str(option),
                   f"assessment/{folder}/{option}/{option}-2{extension}",
                   f"assessment/{folder}/{option}/{option}-3{extension}",
                   f"assessment/{folder}/{option}/{option}-4{extension}"]
        random.shuffle(self.options)
        self.correct_option = self.options.index(self.current_item)
        print(f"Correct option index: {self.current_item}")  # Debugging line

        for i, option in enumerate(self.options):
            frame = tk.Frame(options_frame, bg=self.colors["background"])
            frame.pack(side=tk.LEFT, padx=15)

            try:
                # Load image with correct extension
                if option == str(self.current_item):
                    option = f"assessment/{folder}/{option}/{option}-1{extension}"

                img_path = option

                img = Image.open(img_path)
                img.thumbnail(DISPLAY_SIZE, Image.LANCZOS)

                canvas = Image.new('RGB', DISPLAY_SIZE, (255, 255, 255))
                x_offset = (DISPLAY_SIZE[0] - img.width) // 2
                y_offset = (DISPLAY_SIZE[1] - img.height) // 2
                canvas.paste(img, (x_offset, y_offset))

                img_tk = ImageTk.PhotoImage(canvas)

            except Exception as e:
                print(f"Error loading image {option}: {e}")
                canvas = Image.new('RGB', DISPLAY_SIZE, (240, 240, 240))
                draw = ImageDraw.Draw(canvas)
                font = ImageFont.load_default()
                draw.text((50, 120), option, fill="black", font=font)
                img_tk = ImageTk.PhotoImage(canvas)

            self.option_images.append(img_tk)

            btn = tk.Button(frame,
                            image=img_tk,
                            bg="white",
                            relief=tk.FLAT,
                            command=lambda idx=i: self.process_final_assessment_response(idx))
            btn.pack()
            self.option_buttons.append(btn)

        self.item_start_time = time.time()

    def process_final_assessment_response(self, selected_index):
        """Handle user's response to final assessment item"""
        response_time = time.time() - self.item_start_time
        is_correct = (selected_index == self.correct_option)
        self.final_assessment_results[self.current_item] = {
            "correct": is_correct,
            "response_time": response_time,
            "selected_option": self.options[selected_index],
            "attempts": 1
        }

        self.current_assessment_index += 1
        self.show_final_assessment_item()

    def show_final_results(self):
        """Compare pre/post assessment and offer save option"""
        self.clear_screen()
        self.root.configure(bg=self.colors["background"])

        final_correct = sum(1 for r in self.final_assessment_results.values() if r["correct"])
        initial_correct = sum(1 for r in self.initial_assessment_results.values() if r["correct"])
        improvement = final_correct - initial_correct

        frame = ttk.Frame(self.root, style="TFrame")
        frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)

        tk.Label(frame, text="Final Assessment Complete!",
                 font=("Arial", 24, "bold"),
                 fg=self.colors["primary"],
                 bg=self.colors["background"]).pack(pady=20)

        summary = [
            f"Initial Score: {initial_correct}/{len(self.all_items)}",
            f"Final Score: {final_correct}/{len(self.all_items)}",
            f"Improvement: {improvement} signs"
        ]

        for line in summary:
            tk.Label(frame,
                     text=line,
                     font=("Arial", 14),
                     bg=self.colors["background"]).pack(pady=5)

        self.save_results(initial_correct,final_correct,improvement)

    def save_results(self,initial_correct,final_correct,improvement):
        # """Save all experiment data to a file"""
        # filename = f"asl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data = {
            "participant_id": self.username,  # Optional: prompt user in production
            "learning_mode": self.learning_mode,
            "initial_assessment": self.initial_assessment_results,
            "learning_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "performance": {
                    item: {
                        # "correct": p["correct"],
                        # "incorrect": p["incorrect"],
                        # "response_times": p["response_times"],
                        "learning time": p["learning time"],
                        "last_shown": p["last_shown"].isoformat() if p["last_shown"] else None
                    }
                    for item, p in self.user_performance.items()
                },
                # "learning_order": self.learning_order
            },
            "final_assessment": self.final_assessment_results
        }
        database.save_learning_record(self.username, self.learning_mode, self.initial_assessment_results,
                                      data["learning_session"], self.learning_order, self.final_assessment_results,initial_correct,final_correct,improvement)
        #
        # with open(filename, "w") as f:
        #     json.dump(data, f, indent=2)

        messagebox.showinfo("Results Saved", f"Your results have been saved")

        self.clear_screen()
        game_window = tk.Toplevel(self.root)
        WhackAMoleGameApp(game_window, "Whack-A-Mole Game", 1200, 500, 'test')  # 假设用户名为 "test_user"
        # self.root.destroy()

    def clear_screen(self):
        """Clear all widgets from the root window"""
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLLearningApp(root,username='test')
    root.mainloop()