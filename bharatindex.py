import tkinter as tk
from tkinter import font
import webbrowser
import cv2
from PIL import Image, ImageTk
import threading
import queue
import os
 
# ── Change this to your video file path ───────────────────────────────────────
VIDEO_PATH = r"C:\Users\BHARAT\OneDrive\Desktop\map\full.mp4"   # Put full.mp4 in the same folder as this script
#   OR use full absolute path, e.g.:
#   VIDEO_PATH = r"C:\Users\YourName\Videos\full.mp4"
# ──────────────────────────────────────────────────────────────────────────────
 
VIDEO_WIDTH  = 480
VIDEO_HEIGHT = 500
 
frame_queue = queue.Queue(maxsize=2)   # small buffer between thread and UI
running     = True                     # global flag to stop thread on exit
 
 
def video_reader_thread(path):
    """Runs in background: reads frames and puts them in the queue."""
    global running
    cap = cv2.VideoCapture(path)
 
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}")
        print(f"        File exists: {os.path.exists(path)}")
        return
 
    fps   = cap.get(cv2.CAP_PROP_FPS)
    fps   = fps if fps > 0 else 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    while running:
        ret, frame = cap.read()
 
        if not ret or (total > 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) >= total):
            # Loop: rewind to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
 
        # Resize and convert once in the thread (keeps UI thread light)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT),
                           interpolation=cv2.INTER_LINEAR)
 
        try:
            frame_queue.put(frame, timeout=0.1)
        except queue.Full:
            pass   # drop frame if UI is behind; never block
 
    cap.release()
 
 
def create_app():
    global running
 
    root = tk.Tk()
    root.title("UPES - NavCAMPUS")
    root.geometry("900x500")
    root.resizable(False, False)
    root.configure(bg="#F5EFE6")
 
    # ── Fonts ──────────────────────────────────────────────────────────────────
    heading_font = font.Font(family="Georgia",   size=30, weight="bold")
    sub_font     = font.Font(family="Georgia",   size=13, slant="italic")
    brand_font   = font.Font(family="Helvetica", size=22, weight="bold")
    btn_font     = font.Font(family="Helvetica", size=13, weight="bold")
 
    # ── Left Panel ─────────────────────────────────────────────────────────────
    left = tk.Frame(root, bg="#F5EFE6", width=420, height=500)
    left.pack(side=tk.LEFT, fill=tk.BOTH)
    left.pack_propagate(False)
 
    tk.Label(left, text="WELCOME\nTO UPES", font=heading_font,
             fg="#7B0D1E", bg="#F5EFE6", justify=tk.LEFT, anchor="w"
             ).place(x=50, y=50)
 
    tk.Label(left, text="University of Tomorrow", font=sub_font,
             fg="#7B0D1E", bg="#F5EFE6", anchor="w"
             ).place(x=50, y=155)
 
    tk.Frame(left, bg="#7B0D1E", height=2, width=260).place(x=50, y=178)
 
    # Graduation cap
    cap_canvas = tk.Canvas(left, width=70, height=60,
                           bg="#F5EFE6", highlightthickness=0)
    cap_canvas.place(x=50, y=210)
    cap_canvas.create_polygon(35, 10, 5, 25, 35, 40, 65, 25, fill="#1a1a1a")
    cap_canvas.create_rectangle(28, 38, 42, 52, fill="#1a1a1a", outline="")
    cap_canvas.create_line(65, 25, 65, 45, fill="#1a1a1a", width=2)
    cap_canvas.create_oval(61, 43, 69, 51, fill="#1a1a1a")
 
    tk.Label(left, text="NavCAMPUS", font=brand_font,
             fg="#1a1a1a", bg="#F5EFE6").place(x=130, y=220)
 
    # Click It! button
    btn_frame = tk.Frame(left, bg="#7B0D1E", cursor="hand2")
    btn_frame.place(x=50, y=370, width=200, height=55)
    btn_label = tk.Label(btn_frame, text="Click It!", font=btn_font,
                         fg="white", bg="#7B0D1E", cursor="hand2")
    btn_label.pack(expand=True, fill=tk.BOTH)
 
    def on_click(e=None): webbrowser.open("https://www.upes.ac.in")
    def on_enter(e): btn_frame.config(bg="#5a0a14"); btn_label.config(bg="#5a0a14")
    def on_leave(e): btn_frame.config(bg="#7B0D1E"); btn_label.config(bg="#7B0D1E")
 
    for w in (btn_frame, btn_label):
        w.bind("<Button-1>", on_click)
        w.bind("<Enter>",    on_enter)
        w.bind("<Leave>",    on_leave)
 
    # ── Right Panel – Video Display ────────────────────────────────────────────
    right = tk.Frame(root, bg="#000000", width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    right.pack_propagate(False)
 
    video_label = tk.Label(right, bg="#000000")
    video_label.pack(fill=tk.BOTH, expand=True)
 
    # ── UI updater: pulls frames from queue and shows them ─────────────────────
    def update_ui():
        try:
            frame = frame_queue.get_nowait()
            img   = ImageTk.PhotoImage(Image.fromarray(frame))
            video_label.config(image=img)
            video_label.image = img          # prevent garbage collection
        except queue.Empty:
            pass                             # no frame yet, just wait
        root.after(16, update_ui)            # ~60 fps poll
 
    update_ui()
 
    # ── Start background reader thread ─────────────────────────────────────────
    t = threading.Thread(target=video_reader_thread, args=(VIDEO_PATH,),
                         daemon=True)
    t.start()
 
    # ── Clean shutdown ─────────────────────────────────────────────────────────
    def on_close():
        global running
        running = False
        root.destroy()
 
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
 
 
if __name__ == "__main__":
    create_app()