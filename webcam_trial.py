import cv2
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
import os
import amplify_code
class WebcamApp:
    def __init__(self, root, video_source=0, output_dir="recordings"):
        self.root = root
        self.root.title("Webcam Recorder")

        self.video_source = video_source
        self.output_dir = output_dir
        self.output_path = ""

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(self.video_source)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.is_recording = False

        # Record Button
        self.record_button = ttk.Button(root, text="Record", command=self.toggle_recording)
        self.record_button.pack(pady=10)

        # Save Button
        self.save_button = ttk.Button(root, text="Save", command=self.save_video)
        self.save_button.pack(pady=10)

    def toggle_recording(self):
        if not self.is_recording:
            self.record_button.config(text="Stop Recording")
            self.is_recording = True
            self.output_path = os.path.join(self.output_dir, f"output_video_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            self.record_thread = Thread(target=self.record_video)
            self.record_thread.start()
        else:
            self.record_button.config(text="Record")
            self.is_recording = False

    def record_video(self):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (int(self.cap.get(3)), int(self.cap.get(4))))

        while self.is_recording:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Display the frame in the GUI (optional)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Adjust the size as needed

            # Write the frame to the video file
            out.write(frame)

        # Release the video writer
        out.release()

    def save_video(self):
        if self.output_path:
            self.save_button.config(state=tk.DISABLED)  # Disable the button while saving
            self.is_recording = False  # Stop recording if it's still in progress
            self.record_button.config(text="Record")
            self.cap.release() 
            if hasattr(self, 'record_thread') and self.record_thread.is_alive():
                self.record_thread.join()

            # Enable the save button again
            self.save_button.config(state=tk.NORMAL)
        

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    app.run()
