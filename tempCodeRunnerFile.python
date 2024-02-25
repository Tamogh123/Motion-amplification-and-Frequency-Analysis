import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
class VideoPlayer(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent
        self.video_canvas = tk.Canvas(self, bg="black", width=241, height=250)
        self.video_canvas.pack(padx=20, pady=20)

        self.video_path = None
        self.cap = None
        self.is_playing = False

        self.play_button = tk.Button(self, text="Play", command=self.toggle_play)
        self.play_button.pack(pady=10)

        self.choose_video_button = tk.Button(self, text="Choose Video", command=self.choose_video)
        self.choose_video_button.pack(pady=10)

    def choose_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.video_path = file_path
            self.load_video()

    def load_video(self):
        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)
        self.is_playing = False
        self.play_button.config(text="Play")

        self.play_video()

    def play_video(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.video_canvas.image = frame

                if self.is_playing:
                    self.after(30, self.play_video)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self.play_video()
        else:
            self.play_button.config(text="Play")

def amplify_video(input_path):
        from amplify_code import phase_amplify
        from amplify_code import load_video
        from amplify_code import difference_of_iir
        import mediapy as media;
        from imageio import get_reader, get_writer
        import numpy as np;
        import skimage.transform as sktransform
        video_file = input_path
        video = load_video(video_file)
        video_resized = np.zeros((len(video), 250, 250, 3))
        for i in range(len(video)):
            video_resized[i] = sktransform.resize(video[i], (250,250))
        magnification_factor = 5
        fl = .04
        fh = .4
        fs = 1
        attenuate_other_frequencies=True
        pyr_type = "octave"
        sigma = 5
        temporal_filter = difference_of_iir
        scale_video = .8
        result=phase_amplify(video_resized, magnification_factor, fl, fh, fs, attenuate_other_frequencies=attenuate_other_frequencies,pyr_type = 'octave' ,sigma=sigma, temporal_filter=temporal_filter);
        path = './result.mp4'
        media.write_video(path, result)

def generate_graphs():
    # Your graph generation code here
    pass

def import_video():
    #  file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    # return file_path
    pass

def process_video():
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    amplify_video(file_path)

def create_home_page(notebook, bg_color):
    home_page = tk.Frame(notebook, bg=bg_color)
    background_image = Image.open("wallpaper3.png")
    resized_image = background_image.resize((2048, 1024))
    background_photo = ImageTk.PhotoImage(resized_image)
    # Create a label with the background image
    background_label = tk.Label(home_page, image=background_photo)
    background_label.image = background_photo  # Store the reference
    background_label.pack(expand=True, fill=tk.BOTH)
    label_home = tk.Label(home_page, text="Home Page", font=("Helvetica", 25), bg="black", fg="#ecf0f1")
    label_home.place(relx=0.075, rely=0.05, anchor=tk.N)  # Center the text
    import_button = tk.Button(home_page, text="Import Video", command=import_video, width=20, height=2, font=("Helvetica", 14), bg="black", fg="white")
    import_button.place(relx=0.5, rely=0.4, anchor=tk.N)
    submit_button = tk.Button(home_page, text="Submit", command=process_video, width=20, height=2, font=("Helvetica", 14), bg="black", fg="white")
    submit_button.place(relx=0.5, rely=0.5, anchor=tk.N)

    return home_page

def create_amplified_page(notebook, bg_color):
    amplified_page = tk.Frame(notebook, bg=bg_color)
    background_image = Image.open("wallpaper3.png")
    resized_image = background_image.resize((2048, 1024))
    background_photo = ImageTk.PhotoImage(resized_image)
    # Create a label with the background image
    background_label = tk.Label(amplified_page, image=background_photo)
    background_label.image = background_photo  # Store the reference
    background_label.pack(expand=True, fill=tk.BOTH)
    label_amplified = tk.Label(amplified_page, text="Amplified Video Page", font=("Kozuka Gothic Pr6N L", 25), bg="black", fg="#ecf0f1")
    label_amplified.place(relx=0.1, rely=0.05, anchor=tk.N)  # Center the text
    label_amplified.lift() 
    return amplified_page

def create_results_page(notebook, bg_color):
    results_page = tk.Frame(notebook, bg=bg_color)

    label_results = tk.Label(results_page, text="Final Results Page", font=("Helvetica", 18), bg="#34495e", fg="#ecf0f1")
    label_results.pack(pady=20)

    generate_graphs_button = tk.Button(results_page, text="Generate Graphs", command=generate_graphs, width=20, height=2, font=("Helvetica", 14), bg="#3498db", fg="white")
    generate_graphs_button.pack(pady=20)
    video_player = VideoPlayer(results_page)
    video_player.pack(pady=20)
    return results_page

def create_about_page(notebook, bg_color):
    about_page = tk.Frame(notebook, bg=bg_color)

    label_about = tk.Label(about_page, text="About Page", font=("Helvetica", 18), bg="#34495e", fg="#ecf0f1")
    label_about.pack(pady=20)

    return about_page

def main():
    root = tk.Tk()
    root.title("Scientific App")
    root.geometry("2048x1024")  # Set initial size

    # Use the 'clam' theme for ttk widgets
    style = ttk.Style()
    style.theme_use("clam")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # Use a cool color palette
    bg_color = "#34495e"  # Dark background color

    home_page = create_home_page(notebook, bg_color)
    amplified_page = create_amplified_page(notebook, bg_color)
    results_page = create_results_page(notebook, bg_color)
    about_page = create_about_page(notebook, bg_color)

    notebook.add(home_page, text="Home")
    notebook.add(amplified_page, text="Amplified Video",)
    notebook.add(results_page, text="Final Results")
    notebook.add(about_page, text="About")

    root.mainloop()

if __name__ == "__main__":
    main()
