import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapy as media
global x2
global x1,y1,w1,h1
global stored_value
global freq_x_real
global freq_y_real
global freq_x_amp
global freq_y_amp
stored_value=1
selected_file_path=None
def display_integer():
    # global accuracy_value
    label_var.set(str(integer_value))
def open_new_window():
    new_window = tk.Toplevel(root)
    new_window.title("MultiROIAnalyzer")
    new_window.geometry("600x600")
    new_window.config(bg="black")
    label = tk.Label(new_window, text="Here you can analalyze multiple regions of intrests one by one !",font=("Helvetica", 12))
    label.pack(padx=20, pady=20)
    for index in range(stored_value):
        new_window_button=tk.Button(
            new_window,
            text=f"ROI {index+1}",
            command=open_new_window,
            width=20,
            height=2,
            bg="black",
            fg="white",
        )
        new_window_button.place(relx=0.5,rely=0.2+index*0.1,anchor=tk.N)
def roi_video(input_path):
 
    from ROI1 import phase_amplify1
    from ROI1 import load_video1
    from ROI1 import process_video1
    video_path = input_path
    #output_video_path = 'output_video.mp4'
    process_video1(video_path)
    video_file = "output_video.mp4"
    video = load_video1(video_file)
    video_resized = np.zeros((len(video), 250, 250, 3))
    for i in range(len(video)):
        video_resized[i] = sktransform.resize(video[i], (250,250))
    magnification_factor = 7
    fl = .04
    fh = .4
    fs = 1
    attenuate_other_frequencies=True
    pyr_type = "octave"
    sigma = 5
    temporal_filter = difference_of_iir
    scale_video = .8
    result1=phase_amplify(video_resized, magnification_factor, fl, fh, fs, attenuate_other_frequencies=attenuate_other_frequencies, pyramid_type=pyr_type, sigma=sigma, temporal_filter=temporal_filter)
    media.write_video('result1.mp4',result1)
def amplify_video(input_path):
    from amplify_code import phase_amplify
    from amplify_code import load_video
    from amplify_code import difference_of_iir
    import mediapy as media
    from imageio import get_reader, get_writer
    import numpy as np
    import skimage.transform as sktransform

    video_file = input_path
    video = load_video(video_file)
    # video_resized = np.zeros((len(video), 250, 250, 3))
    # for i in range(len(video)):
    #     video_resized[i] = sktransform.resize(video[i], (250, 250))
    magnification_factor = 5
    fl = 0.04
    fh = 0.4
    fs = 1
    attenuate_other_frequencies = True
    pyr_type = "octave"
    sigma = 5
    temporal_filter = difference_of_iir
    scale_video = 0.8
    result = phase_amplify(
        video,
        magnification_factor,
        fl,
        fh,
        fs,
        attenuate_other_frequencies=attenuate_other_frequencies,
        pyr_type="octave",
        sigma=sigma,
        temporal_filter=temporal_filter,
    )
    path = "./result.mp4"
    media.write_video(path, result,fps=30)
    
class VideoPlayerorg(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent
        self.video_canvas = tk.Canvas(self, bg="black")
        self.video_canvas.pack(padx=20, pady=20)

        self.video_path = None
        self.cap = None
        self.is_playing = False

        self.play_button = tk.Button(self, text="Play", command=self.toggle_play)
        self.play_button.pack(pady=10)
        
        self.choose_video_button = tk.Button(self, text="Choose Video", command=self.choose_video)
        self.choose_video_button.pack(pady=10)

    def choose_video(self):
        self.video_path = selected_file_path
        self.load_video()
    
    def load_video(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.is_playing = False
        self.play_button.config(text="Play")
        width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_canvas.config(width=width,height=height)
        self.play_video()

    def play_video(self,frame_index=None):
        if frame_index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.video_canvas.image = frame

                if self.is_playing:
                    current_frame_index=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if current_frame_index<total_frames-1:
                        self.after(30,self.play_video,current_frame_index+1)
                    else:
                        self.after(30,self.play_video,0)    
                    
            
                    
            else:
                print("Failes to read video")                

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self.play_video()
        else:
            self.play_button.config(text="Play")

    
class VideoPlayer(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent
        self.video_canvas = tk.Canvas(self, bg="black")
        self.video_canvas.pack(padx=20, pady=20)

        self.video_path = None
        self.cap = None
        self.is_playing = False

        self.play_button = tk.Button(self, text="Play", command=self.toggle_play)
        self.play_button.pack(pady=10)
        
        self.choose_video_button = tk.Button(self, text="Choose Video", command=self.choose_video)
        self.choose_video_button.pack(pady=10)

    def choose_video(self):
        self.video_path = 'result.mp4'
        self.load_video()
    
    def load_video(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.is_playing = False
        self.play_button.config(text="Play")
        width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_canvas.config(width=width,height=height)
        self.play_video()

    def play_video(self,frame_index=None):
        if frame_index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)

                self.video_canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.video_canvas.image = frame

                if self.is_playing:
                    current_frame_index=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if current_frame_index<total_frames-1:
                        self.after(30,self.play_video,current_frame_index+1)
                    else:
                        self.after(30,self.play_video,0)    
                    
            
                    
            else:
                print("Failes to read video")                

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self.play_video()
        else:
            self.play_button.config(text="Play")


def generate_graphs_orginial():
    global freq_x_real
    global freq_y_real
    from roi_graphs import roiFreq
    freq_x_real,freq_y_real=roiFreq(selected_file_path)
    
    

def generate_graphs_trimmed(): 
    global freq_x_amp
    global freq_y_amp    
    from roi_graphs import roiFreq
    freq_x_amp,freq_y_amp=roiFreq("result.mp4")
    
# def accuracy_cal():
#     global accuracy_value
#     from accuracy import accuracy
#     accuracy_value=accuracy(freq_x_real,freq_y_real,freq_x_amp,freq_y_amp)

def import_video():
    #  file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    # return file_path
    pass

def roi_process():
    global selected_file_path
    selected_file_path=filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if selected_file_path:
        roi_video(selected_file_path)
def process_video():
    global selected_file_path
    selected_file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if selected_file_path:
        amplify_video(selected_file_path)
    


def create_home_page(notebook, bg_color):
    home_page = tk.Frame(notebook, bg=bg_color)
    background_image = Image.open("wallpaper3.png")
    resized_image = background_image.resize((2048, 1024))
    background_photo = ImageTk.PhotoImage(resized_image)
    # Create a label with the background image
    background_label = tk.Label(home_page, image=background_photo)
    background_label.image = background_photo  # Store the reference
    background_label.pack(expand=True, fill=tk.BOTH)
    label_home = tk.Label(
        home_page, text="Home Page", font=("Helvetica", 25), bg="black", fg="#ecf0f1"
    )
    label_home.place(relx=0.075, rely=0.05, anchor=tk.N)  # Center the text

    roi_button = tk.Button(
        home_page,
        text="ROI",
        command=roi_process,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="black",
        fg="white",
    )
    roi_button.place(relx=0.5, rely=0.4, anchor=tk.N)
    submit_button = tk.Button(
        home_page,
        text="Amplify a video!",
        command=process_video,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="black",
        fg="white",
    )
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
    label_amplified = tk.Label(
        amplified_page,
        text="Multiple ROI",
        font=("Kozuka Gothic Pr6N L", 25),
        bg="black",
        fg="#ecf0f1",
    )
    label_amplified.place(relx=0.1, rely=0.05, anchor=tk.N)  # Center the text
    label_entry=tk.Label(
        amplified_page,
        text="Enter the number of regions you want to select:",
        bg="black",
        font=("Kozuka Gothic Pr6N L", 16),
        fg="#ecf0f1",
    )
    label_entry.place(relx=0.05,rely=0.145)
    entry_var = tk.StringVar()
    entry = tk.Entry(amplified_page, textvariable=entry_var, width=20)
    entry.place(relx=0.30,rely=0.15)

    def store_value():
        global stored_value
        try:
            stored_value = int(entry_var.get())
            print(f"Stored Value: {stored_value}")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    store_button = tk.Button(
        amplified_page,
        text="Enter!",
        command=store_value,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="black",
        fg="white",
    )
    
    store_button.place(relx=0.40,rely=0.13)

    new_window_button = tk.Button(
        amplified_page,
        text="MultiROIAnalyzer",
        command=open_new_window,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="black",
        fg="white",
    )
    new_window_button.place(relx=0.5, rely=0.7, anchor=tk.N)
    return amplified_page


def create_results_page(notebook, bg_color):
    # global accuracy_value
    results_page = tk.Frame(notebook, bg=bg_color)

    label_results = tk.Label(
        results_page,
        text="Final Results Page",
        font=("Helvetica", 18),
        bg="#34495e",
        fg="#ecf0f1",
    )
    label_results.pack(pady=20)

    generate_graphs_button1 = tk.Button(
        results_page,
        text="Generate Graphs Original",
        command=generate_graphs_orginial,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="#3498db",
        fg="white",
    )
    generate_graphs_button2 = tk.Button(
        results_page,
        text="Generate Graphs Amplified",
        command=generate_graphs_trimmed,
        width=20,
        height=2,
        font=("Helvetica", 14),
        bg="#3498db",
        fg="white",
    )    
    label_org = tk.Label(
        results_page,
        text="Original Video",
        font=("Helvetica", 14),
        bg="#34495e",
        fg="#ecf0f1",
    )
    label_org.place(relx=0.22,rely=0.20)
    label_new= tk.Label(
        results_page,
        text="Amplified Video",
        font=("Helvetica", 14),
        bg="#34495e",
        fg="#ecf0f1",
    )
    label_new.place(relx=0.72,rely=0.20)
    generate_graphs_button1.place(relx=0.2,rely=0.1)
    generate_graphs_button2.place(relx=0.7,rely=0.1)
    video_player1 = VideoPlayerorg(results_page)
    video_player1.place(relx=0.25,rely=0.6,anchor=tk.CENTER)
    video_player2 = VideoPlayer(results_page)
    video_player2.place(relx=0.75,rely=0.6,anchor=tk.CENTER)
    # calculateaccuracy=tk.Button(
    #     results_page,
    #     text="Calculate accuracy",
    #     command=accuracy_cal,
    #     bg="black",
    #     fg="white",
    # )
    # calculateaccuracy.place(relx=0.5,rely=0.8)
    # label_accuracy=tk.Label(
    #     results_page,
    #     text="Accuracy",
    # )

    # print(accuracy_value)
    return results_page


def create_about_page(notebook, bg_color):
    about_page = tk.Frame(notebook, bg=bg_color)

    label_about = tk.Label(
        about_page,
        text="About Page",
        font=("Helvetica", 18),
        bg="#34495e",
        fg="#ecf0f1",
    )
    label_about.pack(pady=20)

    return about_page


def main():
    global root
    root = tk.Tk()
    root.title("Motion Amplification and Vibration analyser")
    root.geometry("2048x1024")  # Set initial size

    # Use the 'clam' theme for ttk widgets
    style = ttk.Style()
    style.theme_use("clam")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    # Use a cool color palette
    bg_color = "black"  # Dark background color

    home_page = create_home_page(notebook, bg_color)
    amplified_page = create_amplified_page(notebook, bg_color)
    results_page = create_results_page(notebook, bg_color)
    about_page = create_about_page(notebook, bg_color)

    notebook.add(home_page, text="Home")
    notebook.add(
        amplified_page,
        text="MultipleROI",
    )
    notebook.add(results_page, text="Final Results")
    notebook.add(about_page, text="About")

    root.mainloop()


if __name__ == "__main__":
    main()