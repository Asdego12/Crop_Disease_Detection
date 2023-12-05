from tkinter import *
import tkinter as tk
from tkvideo import tkvideo
from tkinter import filedialog as fd  # used when opening image
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np


def main_window():
    def center_window_on_screen():
        x_cord = int((screen_width / 2) - (width / 2))
        y_cord = int((screen_height / 2) - (height / 2))
        root.geometry("{}x{}+{}+{}".format(width, height, x_cord, y_cord))

    # Going from splash screen to main menu
    def change_to_menu():
        menu_frame.pack(fill='both', expand=1)

        video_frame.forget()

    # Opens file explorer
    def open_file(_sample, _prediction):
        _prediction.text = ""
        _prediction['text'] = ""
        f_types = [('Jpg Files', '*.jpg'),
                   ('PNG Files', '*.png')]  # type of files to select
        filename = tk.filedialog.askopenfilename(multiple=True, filetypes=f_types)

        for f in filename:
            img = Image.open(f)  # reads the image file
            img = img.resize((224, 224))  # set width & height
            img = ImageTk.PhotoImage(img)
            _sample.image = img
            _sample['image'] = img  # garbage collection
            _sample.file = f
        _sample.pack()

 # Performs Classification
    def classifier(_root, _sample, _prediction):
        model = load_model('C:/Users/Diego/Desktop/Plant_Disease_Detection/CNN_data/Plant_disease_Detect.h5')
        data = []
        labels = ["Apple_Scab", "Apple_BR", "Apple_rust", "Apple_healthy", "Corn_Cercospora", "Corn_Common_rust",
                      "Corn_healthy", "Corn_NLeafB",
                      "Grape_BR", "Grape_Esca", "Grape_healthy", "Grape_leafB", "Peach_BacterialS", "Peach_healthy",
                      "Potato_EB", "Potato_healthy", "Potato_LB",
                      "Tomato_BS", "Tomato_EB", "Tomato_healthy", "Tomato_LB", "Tomato_leafMold",
                      "Tomato_Septoria", "Tomato_SpiderM", "Tomato_TargetS", "Tomato_MosaicV", "TomatoYellow"]

        img = Image.open(_sample.file)  # read the image file
        img = img.resize((224, 224))  # new width & height
        img = np.array(img)
        data.append(np.array(img))
        data = np.array(data)
        prediction = model.predict(data)

        predict_index = np.argmax(prediction)
        _prediction.text = labels[predict_index]
        _prediction['text'] = labels[predict_index]
        _prediction.pack()

        print(prediction)

    # APPLICATION
    root = tk.Tk()

    root.title("Crop_Disease_Detection")

    width, height = [1028, 580]
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_window_on_screen()

    # Splash Screen-video
    video_frame = tk.Frame(root)

    video_label = tk.Label(video_frame)
    video_label.pack()

    player = tkvideo("wheel1.mp4", video_label, loop=FALSE,
                     size=(1028, 580))
    player.play()

    # Menu
    menu_frame = tk.Frame(root, bg="#3d6466")
    label_menu = tk.Label(menu_frame, bg="#3d6466", fg="white",
                          text='Crop Disease Detection.Press on button to add image:',
                          font="Raleway")
    label_menu.pack(pady=20)

    # Button for inserting images
    btn_insert = tk.Button(menu_frame, bg="#3d6466", fg="white",
                           text='Insert Image', command=lambda: open_file(my_image, my_prediction))
    btn_insert.pack(pady=10)

    my_image = tk.Label(menu_frame)

    # Button for classification
    btn_classify = tk.Button(menu_frame, bg="#3d6466", fg="white",
                             text='Classify', command=lambda: classifier(menu_frame, my_image, my_prediction))
    btn_classify.pack(pady=50)

    my_prediction = tk.Label(menu_frame)



    # Splash Screen Timer
    video_frame.after(5000, change_to_menu)
    video_frame.pack(fill='both', expand=1)


main_window()

mainloop()
