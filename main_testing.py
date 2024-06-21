from tkinter import *
import tkinter as tk
import cv2
from tkinter import filedialog
import os
import numpy as np                            
import imutils
from PIL import ImageTk, Image, ImageFile
global rep
from numpy import load
from skimage.color import rgb2gray
from tkinter import messagebox
from sklearn.datasets import load_files       
from glob import glob           
from tkinter import Label
import numpy as np
from glob import glob
from keras.preprocessing import image
from tensorflow.keras.models import load_model
from datetime import datetime
import moviepy.editor as mp
import os
import ntpath
import scipy.io.wavfile as wav
import ntpath
import os
from numpy.lib import stride_tricks
from matplotlib import pyplot as plt

# short-time Fourier Transformation(STFT)
def stft(sig, frame_size, overlap_factor=0.5, window=np.hanning):
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap_factor * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frame_size), strides=(samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

def log_scale_spec(spec, sr=44100, factor=20.):
    time_bins, frequency_bins = np.shape(spec)

    scale = np.linspace(0, 1, frequency_bins) ** factor
    scale *= (frequency_bins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # Creates spectrogram with new frequency bins
    new_spectrogram = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            new_spectrogram[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            new_spectrogram[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # Lists center frequency of bins
    all_frequencies = np.abs(np.fft.fftfreq(frequency_bins*2, 1./sr)[:frequency_bins+1])
    frequemcies = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            frequemcies += [np.mean(all_frequencies[int(scale[i]):])]
        else:
            frequemcies += [np.mean(all_frequencies[int(scale[i]):int(scale[i+1])])]

    return new_spectrogram, frequemcies

def plot_audio_spectrogram(audio_path, binsize=2**10, plot_path=None, argv = '', colormap="jet"):
    sample_rate, samples = wav.read(audio_path)
    s = stft(samples, binsize)
    new_spectrogram, freq = log_scale_spec(s, factor=1.0, sr=sample_rate)
    data = 20. * np.log10(np.abs(new_spectrogram) / 10e+6)  #dBFS

    time_bins, freq_bins = np.shape(data)

    print("Time bins: ", time_bins)
    print("Frequency bins: ", freq_bins)
    print("Sample rate: ", sample_rate)
    print("Samples: ",len(samples))
    # horizontal resolution correlated with audio length  (samples / sample length = audio length in seconds). If you use this(I've no idea why). I highly recommend to use "gaussian" interpolation.
    #plt.figure(figsize=(len(samples) / sample_rate, freq_bins / 100))
    plt.figure(figsize=(time_bins/100, freq_bins/100)) # resolution equal to audio data resolution, dpi=100 as default
    plt.imshow(np.transpose(data), origin="lower", aspect="auto", cmap=colormap, interpolation="none")

    # Labels
    plt.xlabel("Time(s)")
    plt.ylabel("Frequency(Hz)")
    plt.xlim([0, time_bins-1])
    plt.ylim([0, freq_bins])


    if 'l' in argv: # Add Labels
        plt.colorbar().ax.set_xlabel('dBFS')
    else: # No Labels
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.axis('off')



    x_locations = np.float32(np.linspace(0, time_bins-1, 10))
    plt.xticks(x_locations, ["%.02f" % l for l in ((x_locations*len(samples)/time_bins)+(0.5*binsize))/sample_rate])
    y_locations = np.int16(np.round(np.linspace(0, freq_bins-1, 20)))
    plt.yticks(y_locations, ["%.02f" % freq[i] for i in y_locations])


    if 's' in argv: # Save
        print('Unlabeled output saved as.png')
        print(plot_path)
        plt.savefig(plot_path)
    else:
        print('Graphic interface...')
        plt.show()

    plt.clf()

    return data


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.pack(fill=BOTH, expand=1) 
##        self.config(bg="white")
        image_path = "image.jpg"  # Replace this with the path to your image
        img = Image.open(image_path)
        # Resize the image to fit the window size (1200x800)
        img = img.resize((1400, 800))  
        # Convert the image to a PhotoImage object to be used in Tkinter
        self.background_image = ImageTk.PhotoImage(img)

        # Create a label with the background image
        self.background_label = Label(self, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1) 
        
        
        # changing the title of our master widget
        self.master.title("Deep Fake Detection")
        
        self.pack(fill=BOTH, expand=1)
        
        w = tk.Label(root, 
		 text="Deep Fake Detection",
		 fg="black",    
		 bg="#E0E9F0",
		 font="Helvetica 20 bold italic")
        w.pack()
        w.place(x=600, y=0)

        # creating a button instance
        quitButton = Button(self,command=self.load, text="Capture Frame",bg="#FFFF00",fg="#4C0099",activebackground="dark red",width=20)
        quitButton.place(x=435, y=220,anchor="w")
##        quitButton = Button(self,command=self.classification,text="predict",bg="#FFFF00",fg="#4C0099",activebackground="dark red",width=20)
##        quitButton.place(x=50,y=250,anchor="w")
       
        load = Image.open(r"D:\fake\logo.jfif")
        render = ImageTk.PhotoImage(load)

##        t1 = tk.Label(root, text="Captured Frame",fg="black", bg='#FFFFFF',font=("Times New Roman", 13, "bold italic"),width=12)
##        t1.pack()
##        t1.place(x=285, y=50)

        image1 = Label(self, image=render, borderwidth=15, highlightthickness=5, height=150, width=150)
        image1.image = render
        image1.place(x=420, y=250)


        

#       Functions

    def load(self, event=None):
        global rep
        global T, rep, output_filename
        
        rep = filedialog.askopenfilenames()
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(rep[0])
        # Read until video is completed
        cn=0
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if cn == 20:
                # Display the resulting frame
##                cv2.imshow('Frame', frame)
                output_filename='temp.jpg'
                cv2.imwrite(output_filename,frame)
                break
            cn+=1
                # Press Q on keyboard to exit

        # When everything done, release
        # the video capture object
        cap.release()
        img = cv2.imread(output_filename)

        # Get the list of classes from the directory structure
        clas1 = ['Real Video', 'RealVideo', 'fake video', 'fake video']
        from keras.preprocessing import image                  
        from tqdm import tqdm
        def path_to_tensor(img_path, width=224, height=224):
            print(img_path)
            img = image.load_img(img_path, target_size=(width, height))
            x = image.img_to_array(img)
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths, width=224, height=224):
            list_of_tensors = [path_to_tensor(img_paths, width, height)]
            return np.vstack(list_of_tensors)
        from tensorflow.keras.models import load_model

        # Load the pre-trained model
        model1 = load_model('trained_model_CNN.h5')

        # Load and preprocess the test image
        test_tensors1 = paths_to_tensor(output_filename) / 255

        # Make predictions using the model
        pred1 = model1.predict(test_tensors1)

        # Get the index of the predicted class
        predicted_class_index1 = np.argmax(pred1)

        # Display the predicted class
        res3 = 'The Predicted video is ' + clas1[predicted_class_index1]
        print(res3)
        #Input_img=img.copy()
        print(rep[0])
        self.from_array = Image.fromarray(cv2.resize(img,(250,250)))
        load = Image.open(output_filename)
        render = ImageTk.PhotoImage(load.resize((250,250)))
        image1=Label(self, image=render,borderwidth=15, highlightthickness=5, height=250, width=250)
        image1.image = render
        image1.place(x=420, y=250)
        
        # Function to extract audio from video
        def extract_audio(video_path, output_path, start_time, end_time):
            video = mp.VideoFileClip(video_path)
            audio = video.audio.subclip(start_time, end_time)
            audio.write_audiofile(output_path)
            video.close()
        extract_audio(rep[0], 'audio.wav', 0,3)
        



        clas1 = ['Real Audio', 'FakeAudio', 'fake audio', 'real audio']

        #inp=input('Do want to test Real audio (Enter1)')
        filename="audio.wav"


        ims = plot_audio_spectrogram(filename, 2**10, ntpath.basename(filename.replace('.wav','')) + '.png',  's')

        filename=ntpath.basename(filename.replace('.wav','')) + '.png'

        from tensorflow.keras.models import load_model
        model = load_model('trained_model_DNN.h5')


        test_tensors = paths_to_tensor('audio.png')/255
        pred=model.predict(test_tensors)
        print(np.argmax(pred))
        print(res3+'   '+str(clas1[np.argmax(pred)]))
        res3=res3+'   '+str(clas1[np.argmax(pred)])

        # Display the predicted class in the GUI
        T = Text(self, height=5, width=40)
        T.place(x=420, y=550)
        T.insert(END, res3)

                    
                    
        def close_window(): 
           Window.destroy()
  

                
root = Tk()
root.geometry("1400x720")
app = Window(root)
root.mainloop()

        
