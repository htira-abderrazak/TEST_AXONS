import cv2
import wave
import statistics as stats
import numpy as np 
import matplotlib.pyplot as plt 
from moviepy.editor import VideoFileClip

#Read the Video

vid = cv2.VideoCapture('video1.mp4') 


#Extract the frames from the video

frames = [] 
while vid.isOpened():
    ret, frame = vid.read()
    if ret == True:
        frames.append(frame) 
    else: 
        break
vid.release()

#Calculate the amount of blinks in each frame

blinksperframe = [] 
for frame in frames: 
    # Detect the face in the frame
    facecascade = cv2.CascadeClassifier('C:\\Users\\asus\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml') 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5) 
    # Detect the eyes in the face  
    eyecascade = cv2.CascadeClassifier('C:\\Users\\asus\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml') 
    eyes = eyecascade.detectMultiScale(gray, 1.3, 5) 
    # Count the number of blinks 
    blinkcount = 2 
    for (ex, ey, ew, eh) in eyes: 
        blinkcount = blinkcount - 1 
    # Append the blink count to the list
    blinksperframe.append(blinkcount) 




 # create an empty list to store hand movement data

handmovementdata = []


 # loop through the frames of the video

for frame in frames: 
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect hand movements
    hand_movement = cv2.calcOpticalFlowFarneback(gray,frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # calculate the magnitude of the movement
    magnitude = (hand_movement[:,:,0]**2 + hand_movement[:,:,1]**2)**0.5

    # append the data to the list
    handmovementdata.append(magnitude.mean())


#open an MP4 file and save it as a WAV file

video1 = VideoFileClip("video1.mp4")
audio = video1.audio
audio.write_audiofile("audio.wav")
video = wave.open('audio.wav', 'rb')


#create an empty list to store sound data

sound_data = []


#loop through the frames of the video

for i in range(video.getnframes()):
    # read the frame
    frame = video.readframes(1)
    # convert the frame to an array
    framearray = np.frombuffer(frame, dtype="int16")
    # calculate the magnitude of the sound
    magnitude = (framearray[0]**2 + framearray[1]**2)**0.5
    # append the data to the list
    sound_data.append(magnitude)


#close the video
video.close()

#create an empty list to store sound data body data

body_data = []


#loop through the frames of the video

for frame in frames: 
    # calculate the magnitude of the body movement
    magnitude = np.sum(frame) / 255
    # append the data to the list
    body_data.append(magnitude)
    
    
#create an empty list to store sound data eye  movement   

eye_data = []


#loop through the frames of the video

for frame in frames:

    # detect eyes in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = cv2.CascadeClassifier('C:\\Users\\asus\\anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml').detectMultiScale(gray, 1.3, 5)
    # calculate the direction of the eyes
    if len(eyes) > 0:
        direction = np.sum(frame[eyes[0][1]:eyes[0][3],eyes[0][0]:eyes[0][2]]) / 255
        # append the data to the list
        eye_data.append(direction)

        
#final result

final_stress = stats.mean(eye_data)*0.2 + stats.mean(body_data)*0.2 + stats.mean(sound_data)*0.2 + stats.mean(handmovementdata)*0.2 + stats.mean(blinksperframe)*0.2
print(final_stress)


#plot the graph

plt.plot(eye_data)
plt.title('Eye Movement Graph')
plt.xlabel('Time (in seconds)')
plt.ylabel('Stress Level(in Eye Movement/second)')
plt.show()


plt.plot(body_data)
plt.title('Body Movement Graph')
plt.xlabel('Time (in seconds)')
plt.ylabel('Stress Level(in Body Movement/second)')
plt.show()


plt.plot(sound_data)
plt.title('Sound Graph')
plt.xlabel('Time (in seconds)')
plt.ylabel('Stress Level(in Sound/second)')
plt.show()


plt.plot(handmovementdata)
plt.title('Hand Movement Graph')
plt.xlabel('Time (in seconds)')
plt.ylabel('Stress Level(in Hand Movement/second)')
plt.show()

plt.plot(blinksperframe) 
plt.title('blinks')
plt.xlabel('Time (in seconds)') 
plt.ylabel('Stress Level (in Eye Movement Graph/second)') 
plt.show()

#Aggregated results

plt.plot(eye_data, color='r',label='Eye Movement')
plt.plot(body_data, color='g',label='Body Movement')
plt.plot(sound_data, color='b',label='Sound')
plt.plot(handmovementdata, color='c',label='Hand Movement')
plt.plot(blinksperframe, color='m',label='blinks')
plt.title('Aggregated results')
plt.xlabel('Time (in seconds)')
plt.ylabel('Stress Level')
plt.legend()
plt.show()