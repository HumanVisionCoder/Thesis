import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('KN5Jan7_x264.mp4')

#ammount of frames per second
framespersecond= int(cap.get(cv2.CAP_PROP_FPS))

print(framespersecond)

#%%
try:
    if not os.path.exists('Data'): #is there a file named data
        os.makedirs('Data') #creates file 
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while currentFrame!= 22475:
    # Capture frame-by-frame
    ret, frame = cap.read()


    if currentFrame % framespersecond == 0:
        name = 'C:/Users/tobia/OneDrive/Documenten/thesis/Data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        test = frame
        cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1
    
    #print done when there are enough frames
    if currentFrame == 22475:
        print("It's done check the folder")
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

