import tensorflow.keras, cv2
from PIL import Image, ImageOps
import numpy as np


##Capture from camera
video = cv2.VideoCapture(0)

##Face Cascade
face_cascade = cv2.CascadeClassifier("C:\\Users\\Swapnil\\PycharmProjects\\pythondemo\\venv\\NumpyDemo\\facial_recognition_model.xml")

##Path for labels
label_path = "C:\\Users\\Swapnil\\PycharmProjects\\pythondemo\\venv\\OpenCV\\labels.txt"

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224), dtype=np.float32)

while True:
    check, frame = video.read()

    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = cv2.resize(gray_image, size)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    #normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.0)

    face = face_cascade.detectMultiScale(frame,scaleFactor=1.05,minNeighbors=3)

    for (x, y, w, h) in face:
        rectangle = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        #Load the image into the array
        data[0] = image_array

        # run the inference
        prediction = model.predict(data)
        print(prediction)

        ##get index number
        for value in prediction:
            index = np.where(value == np.amax(value))
            index_number = int(index[0])

        ##Get label
        with open(label_path, "r") as f:
            labels = f.readlines()
            for i in range(len(labels)):
               label_name = labels[index_number].split()[1]

        ##Show name along with frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = label_name
        color = (255, 255, 255)
        stroke = 1
        cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

    ##Show frame
    cv2.imshow('Video', frame)

    ##Stream video until q is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

video.release()

cv2.destroyAllWindows()