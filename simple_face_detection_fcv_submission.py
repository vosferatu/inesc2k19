import cv2
import sys
import os
import matplotlib.pyplot as plt

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Define function that will do detection
def detect(gray, frame):
    """ Input = greyscale image or frame from video stream
        Output = Image with rectangle box in the face
    """
    # Now get the tuples that detect the faces using above cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces are the tuples of 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face
    # grey means the input image to the detector
    # 1.3 is the kernel size or size of image reduced when applying the detection
    # 5 is the number of neighbors after which we accept that is a face

    #visualize de rectangles
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_roi(roi_color, ex, ey, ew, eh)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.imshow('img',frame)
    cv2.waitKey(0)

    # Now iterate over the faces and detect eyes
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness

        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes now
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        eye_center = {}
        
        for counter, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_center[counter] = (ex+(ew/2), ey+(eh/2))
        if(len(eye_center) < 2):
            eye_center[0] = (0,0)
            eye_center[1] = (0,0)
    return eye_center[0], eye_center[1], frame

def eye_roi(region, ex, ey, ew, eh):
    roi = region[ey:ey+eh, ex:ex+ew]
    for i in range(0, eh):
        for j in range(0,ew):
            pixel = roi[i,j]
            if(150 < pixel[2] < 255 and pixel[0] < 100 and pixel[1] < 100):
                roi[i,j] = [255,255,0]

    return None

if(len(sys.argv) != 3):
    print('Check.exe <faceimagefile> <outputfile>')
    sys.exit(0)

image_file_name = sys.argv[1]
result_file_name = sys.argv[2]


frame = cv2.imread(image_file_name)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

(left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y), canvas = detect(gray, frame)

if(left_eye_center_x > right_eye_center_x):
    ((left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y)) = \
        ((right_eye_center_x, right_eye_center_y), (left_eye_center_x, left_eye_center_y))

file = open(result_file_name,'a')

result = os.path.basename(image_file_name) + " 1 " + str(left_eye_center_x) + " " + str(left_eye_center_y) + " " + \
            str(right_eye_center_x) + " " + str(right_eye_center_y)

for test in range(2,25):
    result = result + " - "

file.write(result)
file.write("\n")

file.close()