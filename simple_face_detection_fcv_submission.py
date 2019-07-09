import cv2
import sys
import os
import matplotlib.pyplot as plt

# Define tests array
tests = []

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

    # visualize de rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        eyes_red_percentages = []
        for (ex, ey, ew, eh) in eyes:
            eyes_red_percentages.append(red_eye_test(roi_color, ex, ey, ew, eh))
            centre = (ex+(ew/2), ey+(eh/2))
            new_ex = centre[0] - (ew/4)
            new_ey = centre[1] - (eh/4)
            cv2.rectangle(roi_color, (new_ex, new_ey), (new_ex+ew/2, new_ey+eh/2), (0, 255, 0), 2)
            cv2.imshow('img', frame)
        mean = sum(eyes_red_percentages)/2
        value = 0
        if(mean < 0.02):
            value = 100
        elif(mean < 0.04):
            value = 90
        elif(mean < 0.05):
            value = 80
        elif(mean < 0.06):
            value = 70
        elif(mean < 0.07):
            value = 60
        elif(mean < 0.08):
            value = 50
        elif(mean < 0.09):
            value = 40
        elif(mean < 0.1):
            value = 30
        elif(mean < 0.2):
            value = 20
        elif(mean < 0.3):
            value = 10
        elif(mean < 0.4):
            value = 0
        tests.append(["Test_14", value])
    cv2.waitKey(0)

    # Now iterate over the faces and detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness

        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes now
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)

        eye_center = {}

        for counter, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_center[counter] = (ex+(ew/2), ey+(eh/2))
        if(len(eye_center) < 2):
            eye_center[0] = (0, 0)
            eye_center[1] = (0, 0)
    return eye_center[0], eye_center[1], frame

def red_eye_test(region, ex, ey, ew, eh):
    pixel_counter = 0.00
    red_pixel_counter = 0.00
    red_percentage = 0.00
    centre = (ex+(ew/2), ey+(eh/2))
    new_ex = centre[0] - (ew/4)
    new_ey = centre[1] - (eh/4)
    roi = region[new_ey:ey+eh, new_ex:ex+ew]
    for i in range(0, eh/2):
        for j in range(0, ew/2):
            pixel_counter += 1
            pixel = roi[i, j]
            if(130 < pixel[2] < 255 and pixel[0] < 100 and pixel[1] < 100):
                roi[i, j] = [255, 255, 0]
                red_pixel_counter += 1
    red_percentage = red_pixel_counter/pixel_counter
    return red_percentage

def red_eye_test_old(region, ex, ey, ew, eh):
    pixel_counter = 0.00
    red_pixel_counter = 0.00
    red_percentage = 0.00
    roi = region[ey:ey+eh, ex:ex+ew]
    for i in range(0, eh):
        for j in range(0, ew):
            pixel_counter += 1
            pixel = roi[i, j]
            if(150 < pixel[2] < 255 and pixel[0] < 100 and pixel[1] < 100):
                roi[i, j] = [255, 255, 0]
                red_pixel_counter += 1
    red_percentage = red_pixel_counter/pixel_counter
    return red_percentage


if(len(sys.argv) != 3):
    print('Check.exe <faceimagefile> <outputfile>')
    sys.exit(0)

image_file_name = sys.argv[1]
result_file_name = sys.argv[2]


frame = cv2.imread(image_file_name)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

(left_eye_center_x, left_eye_center_y), (right_eye_center_x,
                                         right_eye_center_y), canvas = detect(gray, frame)

if(left_eye_center_x > right_eye_center_x):
    ((left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y)) = \
        ((right_eye_center_x, right_eye_center_y),
         (left_eye_center_x, left_eye_center_y))

file = open(result_file_name, 'a')

result = os.path.basename(image_file_name) + " 1 " + str(left_eye_center_x) + " " + str(left_eye_center_y) + " " + \
    str(right_eye_center_x) + " " + str(right_eye_center_y)

file.write(result)
file.write("\n")

for test in tests:
    file.write(test[0] + " " + str(test[1]))

file.write("\n")

file.close()
