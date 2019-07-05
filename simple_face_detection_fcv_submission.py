import cv2
import sys
import os

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


'''
    Each line of the outputfile must contain the following fields separated by a blank space character:
    ImageName is the image file name (e.g., image1.png).
    RetVal is an integer value indicating if the input image can be processed or not by the executable3:
    1 if the image can be processed.
    0 if the image cannot be processed and no more information is available.
    -1 if the image cannot be processed due to unsupported image size.
    -2 if the image cannot be processed due to unsupported image format.
    -3 if the image cannot be processed due to unuseful image content.
    LE_x is an integer value indicating the X coordinate (in pixels) of the left eye center4.
    LE_y is an integer value indicating the Y coordinate (in pixels) of the left eye center4.
    RE_x is an integer value indicating the X coordinate (in pixels) of the right eye center4.
    RE_y is an integer value indicating the Y coordinate (in pixels) of the right eye center4.
    Test_2 is an integer value in the range [0;100] indicating the compliance degree of the input image with respect to the Compliance Test 2. 0 means no compliancy, 100 maximum compliancy. Moreover, the following special characters have been defined:
    ‘-‘ if the executable is not able to evaluate this requirement.
    ‘?’ if, usually, the executable is able to evaluate this requirement but it was not able to evaluate it on the current input image for an identified failure (e.g., the SDK is not able to evaluate the presence of red eyes since the eyes are closed).
    ‘!’ if, usually, the executable is able to evaluate this requirement but it was not able to evaluate it on the current input image for an unknown failure.
    Test_3 analogous to test 2 but related to test 3.
    ...
    Test_24 analogous to test 2 but related to test 24.
    '''
file = open(result_file_name,'a')

result = os.path.basename(image_file_name) + " 1 " + str(left_eye_center_x) + " " + str(left_eye_center_y) + " " + \
            str(right_eye_center_x) + " " + str(right_eye_center_y)

for test in range(2,25):
    result = result + " - "

file.write(result)
file.write("\n")

file.close()
