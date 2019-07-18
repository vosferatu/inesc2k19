import sys
import os
import math
import dlib
import glob
import cv2
import numpy as np
#from PIL import Image

#############
# FUNCTIONS #
#############

# define and return each eye centre coordinates
def eye_centers(eyes):
    # left eye
    left_eye_centre = []
    point37 = eyes[0][1]
    point38 = eyes[0][2]
    point41 = eyes[0][5]
    h_left = point41.y - point37.y
    w_left = point38.x - point37.x
    left_eye_centre = [point37.x + (w_left/2), point37.y + (h_left/2)]

    # right eye
    right_eye_centre = []
    point43 = eyes[1][1]
    point44 = eyes[1][2]
    point47 = eyes[1][5]
    h_right = point47.y - point43.y
    w_right = point44.x - point43.x
    right_eye_centre = [point43.x + (w_right/2), point43.y + (h_right/2)]

    return [left_eye_centre, right_eye_centre]

#########
# TESTS #
#########

# TEST 10 #

def test10(faces, img, eye_cascade):
    eye_counter = 0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_counter += 1
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if(eye_counter == 2):
        return True
    elif(eye_counter == 1):
        return True
    else:
        return False


# test10 (closed eyes)
def eye_Pcnt(gray, eyes):
    # left eye
    point36 = eyes[0][0]
    point37 = eyes[0][1]
    point38 = eyes[0][2]
    point39 = eyes[0][3]
    point40 = eyes[0][4]
    point41 = eyes[0][5]
    h_left = point41.y - point37.y
    w_left = point39.x - point36.x
    corner_left_x = point36.x
    corner_left_y = point37.y
    roi_left = gray[corner_left_y: corner_left_y + h_left, corner_left_x: corner_left_x + w_left]
    left_eye_percentage = sclera_eye_region(roi_left, h_left, w_left)

    # right eye
    point42 = eyes[1][0]
    point43 = eyes[1][1]
    point44 = eyes[1][2]
    point45 = eyes[1][3]
    point46 = eyes[1][4]
    point47 = eyes[1][5]
    h_right = point47.y - point43.y
    w_right = point45.x - point42.x
    corner_right_x = point42.x
    corner_right_y = point43.y
    roi_right = gray[corner_right_y: corner_right_y + h_right, corner_right_x: corner_right_x + w_right]
    right_eye_percentage = sclera_eye_region(roi_right, h_right, w_right)

    # draw eye region
    rect = dlib.rectangle(corner_left_x, corner_left_y,
                          corner_left_x + w_left, corner_left_y + h_left)
    win.add_overlay(rect)
    rect = dlib.rectangle(corner_right_x, corner_right_y,
                          corner_right_x + w_right, corner_right_y + h_right)
    win.add_overlay(rect)

    tests.append(func10(left_eye_percentage) + func10(right_eye_percentage))
    return func10(left_eye_percentage) + func10(right_eye_percentage)

# function to convert closed eye percentage to compliance value
def func10(x):
    return {
        (x >= 0.5): 50,
        (0.3 <= x < 0.5): 35,
        (0.1 <= x < 0.3): 25,
        (0.01 <= x < 0.1): 15,
        (x < 0.01): 0,
    }.get(True)

# define eye region and return sclera percentage
def sclera_eye_region(roi, h, w):
    pixel_counter = 0.00
    sclera_pixel_counter = 0.00
    sclera_percentage = 0.00
    for k in range(0, h):
        for j in range(0, w):
            pixel_counter += 1
            pixel = roi[k, j]
            if(pixel == 0):
                sclera_pixel_counter += 1
    sclera_percentage = sclera_pixel_counter/pixel_counter
    return sclera_percentage

# TEST 12 #

# test12 (roll/pitch/yaw)
def test12(image, points):
    size = image.shape
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                            (points[0].x, points[0].y),     # Nose tip
                            (points[1].x, points[1].y),     # Chin
                            (points[2].x, points[2].y),     # Left eye left corner
                            (points[3].x, points[3].y),     # Right eye right corne
                            (points[4].x, points[4].y),     # Left Mouth corner
                            (points[5].x, points[5].y)      # Right mouth corner
                        ], dtype="double")
 
    # 3D model points.
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                        ])
 
    
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
 
    #print("Camera Matrix :\n {0}".format(camera_matrix))
 
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
 
    #print("Rotation Vector:\n {0}".format(rotation_vector))
    #print("Translation Vector:\n {0}".format(translation_vector))
        
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
    p1 = dlib.point(( int(image_points[0][0]), int(image_points[0][1])))
    p2 = dlib.point(( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])))
    
    line = dlib.line(p1, p2)
    win.add_overlay(line)

    
    pitch_compliance = 0 if (abs(pitch) > 5) else (-20*(abs(pitch))+100)
    print(pitch_compliance)
    roll_compliance = 0 if (abs(roll) > 8) else (-(100/8)*(abs(roll))+100)
    print(roll_compliance)
    yaw_compliance = 0 if (abs(yaw) > 5) else ((-20*abs(int(yaw)))+100)
    print(yaw_compliance)
    return int((pitch_compliance+roll_compliance+yaw_compliance)/3)
    
# TEST 14 #

# test14 (red eyes)
def test14(eyes):
    # left eye
    point37 = eyes[0][1]
    point38 = eyes[0][2]
    point40 = eyes[0][4]
    point41 = eyes[0][5]
    h_left = point41.y - point37.y
    w_left = point38.x - point37.x
    roi_left = img[point37.y:point37.y + h_left, point37.x:point37.x + w_left]
    left_eye_percentage = red_eye_region(roi_left, h_left, w_left)

    # right eye
    point43 = eyes[1][1]
    point44 = eyes[1][2]
    point46 = eyes[1][4]
    point47 = eyes[1][5]
    h_right = point47.y - point43.y
    w_right = point44.x - point43.x
    roi_right = img[point43.y:point43.y +
                    h_right, point43.x:point43.x + w_right]
    right_eye_percentage = red_eye_region(roi_right, h_right, w_right)

    # draw eye region
    rect = dlib.rectangle(point37.x, point37.y,
                          point37.x + w_left, point37.y + h_left)
    win.add_overlay(rect)
    rect = dlib.rectangle(point43.x, point43.y,
                          point43.x + w_right, point43.y + h_right)
    win.add_overlay(rect)

    # mean value for red eye percentage
    mean = (left_eye_percentage+right_eye_percentage)/2
    tests.append(func14(mean))
    return(func14(mean))

# function to convert red eye percentage to compliance value
def func14(x):
    return {
        (x < 0.05): 100,
        (0.05 < x < 0.10): 95,
        (0.10 < x < 0.15): 90,
        (0.15 < x < 0.20): 80,
        (0.20 < x < 0.25): 70,
        (0.25 < x < 0.30): 60,
        (0.30 < x < 0.35): 50,
        (0.35 < x < 0.40): 40,
        (0.40 < x < 0.45): 30,
        (0.45 < x < 0.50): 20,
        (0.50 < x < 0.55): 10,
        (x == 1): 0
    }.get(True)

# define eye region and return red eye percentage
def red_eye_region(roi, h, w):
    pixel_counter = 0.00
    red_pixel_counter = 0.00
    red_percentage = 0.00
    for k in range(0, h):
        for j in range(0, w):
            pixel_counter += 1
            pixel = roi[k, j]
            if(130 < pixel[0] < 255 and pixel[1] < 100 and pixel[2] < 100):
                red_pixel_counter += 1
    red_percentage = red_pixel_counter/pixel_counter
    return red_percentage

# TEST 23 #

# test23 (mouth open)
def test23(mouth):
    # top lip
    point61 = mouth[0][0]
    point62 = mouth[0][1]
    point63 = mouth[0][2]

    # bottom lip
    point65 = mouth[1][0]
    point66 = mouth[1][1]
    point67 = mouth[1][2]

    # mouth width
    width = mouth[1][3].x - mouth[0][3].x

    # left side distance
    left_d = point67.y - point61.y
    left_percentage = func23(left_d, width)
    # centre distance
    centre_d = point66.y - point62.y
    centre_percentage = func23(centre_d, width)
    # right side distance
    right_d = point65.y - point63.y
    right_percentage = func23(right_d, width)

    return int((left_percentage + centre_percentage + right_percentage)/3)

# function to convert open mouth percentage to compliance value
def func23(x, max):
    percentage = (x*100) / max
    if(percentage < 0):
        return 0
    else:
        return percentage

# HAIR TEST #

# hairtest 
def check_hair(hair, eyes):
    # left eye
    point37 = eyes[0][1]
    point38 = eyes[0][2]
    point40 = eyes[0][4]
    point41 = eyes[0][5]
    h_left = point41.y - point37.y
    w_left = point38.x - point37.x
    roi_left = img[point37.y:point37.y + h_left, point37.x:point37.x + w_left]
    left_eye_percentage = hair_eye_region(roi_left, h_left, w_left)
    print(left_eye_percentage)

    # right eye
    point43 = eyes[1][1]
    point44 = eyes[1][2]
    point46 = eyes[1][4]
    point47 = eyes[1][5]
    h_right = point47.y - point43.y
    w_right = point44.x - point43.x
    roi_right = img[point43.y:point43.y +
                    h_right, point43.x:point43.x + w_right]
    right_eye_percentage = hair_eye_region(roi_right, h_right, w_right)
    print(right_eye_percentage)

    # draw eye region
    rect = dlib.rectangle(point37.x, point37.y,
                          point37.x + w_left, point37.y + h_left)
    win.add_overlay(rect)
    rect = dlib.rectangle(point43.x, point43.y,
                          point43.x + w_right, point43.y + h_right)
    win.add_overlay(rect)

    return None

# define eye region and return hair percentage in eye
def hair_eye_region(roi, h, w):
    pixel_counter = 0.00
    hair_pixel_counter = 0.00
    hair_percentage = 0.00
    for k in range(0, h):
        for j in range(0, w):
            pixel_counter += 1
            pixel = roi[k, j]
            if(pixel[0] > 90 and pixel[1] > 90 and pixel[2] > 90):
                hair_pixel_counter += 1
    hair_percentage = hair_pixel_counter/pixel_counter
    return hair_percentage

########
# MAIN #
########

# paths to files
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face = sys.argv[1]
output = sys.argv[2]

# open output file
file = open(output, 'a')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

tests = []
eyes = []
eye_centre_coordinates = []
nose_tip = []
mouth = []
points = []

img = dlib.load_rgb_image(face)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
image = cv2.imread(face)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
IMD = 'IMD436'
# Remove hair with opening
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# Blur the image for smoother ROI
blur = cv2.blur(opening,(15,15))
# Perform another OTSU threshold and search for biggest contour
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)
# Create a new mask for the result image
h, w = image.shape[:2]
mask = np.zeros((h, w), np.uint8)
# Draw the contour on the new mask and perform the bitwise operation
cv2.drawContours(mask, [cnt],-1, 255, -1)
hair_res = cv2.bitwise_and(image, image, mask=mask)


win.clear_overlay()
win.set_image(img)
#win.set_image(hair_res)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    # left eye landmarks
    eyes.append([shape.part(36), shape.part(37), shape.part(38), shape.part(39),
                 shape.part(40), shape.part(41)])
    # right eye landmarks
    eyes.append([shape.part(42), shape.part(43), shape.part(44), shape.part(45),
                 shape.part(46), shape.part(47)])
    # nose tip landmarks
    nose_tip.append([shape.part(30), shape.part(31), shape.part(32), shape.part(33), shape.part(34), shape.part(35)])
    # mouth upper bound landmark
    mouth_upper_bound = shape.part(51)
    # top lip landmarks
    mouth.append([shape.part(61), shape.part(62), shape.part(63), shape.part(60)])
    # bottom lip landmarks
    mouth.append([shape.part(65), shape.part(66), shape.part(67), shape.part(64)])

    points.append(shape.part(30))
    points.append(shape.part(8))
    points.append(shape.part(36))
    points.append(shape.part(45))
    points.append(shape.part(48))
    points.append(shape.part(54))
    # Draw the face landmarks on the screen.
    win.add_overlay(shape)

# run tests
eye_centre_coordinates = eye_centers(eyes)
teste10 = test10(faces, image, eye_cascade) # ESTA A DETETAR OS OLHOS COM UMA haarcascade_eye FILE
test12 = test12(image, points)
test14 = test14(eyes)
test23 = test23(mouth)
#hair = check_hair(hair_res, eyes)

# write results to file
file.write(face)
file.write("\n")
file.write(str(eye_centre_coordinates[0][0]) + " " + str(eye_centre_coordinates[0][1]) +
           " " + str(eye_centre_coordinates[1][0]) + " " + str(eye_centre_coordinates[1][1]))
if(teste10 == True):
    eyes_close = eye_Pcnt(thresh, eyes)
    file.write("\n")
    file.write("Test10 " + str(eyes_close))
elif(teste10 == False):
    eyes_close = eye_Pcnt(thresh, eyes)
    file.write("\n")
    file.write("Test10 " + str(0))
file.write("\n")
file.write("Test12 " + str(test12))
file.write("\n")
file.write("Test14 " + str(test14))
file.write("\n")
file.write("Test23 " + str(test23))
file.write("\n")

file.close()

win.add_overlay(dets)

dlib.hit_enter_to_continue()
