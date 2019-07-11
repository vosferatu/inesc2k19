import sys
import os
import dlib
import glob
import cv2

# function to convert open mouth percentage to compliance value
def func9(x):
    return {
        (20 <= x < 30): 0,
        (10 <= x < 20): 10,
        (5 <= x < 10): 20,
        (x < 5): 30,
    }.get(True)

# function to convert closed eye percentage to compliance value
def func10(x):
    return {
        (14 <= x < 20): 50,
        (10 <= x < 14): 25,
        (5 <= x < 10): 15,
        (x < 5): 0,
    }.get(True)

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

# define and return each eye centre coordinates
def eye_centers(eyes):
    # left eye
    left_eye_centre = []
    point37 = eyes[0][0]
    point38 = eyes[0][1]
    point41 = eyes[0][3]
    h_left = point41.y - point37.y
    w_left = point38.x - point37.x
    left_eye_centre = [point37.x + (w_left/2), point37.y + (h_left/2)]

    # right eye
    right_eye_centre = []
    point43 = eyes[1][0]
    point44 = eyes[1][1]
    point47 = eyes[1][3]
    h_right = point47.y - point43.y
    w_right = point44.x - point43.x
    right_eye_centre = [point43.x + (w_right/2), point43.y + (h_right/2)]

    return [left_eye_centre, right_eye_centre]

# define eye region and return red eye percentage
def eye_region(roi, h, w):
    pixel_counter = 0.00
    red_pixel_counter = 0.00
    red_percentage = 0.00
    for k in range(0, h):
        for j in range(0, w):
            pixel_counter += 1
            pixel = roi[k, j]
            if(130 < pixel[0] < 255 and pixel[1] < 100 and pixel[2] < 100):
                roi[k, j] = [255, 255, 0]
                red_pixel_counter += 1
    red_percentage = red_pixel_counter/pixel_counter
    return red_percentage

# test9 (hair across eyes)
def test9(mouth):
    # top lip
    point61 = mouth[0][0]
    point62 = mouth[0][1]
    point63 = mouth[0][2]
    
    # bottom lip 
    point65 = mouth[1][0]
    point66 = mouth[1][1]
    point67 = mouth[1][2]

    # left side distance
    left_d = point67.y - point61.y
    left_percentage = func9(left_d)
    # centre distance
    centre_d = point66.y - point62.y
    centre_percentage = func9(centre_d)
    # right side distance 
    right_d = point65.y - point63.y
    right_percentage = func9(right_d)
    
    return (left_percentage + centre_percentage + right_percentage)

# test10 (closed eyes)
def test10(eyes):
    # left eye
    point37 = eyes[0][0]
    point38 = eyes[0][1]
    point40 = eyes[0][2]
    point41 = eyes[0][3]
    h_left = point41.y - point37.y
    left_percentage = func10(h_left)

    # right eye
    point43 = eyes[1][0]
    point44 = eyes[1][1]
    point46 = eyes[1][2]
    point47 = eyes[1][3]
    h_right = point47.y - point43.y
    right_percentage = func10(h_right)

    tests.append(left_percentage + right_percentage)
    return (left_percentage + right_percentage)

# test14 (red eyes)
def test14(eyes):
    # left eye
    point37 = eyes[0][0]
    point38 = eyes[0][1]
    point40 = eyes[0][2]
    point41 = eyes[0][3]
    h_left = point41.y - point37.y
    w_left = point38.x - point37.x
    roi_left = img[point37.y:point37.y + h_left, point37.x:point37.x + w_left]
    left_eye_percentage = eye_region(roi_left, h_left, w_left)

    # right eye
    point43 = eyes[1][0]
    point44 = eyes[1][1]
    point46 = eyes[1][2]
    point47 = eyes[1][3]
    h_right = point47.y - point43.y
    w_right = point44.x - point43.x
    roi_right = img[point43.y:point43.y +
                    h_right, point43.x:point43.x + w_right]
    right_eye_percentage = eye_region(roi_right, h_right, w_right)

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
mouth = []

img = dlib.load_rgb_image(face)

win.clear_overlay()
win.set_image(img)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, d)
    # left eye landmarks
    eyes.append([shape.part(37), shape.part(38),
                 shape.part(40), shape.part(41)])
    # right eye landmarks
    eyes.append([shape.part(43), shape.part(44),
                 shape.part(46), shape.part(47)])
    # top lip landmarks
    mouth.append([shape.part(61), shape.part(62), shape.part(63)])
    # bottom lip landmarks
    mouth.append([shape.part(65), shape.part(66), shape.part(67)])
    # Draw the face landmarks on the screen.
    win.add_overlay(shape)

# run tests
eye_centre_coordinates = eye_centers(eyes)
test9 = test9(mouth)
test10 = test10(eyes)
test14 = test14(eyes)


# write results to file
file.write(face)
file.write("\n")
file.write(str(eye_centre_coordinates[0][0]) + " " + str(eye_centre_coordinates[0][1]) +
           " " + str(eye_centre_coordinates[1][0]) + " " + str(eye_centre_coordinates[1][1]))
file.write("\n")
file.write("Test9 " + str(test9))
file.write("\n")
file.write("Test10 " + str(test10))
file.write("\n")
file.write("Test14 " + str(test14))
file.write("\n")
file.close()

win.add_overlay(dets)

dlib.hit_enter_to_continue()
