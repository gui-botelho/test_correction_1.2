# Install OPENCV and figure out how to use it. xD
import copy
import tkinter

import numpy as np
import cv2 as cv
from imutils.perspective import four_point_transform
from tkinter import *
import re


def get_answer_key():
    key_feed = cv.VideoCapture(0)
    detector = cv.QRCodeDetector()

    while True:
        _, key_frame = key_feed.read()
        cv.putText(key_frame, 'Press space to exit.',
                   (key_frame.shape[0] // 2, key_frame.shape[0] - 10),
                   cv.FONT_HERSHEY_PLAIN, 1,
                   (200, 255, 0))
        cv.imshow('Scan QR code', key_frame)
        value, points, straight_qr_code = detector.detectAndDecode(img=key_frame)

        if cv.waitKey(1) == ord(" "):
            exit(0)

        if value and re.findall('^gui', value):
            answer_key = re.split('_', value)
            print('Got answer key.')
            print(answer_key)

            cv.destroyAllWindows()
            return answer_key
        else:
            continue


def binary_clean(frame):
    # returns the list of lists with any pixel value above a threshold as 1 and any below it as 0.
    _, binary = cv.threshold(frame, 150, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return binary


def get_rectangular_contours(contours):
    rectangular_contour = []
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        approximation = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            rectangular_contour.append(approximation)

    rectangular_contour = sorted(rectangular_contour, key=cv.contourArea, reverse=True)
    return rectangular_contour


def get_video_feed():
    # initiates video feed from native camera (0)
    camera_feed = cv.VideoCapture(0)
    while True:
        # Gets new frames from camera until exit condition is met.
        isTrue, frame = camera_feed.read()
        # frame here is the image captured by the camera
        cv.putText(frame, 'Press space to exit.',
                   (frame.shape[0] // 2, frame.shape[0] - 10),
                   cv.FONT_HERSHEY_PLAIN, 1,
                   (200, 255, 0))

        cv.imshow('Camera 1 feed', frame)

        # makes image grayscale
        b_y_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # cleans out the grayscale based on a threshold value
        bin_frame = binary_clean(b_y_frame)
        # Creates a gaussian blue on the black and white image to increase
        # edge detection sensitivity.
        b_y_blur = cv.GaussianBlur(bin_frame, (3, 3), 0)

        # Detects the edges by the amount of change in the value of adjoining pixels.
        # Here, since everything is either black or white, it is easier to detect the edges.

        def get_edges_and_contours(source_image):
            edges = cv.Canny(source_image, 100, 255)
            contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            return contours, edges

        contours, edges = get_edges_and_contours(b_y_blur)

        rectangular_contours = get_rectangular_contours(contours)

        if rectangular_contours:

            analysis_frame = bin_frame.copy()

            birds_eye_view = four_point_transform(analysis_frame, rectangular_contours[0].reshape(4, 2))
            rows, columns = birds_eye_view.shape

            marker_proportion = {
                'marker1': [int((0.3 / 9.2) * rows), int((0.45 / 9.2) * rows),
                            int((0.35 / 8.8) * columns), int((0.7 / 8.8) * columns)],
                'marker2': [int((8.65 / 9.2) * rows), int((8.85 / 9.2) * rows),
                            int((0.35 / 8.8) * columns), int((0.8 / 8.8) * columns)],
                'marker3': [int((8.6 / 9.2) * rows), int((8.9 / 9.2) * rows),
                            int((4.35 / 8.8) * columns), int((4.8 / 8.8) * columns)],
                'white': [int((8.55 / 9.2) * rows), int((8.8 / 9.2) * rows),
                          int((.95 / 8.8) * columns), int((3.4 / 8.8) * columns)]
            }



            marker_template = [np.zeros((marker_proportion['marker1'][1] - marker_proportion['marker1'][0],
                                         marker_proportion['marker1'][3] - marker_proportion['marker1'][2]), int),
                               np.zeros((marker_proportion['marker2'][1] - marker_proportion['marker2'][0],
                                         marker_proportion['marker2'][3] - marker_proportion['marker2'][2]), int),
                               np.zeros((marker_proportion['marker3'][1] - marker_proportion['marker3'][0],
                                         marker_proportion['marker3'][3] - marker_proportion['marker3'][2]), int),
                               np.full((marker_proportion['white'][1] - marker_proportion['white'][0],
                                        marker_proportion['white'][3] - marker_proportion['white'][2]), 255, int)]

            def identify_all_markers():
                markers = [birds_eye_view[marker_proportion[key][0]:marker_proportion[key][1],
                           marker_proportion[key][2]:marker_proportion[key][3]] for key in marker_proportion]

                if markers[0].all() == marker_template[0].all() and 6 < len(markers[0]) < 9:
                    print('True 1')
                    cv.imshow('marker 1', markers[0])

                    if markers[1].all() == marker_template[1].all():
                        print('True 2')
                        cv.imshow('marker 2', markers[0])
                        if markers[2].all() == marker_template[2].all():
                            print('True 3')
                            cv.imshow('marker 3', markers[0])
                            if markers[3].all() == marker_template[3].all():
                                print('True 4')
                                cv.imshow('marker 4                                      ', markers[0])
                                marked = birds_eye_view.copy()

                                return True, marked

            found = identify_all_markers()
            if found:
                cv.destroyAllWindows()
                camera_feed.release()
                return found

        # now that I can find the marker, I just need to make a copy of the answer sheet image when the marker
        # is detected, and proceed to separate into rows and columns as well as get the answer. Then, maybe,
        # figure out how to make it work in android T-T

        if cv.waitKey(5) & 0xFF == ord(' '):
            exit(0)


def get_assigned_bubbles(image, answer_key):
    rows, columns = image.shape
    question_counter = 1
    bubbles = []
    answers = [0] * 20

    horizontal_buffer = int(0.1094 * columns)
    vertical_buffer = int(0.083311018 * rows)

    new_image_left = image[vertical_buffer::, horizontal_buffer:int(columns / 2)]
    left_rows, left_columns = new_image_left.shape
    left_single_row = int(left_rows / 10)
    left_single_column = int(left_columns / 5)

    left_copy = copy.copy(new_image_left)
    rgb_left = cv.cvtColor(left_copy, cv.COLOR_GRAY2RGB)

    new_image_right = image[vertical_buffer::, horizontal_buffer + int(columns / 2)::]
    right_rows, right_columns = new_image_right.shape
    right_single_row = int(right_rows / 11)
    right_single_column = int(right_columns / 5)

    right_copy = copy.copy(new_image_right)
    rgb_right = cv.cvtColor(right_copy, cv.COLOR_GRAY2RGB)

    up_down_buffer = 0
    left_right_buffer = 0

    while question_counter < 21:
        if question_counter < 10:
            for row in range(0, 10):
                if row != 0:
                    up_down_buffer -= 2
                else:
                    up_down_buffer = 0
                for column in range(0, 5):
                    if column != 0:
                        left_right_buffer -= 1
                    else:
                        left_right_buffer = 0

                    top_value = (row * left_single_row) + up_down_buffer + 1
                    bottom_value = ((row + 1) * left_single_row) + up_down_buffer - 1
                    left_value = column * left_single_column + left_right_buffer + 2
                    right_value = ((column + 1) * left_single_column) + left_right_buffer - 1

                    bubble_area = new_image_left[top_value: bottom_value, left_value: right_value]

                    bubbles.append(bubble_area)

                    if 0 in new_image_left[top_value: bottom_value, left_value: right_value]:
                        answers[row] = column + 1

                        circle_hor_pos = left_value + ((right_value - left_value) // 2)
                        circle_ver_pos = top_value + ((bottom_value - top_value) // 2)

                        if str(answers[row]) == answer_key[row]:
                            circle_color = (255, 0, 0)
                        else:
                            circle_color = (0, 0, 255)

                        cv.circle(rgb_left, (circle_hor_pos, circle_ver_pos),
                                  5,
                                  circle_color,
                                  -1)

                question_counter += 1

        else:
            for row in range(0, 10):
                if row != 0:
                    up_down_buffer += 2
                else:
                    up_down_buffer = 0
                for column in range(0, 5):
                    if column != 0:
                        left_right_buffer -= 3
                    else:
                        left_right_buffer = 0
                    top_value = (row * right_single_row) + up_down_buffer + 2
                    bottom_value = ((row + 1) * right_single_row) + up_down_buffer - 2
                    left_value = (column * right_single_column) + left_right_buffer + 2
                    right_value = ((column + 1) * right_single_column) + left_right_buffer - 2

                    bubble_area = new_image_right[top_value: bottom_value, left_value: right_value]

                    bubbles.append(bubble_area)

                    if 0 in new_image_right[top_value: bottom_value, left_value: right_value]:
                        answers[row + 10] = column + 1

                        circle_hor_pos = left_value + ((right_value - left_value) // 2)
                        circle_ver_pos = top_value + ((bottom_value - top_value) // 2)

                        if str(answers[row+10]) == answer_key[row+10]:
                            circle_color = (255, 0, 0)
                        else:
                            circle_color = (0, 0, 255)

                        cv.circle(rgb_right, (circle_hor_pos, circle_ver_pos),
                                  5,
                                  circle_color,
                                  -1)

                question_counter += 1

    # for index, bubble in enumerate(bubbles):
    # if index > 48:
    # cv.imshow(f'{index+1}', bubble)

    assigned_bubbles = np.concatenate((rgb_left, rgb_right), 1)
    cv.imshow('Answer sheet', assigned_bubbles)

    good_read = ''
    cv.waitKey(1)

    def ask_good_read(read):
        nonlocal good_read
        good_read = read
        root.destroy()

    root = Tk()
    root.title('Is this a good read?')
    root.geometry('+400+200')
    yes_button = tkinter.Button(text='Yes', width=10, height=3, command=lambda: ask_good_read('yes'))
    no_button = tkinter.Button(text='No', width=10, height=3, command=lambda: ask_good_read('no'))

    yes_button.grid(row=0, column=0)
    no_button.grid(row=0, column=1)

    root.mainloop()

    if good_read == 'yes':
        return answers
    else:
        cv.destroyAllWindows()
        new_found = get_video_feed()
        return get_assigned_bubbles(new_found[1], answer_key)


def grade_test(answers, answer_key, test_type):
    grade = 0
    for index, key in enumerate(answer_key):
        if str(answers[index]) == key:
            grade += 1

    grade_percent = (grade / len(answer_key)) * 100

    if test_type == 'of1':
        final_grade = grade_percent * 10
    elif test_type == 'of2':
        final_grade = grade_percent * 40
    elif test_type == 'ex':
        final_grade = grade_percent * 50
    else:
        final_grade = grade_percent

    display_grade = Tk()
    display_grade.geometry('+400+200')

    def ok():
        display_grade.destroy()

    score_label = tkinter.Label(height=5, text=f'{grade} questões corretas.')
    grade_label = tkinter.Label(width=10, height=3, text=f'{int(final_grade)}')
    ok_button = tkinter.Button(width=10, height=3, text='Ok', command=ok)
    score_label.pack()
    grade_label.pack()
    ok_button.pack()

    display_grade.mainloop()

    return grade_percent


while True:
    try:
        name, subject, test, key = get_answer_key()

        found = get_video_feed()

        response = get_assigned_bubbles(found[1], key)
        grade_test(response, key, test)

    except TypeError as err:
        print(err)
        print('Image not loaded')
        break

    if cv.waitKey(5) & 0xFF == ord(' '):
        break
