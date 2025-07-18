import cv2 as cv

def draw_landmarks(image, landmark_point):
    # Define colors
    white = (255, 255, 255)
    black = (0, 0, 0)

    # Set all skeletal lines to the same color
    skeletal_color = white
    outline_color = black

    if len(landmark_point) > 0:
        # Thumb [3, 4]
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), outline_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), outline_color, 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), skeletal_color, 2)

        # Index finger [6, 7, 8]
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), outline_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), outline_color, 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), outline_color, 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), skeletal_color, 2)

        # Middle finger [10, 11, 12]
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), outline_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), outline_color, 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), outline_color, 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), skeletal_color, 2)

        # Ring finger [14, 15, 16]
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), outline_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), outline_color, 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), outline_color, 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), skeletal_color, 2)

        # Little finger [18, 19, 20]
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), outline_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), outline_color, 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), outline_color, 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), skeletal_color, 2)

        # Palm [1, 5, 9, 13, 17, 0]
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), outline_color, 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), outline_color, 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), outline_color, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), outline_color, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), outline_color, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), outline_color, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), outline_color, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), skeletal_color, 2)

    # Key Points (unchanged)
    for index, landmark in enumerate(landmark_point):
        cv.circle(image, (landmark[0], landmark[1]), 5 if index not in [4, 8, 12, 16, 20] else 8, skeletal_color, -1)
        cv.circle(image, (landmark[0], landmark[1]), 5 if index not in [4, 8, 12, 16, 20] else 8, outline_color, 1)

    return image
