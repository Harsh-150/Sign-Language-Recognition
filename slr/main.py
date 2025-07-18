print("INFO: Initializing System")
import copy
import csv
import os
import datetime
import textwrap
import time
import threading

import pyautogui
import cv2 as cv
import mediapipe as mp
from dotenv import load_dotenv

from slr.model.classifier import KeyPointClassifier

from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks

from slr.utils.draw_debug import get_result_image
from slr.utils.draw_debug import get_fps_log_image
from slr.utils.draw_debug import draw_bounding_rect
from slr.utils.draw_debug import draw_hand_label
from slr.utils.draw_debug import show_fps_log
from slr.utils.draw_debug import show_result

from slr.utils.pre_process import calc_bounding_rect
from slr.utils.pre_process import calc_landmark_list
from slr.utils.pre_process import pre_process_landmark

from slr.utils.logging import log_keypoints
from slr.utils.logging import get_dict_form_list
from slr.utils.logging import get_mode

# Import the TTS function from tts.py
from .tts import speak_word

# Import the speech-to-text recognition function
from .speech_to_text import recognize_speech

VIDEO_X, VIDEO_Y = 10, 10
VIDEO_W, VIDEO_H = 400, 300
TEXT_X = VIDEO_X
TEXT_Y_START = VIDEO_Y + VIDEO_H + 20

BUTTON_Y = TEXT_Y_START + 120
BUTTON_W, BUTTON_H = 110, 50  # Slightly reduced width to fit all buttons
BUTTON_GAP = 15  # Reduced gap to fit all buttons

# Button definitions - make sure these are all defined
button_word = (TEXT_X, BUTTON_Y, TEXT_X + BUTTON_W, BUTTON_Y + BUTTON_H)
button_sentence = (TEXT_X + BUTTON_W + BUTTON_GAP, BUTTON_Y, TEXT_X + 2*BUTTON_W + BUTTON_GAP, BUTTON_Y + BUTTON_H)
button_delete_char = (TEXT_X + 2*(BUTTON_W + BUTTON_GAP), BUTTON_Y, TEXT_X + 3*BUTTON_W + 2*BUTTON_GAP, BUTTON_Y + BUTTON_H)
button_delete_word = (TEXT_X + 3*(BUTTON_W + BUTTON_GAP), BUTTON_Y, TEXT_X + 4*BUTTON_W + 3*BUTTON_GAP, BUTTON_Y + BUTTON_H)
button_speak_sentence = (TEXT_X + 4*(BUTTON_W + BUTTON_GAP), BUTTON_Y, TEXT_X + 5*BUTTON_W + 4*BUTTON_GAP, BUTTON_Y + BUTTON_H)
button_clear = (TEXT_X + 5*(BUTTON_W + BUTTON_GAP), BUTTON_Y, TEXT_X + 6*BUTTON_W + 5*BUTTON_GAP, BUTTON_Y + BUTTON_H)
button_speech = (TEXT_X, BUTTON_Y + BUTTON_H + BUTTON_GAP, TEXT_X + BUTTON_W, BUTTON_Y + 2*BUTTON_H + BUTTON_GAP)

clicked_button = [None]

# ---- Speech image display state ----
IMAGE_SIZE = (180, 180)  # Larger images
IMAGE_X = button_clear[2] + 20
IMAGE_Y = BUTTON_Y

speech_images_queue = []
speech_image_start_time = None

# --- For non-blocking speech recognition ---
speech_thread = None
speech_text_result = [None]  # Mutable container for thread result

# --- To display recognized speech text ---
speech_display_text = ""

def speak_sentence(sentence):
    """Speak the complete sentence"""
    if sentence:
        speak_word(sentence)

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if button_word[0] <= x <= button_word[2] and button_word[1] <= y <= button_word[3]:
            clicked_button[0] = "word"
        elif button_sentence[0] <= x <= button_sentence[2] and button_sentence[1] <= y <= button_sentence[3]:
            clicked_button[0] = "sentence"
        elif button_delete_char[0] <= x <= button_delete_char[2] and button_delete_char[1] <= y <= button_delete_char[3]:
            clicked_button[0] = "delete_char"
        elif button_delete_word[0] <= x <= button_delete_word[2] and button_delete_word[1] <= y <= button_delete_word[3]:
            clicked_button[0] = "delete_word"
        elif button_speak_sentence[0] <= x <= button_speak_sentence[2] and button_speak_sentence[1] <= y <= button_speak_sentence[3]:
            clicked_button[0] = "speak_sentence"
        elif button_clear[0] <= x <= button_clear[2] and button_clear[1] <= y <= button_clear[3]:
            clicked_button[0] = "clear"
        elif button_speech[0] <= x <= button_speech[2] and button_speech[1] <= y <= button_speech[3]:
            clicked_button[0] = "speech"

def prepare_speech_images(text, images_folder):
    queue = []
    words = text.split()
    for i, word in enumerate(words):
        for c in word:
            if c.isalpha():
                img_path = os.path.join(images_folder, f"{c.upper()}.png")
                if os.path.exists(img_path):
                    img = cv.imread(img_path)
                    if img is not None:
                        img_small = cv.resize(img, IMAGE_SIZE)
                        queue.append((img_small, 1000))  # 1 second per letter
        if i < len(words) - 1:
            queue.append((None, 2000))  # 2 seconds pause between words
    return queue

def display_speech_image(display_image):
    global speech_images_queue, speech_image_start_time
    if not speech_images_queue:
        return

    current_time = int(time.time() * 1000)
    if speech_image_start_time is None:
        speech_image_start_time = current_time

    img, duration = speech_images_queue[0]
    elapsed = current_time - speech_image_start_time
    if elapsed > duration:
        speech_images_queue.pop(0)
        speech_image_start_time = current_time
        if not speech_images_queue:
            return
        img, duration = speech_images_queue[0]

    if img is not None:
        h, w = img.shape[:2]
        max_h, max_w = display_image.shape[:2]
        y1, y2 = IMAGE_Y, IMAGE_Y + h
        x1, x2 = IMAGE_X, IMAGE_X + w
        if y2 <= max_h and x2 <= max_w:
            display_image[y1:y2, x1:x2] = img

def speech_recognition_thread():
    text = recognize_speech()
    speech_text_result[0] = text

def main():
    global speech_images_queue, speech_image_start_time, speech_thread, speech_text_result, speech_display_text
    gesture_buffer = []
    sentence_buffer = []
    last_gesture = None
    frames_since_last_gesture = 0
    hold_count = 0
    hold_threshold = 30

    load_dotenv()
    args = get_args()

    keypoint_file = "slr/model/keypoint.csv"
    counter_obj = get_dict_form_list(keypoint_file)

    CAP_DEVICE = args.device
    CAP_WIDTH = args.width
    CAP_HEIGHT = args.height

    USE_STATIC_IMAGE_MODE = True
    MAX_NUM_HANDS = args.max_num_hands
    MIN_DETECTION_CONFIDENCE = args.min_detection_confidence
    MIN_TRACKING_CONFIDENCE = args.min_tracking_confidence

    USE_BRECT = args.use_brect
    MODE = args.mode
    DEBUG = int(os.environ.get("DEBUG", "0")) == 1
    CAP_DEVICE = 0

    print("INFO: System initialization Successful")
    print("INFO: Opening Camera")

    cap = cv.VideoCapture(CAP_DEVICE)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    background_image = cv.imread("resources/background.png")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=USE_STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )

    keypoint_classifier = KeyPointClassifier()

    keypoint_labels_file = "slr/model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]

    cv_fps = CvFpsCalc(buffer_len=10)
    print("INFO: System is up & running")

    window_name = "Sign Language Recognition"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.setMouseCallback(window_name, mouse_callback)

    while True:
        fps = cv_fps.get()
        key = cv.waitKey(1)
        if key == 27:
            print("INFO: Exiting...")
            break
        elif key == 57:
            name = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(f'ss/{name}.png')

        success, image = cap.read()
        if not success:
            continue

        image = cv.resize(image, (VIDEO_W, VIDEO_H))
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()

        debug_image = cv.flip(debug_image, 1)
        image_rgb = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        if DEBUG:
            MODE = get_mode(key, MODE)
            fps_log_image = show_fps_log(fps_log_image, fps)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                use_brect = True
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                if MODE == 0:
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 25:
                        hand_sign_text = ""
                    else:
                        hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                    result_image = show_result(result_image, handedness, hand_sign_text)
                    if hand_sign_text != "" and hand_sign_text == last_gesture:
                        hold_count += 1
                    else:
                        hold_count = 0
                        last_gesture = hand_sign_text
                    if hold_count == hold_threshold:
                        gesture_buffer.append(hand_sign_text)
                        hold_count = 0
                        frames_since_last_gesture = 0
                    elif hand_sign_text == "":
                        frames_since_last_gesture += 1
                elif MODE == 1:
                    log_keypoints(key, pre_processed_landmark_list, counter_obj, data_limit=1000)
                debug_image = draw_bounding_rect(debug_image, use_brect, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)

        display_image = background_image.copy()
        display_image[VIDEO_Y:VIDEO_Y+VIDEO_H, VIDEO_X:VIDEO_X+VIDEO_W] = debug_image

        # Display current gesture buffer
        cv.putText(
            display_image,
            'Current: ' + ''.join(gesture_buffer),
            (TEXT_X, TEXT_Y_START),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv.LINE_AA
        )

        # Display last added word
        if sentence_buffer:
            last_word = sentence_buffer[-1]
        else:
            last_word = ""
        cv.putText(
            display_image,
            'Last Word: ' + last_word,
            (TEXT_X, TEXT_Y_START + 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
            cv.LINE_AA
        )

        # Display complete sentence with word wrap
        sequence_text = 'Sentence: ' + ' '.join(sentence_buffer)
        wrapped_lines = textwrap.wrap(sequence_text, width=40)
        for i, line in enumerate(wrapped_lines):
            y = TEXT_Y_START + 60 + i * 30
            cv.putText(
                display_image,
                line,
                (TEXT_X, y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 0),
                1,
                cv.LINE_AA
            )

        # Draw buttons - make sure to use button_delete_word instead of button_delete
        cv.rectangle(display_image, (button_word[0], button_word[1]), (button_word[2], button_word[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_sentence[0], button_sentence[1]), (button_sentence[2], button_sentence[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_delete_char[0], button_delete_char[1]), (button_delete_char[2], button_delete_char[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_delete_word[0], button_delete_word[1]), (button_delete_word[2], button_delete_word[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_speak_sentence[0], button_speak_sentence[1]), (button_speak_sentence[2], button_speak_sentence[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_clear[0], button_clear[1]), (button_clear[2], button_clear[3]), (200,200,200), -1)
        cv.rectangle(display_image, (button_speech[0], button_speech[1]), (button_speech[2], button_speech[3]), (150, 200, 250), -1)

        # Button labels
        cv.putText(display_image, "Speak", (button_word[0]+25, button_word[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv.putText(display_image, "Word", (button_word[0]+25, button_word[1]+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv.putText(display_image, "Add to", (button_sentence[0]+25, button_sentence[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv.putText(display_image, "Sentence", (button_sentence[0]+15, button_sentence[1]+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv.putText(display_image, "Delete", (button_delete_char[0]+15, button_delete_char[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv.putText(display_image, "Character", (button_delete_char[0]+5, button_delete_char[1]+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv.putText(display_image, "Delete", (button_delete_word[0]+15, button_delete_word[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv.putText(display_image, "Word", (button_delete_word[0]+25, button_delete_word[1]+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv.putText(display_image, "Speak", (button_speak_sentence[0]+25, button_speak_sentence[1]+20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv.putText(display_image, "Sentence", (button_speak_sentence[0]+15, button_speak_sentence[1]+40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv.putText(display_image, "Clear", (button_clear[0]+18, button_clear[1]+28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv.putText(display_image, "Speech", (button_speech[0]+10, button_speech[1]+28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
        # Display recognized speech text
        text_x = button_speech[2] + 10
        text_y = button_speech[1] + 30
        if speech_display_text:
            cv.putText(display_image, f"Text: {speech_display_text}", (text_x, text_y),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        # Display speech images if active
        display_speech_image(display_image)

        # Handle speech recognition thread
        if clicked_button[0] == "speech":
            if speech_thread is None or not speech_thread.is_alive():
                speech_text_result[0] = None
                speech_thread = threading.Thread(target=speech_recognition_thread)
                speech_thread.start()
            clicked_button[0] = None

        # Process speech recognition result
        if speech_text_result[0] is not None:
            if speech_text_result[0]:
                speech_display_text = speech_text_result[0]
                speech_images_queue = prepare_speech_images(speech_text_result[0], "alphabet_images")
                speech_image_start_time = None
            else:
                speech_display_text = ""
            speech_text_result[0] = None

            # Handle button clicks
        if clicked_button[0] == "word":
            if gesture_buffer:
                word = ''.join(gesture_buffer)
                speak_word(word)
                gesture_buffer.clear()
                last_gesture = None
                frames_since_last_gesture = 0
            clicked_button[0] = None
            
        elif clicked_button[0] == "sentence":
            if gesture_buffer:
                word = ''.join(gesture_buffer)
                sentence_buffer.append(word)
                gesture_buffer.clear()
                last_gesture = None
                frames_since_last_gesture = 0
            clicked_button[0] = None
            
        elif clicked_button[0] == "delete_char":
            if gesture_buffer:
                gesture_buffer.pop()
                print(f"Deleted last character. Current: {''.join(gesture_buffer)}")
            clicked_button[0] = None
            
        elif clicked_button[0] == "delete_word":
            if sentence_buffer:
                removed_word = sentence_buffer.pop()
                print(f"Removed word: {removed_word}")
            clicked_button[0] = None
            
        elif clicked_button[0] == "speak_sentence":
            if sentence_buffer:
                threading.Thread(target=speak_sentence, args=(' '.join(sentence_buffer),)).start()
            clicked_button[0] = None
            
        elif clicked_button[0] == "clear":
            gesture_buffer.clear()
            sentence_buffer.clear()
            last_gesture = None
            frames_since_last_gesture = 0
            clicked_button[0] = None

        cv.imshow(window_name, display_image)

    cap.release()
    cv.destroyAllWindows()
    print("Camera Off!")

if __name__ == "__main__":
    main()