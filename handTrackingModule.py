import cv2
import mediapipe as mp
import math
import pyautogui
import time
import numpy as np


# =========================================================
# HAND DETECTOR
# =========================================================

class HandDetector:

    def __init__(
        self,
        mode=False,
        maxHands=1,
        model_complexity=1,
        detectionCon=0.7,
        trackCon=0.7
    ):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            model_complexity=model_complexity,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

        self.lmList = []
        self.results = None

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                if draw:

                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS
                    )

        return img

    def findPosition(self, img, handNo=0):

        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:

            if handNo < len(
                self.results.multi_hand_landmarks
            ):

                myHand = self.results.multi_hand_landmarks[
                    handNo
                ]

                h, w, c = img.shape

                for id, lm in enumerate(
                    myHand.landmark
                ):

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    self.lmList.append(
                        [id, cx, cy]
                    )

        return self.lmList

    def fingersUp(self):

        fingers = []

        if len(self.lmList) == 0:
            return fingers

        # Thumb
        if (
            self.lmList[4][1]
            <
            self.lmList[3][1]
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):

            if (
                self.lmList[self.tipIds[id]][2]
                <
                self.lmList[self.tipIds[id] - 2][2]
            ):
                fingers.append(1)

            else:
                fingers.append(0)

        return fingers


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector()

    # -----------------------------------------------------
    # SCREEN
    # -----------------------------------------------------

    screenWidth, screenHeight = pyautogui.size()

    # -----------------------------------------------------
    # MOUSE SETTINGS
    # -----------------------------------------------------

    prevMouseX = 0
    prevMouseY = 0

    smoothening = 5

    lastClickTime = 0

    # -----------------------------------------------------
    # DRAWING CANVAS
    # -----------------------------------------------------

    canvas = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8
    )

    # Store previous point
    prevX = 0
    prevY = 0

    # -----------------------------------------------------
    # DRAWING SETTINGS
    # -----------------------------------------------------

    penColor = (255, 0, 255)

    penSize = 6

    eraserSize = 40

    # History for undo
    history = []

    # -----------------------------------------------------
    # MODES
    # -----------------------------------------------------

    mode = "MOUSE"

    # -----------------------------------------------------
    # COLORS
    # -----------------------------------------------------

    colors = {

        "PINK": (255, 0, 255),

        "BLUE": (255, 0, 0),

        "GREEN": (0, 255, 0),

        "RED": (0, 0, 255)
    }

    colorName = "PINK"

    # -----------------------------------------------------
    # FPS
    # -----------------------------------------------------

    previousTime = 0

    print()
    print("======================================")
    print("     AIR CANVAS + VIRTUAL MOUSE")
    print("======================================")
    print()
    print("M = Mouse mode")
    print("W = Writing mode")
    print("E = Eraser")
    print("C = Clear")
    print("U = Undo")
    print("S = Save drawing")
    print()
    print("1 = Pink")
    print("2 = Blue")
    print("3 = Green")
    print("4 = Red")
    print()
    print("Q = Quit")
    print("======================================")

    # =====================================================
    # LOOP
    # =====================================================

    while True:

        success, img = cap.read()

        if not success:
            print("Camera error")
            break

        # Mirror camera
        img = cv2.flip(img, 1)

        # Resize
        img = cv2.resize(
            img,
            (1280, 720)
        )

        # -------------------------------------------------
        # HAND DETECTION
        # -------------------------------------------------

        img = detector.findHands(
            img,
            draw=True
        )

        lmList = detector.findPosition(img)

        # =================================================
        # HAND DETECTED
        # =================================================

        if len(lmList) != 0:

            fingers = detector.fingersUp()

            # Index finger
            indexX = lmList[8][1]
            indexY = lmList[8][2]

            # Thumb
            thumbX = lmList[4][1]
            thumbY = lmList[4][2]

            # Middle finger
            middleX = lmList[12][1]
            middleY = lmList[12][2]

            # Ring finger
            ringX = lmList[16][1]
            ringY = lmList[16][2]

            # Little finger
            pinkyX = lmList[20][1]
            pinkyY = lmList[20][2]

            # -------------------------------------------------
            # THUMB + INDEX DISTANCE
            # -------------------------------------------------

            pinchDistance = math.hypot(
                thumbX - indexX,
                thumbY - indexY
            )

            # =================================================
            # MOUSE MODE
            # =================================================

            if mode == "MOUSE":

                # Index finger only
                if fingers == [0, 1, 0, 0, 0]:

                    # Map camera to screen

                    mouseX = np.interp(
                        indexX,
                        [100, 1180],
                        [0, screenWidth]
                    )

                    mouseY = np.interp(
                        indexY,
                        [100, 620],
                        [0, screenHeight]
                    )

                    # Smooth
                    currX = (
                        prevMouseX
                        +
                        (mouseX - prevMouseX)
                        / smoothening
                    )

                    currY = (
                        prevMouseY
                        +
                        (mouseY - prevMouseY)
                        / smoothening
                    )

                    pyautogui.moveTo(
                        int(currX),
                        int(currY)
                    )

                    prevMouseX = currX
                    prevMouseY = currY

                    # Green tracking circle

                    cv2.circle(
                        img,
                        (indexX, indexY),
                        15,
                        (0, 255, 0),
                        cv2.FILLED
                    )

                # -----------------------------------------
                # LEFT CLICK
                # -----------------------------------------

                if (
                    fingers[1] == 1
                    and
                    pinchDistance < 35
                ):

                    currentTime = time.time()

                    if (
                        currentTime
                        -
                        lastClickTime
                        >
                        0.6
                    ):

                        pyautogui.click()

                        lastClickTime = currentTime

                        cv2.circle(
                            img,
                            (indexX, indexY),
                            25,
                            (0, 0, 255),
                            cv2.FILLED
                        )

                # -----------------------------------------
                # RIGHT CLICK
                # -----------------------------------------

                if (
                    fingers[1] == 1
                    and
                    fingers[2] == 1
                    and
                    fingers[3] == 0
                    and
                    fingers[4] == 0
                ):

                    currentTime = time.time()

                    if (
                        currentTime
                        -
                        lastClickTime
                        >
                        0.8
                    ):

                        pyautogui.rightClick()

                        lastClickTime = currentTime

                        cv2.circle(
                            img,
                            (indexX, indexY),
                            25,
                            (255, 0, 0),
                            cv2.FILLED
                        )

            # =================================================
            # WRITING MODE
            # =================================================

            elif mode == "WRITE":

                # ---------------------------------------------
                # INDEX ONLY = DRAW
                # ---------------------------------------------

                if fingers == [0, 1, 0, 0, 0]:

                    cv2.circle(
                        img,
                        (indexX, indexY),
                        12,
                        penColor,
                        cv2.FILLED
                    )

                    if (
                        prevX != 0
                        and
                        prevY != 0
                    ):

                        cv2.line(
                            canvas,
                            (prevX, prevY),
                            (indexX, indexY),
                            penColor,
                            penSize
                        )

                    prevX = indexX
                    prevY = indexY

                else:

                    prevX = 0
                    prevY = 0

            # =================================================
            # ERASER MODE
            # =================================================

            elif mode == "ERASER":

                if fingers == [0, 1, 0, 0, 0]:

                    cv2.circle(
                        img,
                        (indexX, indexY),
                        eraserSize,
                        (255, 255, 255),
                        2
                    )

                    cv2.circle(
                        canvas,
                        (indexX, indexY),
                        eraserSize,
                        (0, 0, 0),
                        cv2.FILLED
                    )

                    prevX = 0
                    prevY = 0

        else:

            prevX = 0
            prevY = 0

        # =================================================
        # COMBINE CAMERA + CANVAS
        # =================================================

        grayCanvas = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2GRAY
        )

        _, mask = cv2.threshold(
            grayCanvas,
            20,
            255,
            cv2.THRESH_BINARY
        )

        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR
        )

        # Remove canvas area from image
        imgWithoutCanvas = cv2.bitwise_and(
            img,
            cv2.bitwise_not(mask)
        )

        # Add drawing
        img = cv2.bitwise_or(
            imgWithoutCanvas,
            canvas
        )

        # =================================================
        # TOP UI
        # =================================================

        cv2.rectangle(
            img,
            (0, 0),
            (1280, 80),
            (20, 20, 20),
            cv2.FILLED
        )

        # Mode
        cv2.putText(
            img,
            "MODE: " + mode,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Color
        cv2.putText(
            img,
            "COLOR: " + colorName,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            penColor,
            2
        )

        # FPS
        currentTime = time.time()

        fps = 1 / (
            currentTime
            -
            previousTime
            +
            0.0001
        )

        previousTime = currentTime

        cv2.putText(
            img,
            "FPS: " + str(int(fps)),
            (1100, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # =================================================
        # BOTTOM UI
        # =================================================

        cv2.rectangle(
            img,
            (0, 660),
            (1280, 720),
            (20, 20, 20),
            cv2.FILLED
        )

        cv2.putText(
            img,
            "M Mouse | W Write | E Eraser | C Clear | U Undo | S Save | Q Quit",
            (20, 695),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # =================================================
        # SHOW
        # =================================================

        cv2.imshow(
            "AIR CANVAS + VIRTUAL MOUSE",
            img
        )

        # =================================================
        # KEYBOARD
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):

            break

        # Mouse
        elif key == ord("m"):

            mode = "MOUSE"

            prevX = 0
            prevY = 0

        # Writing
        elif key == ord("w"):

            mode = "WRITE"

            prevX = 0
            prevY = 0

        # Eraser
        elif key == ord("e"):

            mode = "ERASER"

            prevX = 0
            prevY = 0

        # Clear
        elif key == ord("c"):

            canvas = np.zeros(
                (720, 1280, 3),
                dtype=np.uint8
            )

            history.clear()

            print("Canvas cleared")

        # Undo
        elif key == ord("u"):

            if len(history) > 0:

                canvas = history.pop()

                print("Undo")

        # Save
        elif key == ord("s"):

            filename = (
                "air_drawing_"
                +
                str(int(time.time()))
                +
                ".png"
            )

            cv2.imwrite(
                filename,
                canvas
            )

            print(
                "Drawing saved as:",
                filename
            )

        # -------------------------------------------------
        # COLORS
        # -------------------------------------------------

        elif key == ord("1"):

            penColor = colors["PINK"]
            colorName = "PINK"

        elif key == ord("2"):

            penColor = colors["BLUE"]
            colorName = "BLUE"

        elif key == ord("3"):

            penColor = colors["GREEN"]
            colorName = "GREEN"

        elif key == ord("4"):

            penColor = colors["RED"]
            colorName = "RED"

    # =====================================================
    # CLOSE
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()