import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
import os

path = './PA1_Dataset/05/'
save_path = './PA1_test/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
image = glob.glob(path + '*')

BLUE, GREEN, RED, BLACK, WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255)
DRAW_BG = {'color': BLACK, 'val': 0}  # BackGround
DRAW_FG = {'color': WHITE, 'val': 1}  # ForeGround

rect = (0, 0, 1, 1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
value = DRAW_FG
thickness = 3


def onMouse(event, x, y, flags, param):
    global ix, iy, img1, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle:
            img1 = img2.copy()
            cv2.rectangle(img1, (ix, iy), (x, y), RED, 2)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True

        cv2.rectangle(img1, (ix, iy), (x, y), RED, 2)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        print('n: apply')

    if event == cv2.EVENT_LBUTTONDOWN:
        if not rect_over:
            print('select foreground pushing your left button of mouse')
        else:
            drawing = True
            cv2.circle(img1, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img1, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            cv2.circle(img1, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    return


def graph_cut():
    global ix, iy, img1, img2, drawing, value, mask, rectangle
    global rect, rect_or_mask, rect_over

    img1 = cv2.imread(image[0])
    img2 = img1.copy()

    mask = np.zeros(img1.shape[: 2], dtype=np.uint8)
    output = np.zeros(img1.shape, dtype=np.uint8)

    cv2.namedWindow('input')
    cv2.namedWindow('output')
    cv2.setMouseCallback('input', onMouse, (img1, img2))
    cv2.moveWindow('input', img1.shape[1] + 10, 90)
    print('After select region pushing your right button of mouse, press n')

    while True:
        cv2.imshow('output', output)
        cv2.imshow('input', img1)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            break

        if k == ord('0'):
            print('After select eliminating region you want pushing your left button of mouse, press n')
            value = DRAW_BG
        elif k == ord('1'):
            print('After select restoring region you want pushing your left button of mouse, press n')
            value = DRAW_FG
        elif k == ord('r'):
            print('reset')
            rect = (0, 0, 1, 1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            img1 = img2.copy()
            mask = np.zeros(img1.shape[: 2], dtype=np.uint8)
            output = np.zeros(img1.shape, np.uint8)
            print('0: select elimination background', '1: select restoration foreground', 'n: apply', 'r: reset')
        elif k == ord('n'):
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            if rect_or_mask == 0:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:
                cv2.grabCut(img2, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

            print('0: select elimination background', '1: select restoration foreground', 'n: apply', 'r: reset')

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img2, img2, mask=mask2)
    cv2.destroyAllWindows()


graph_cut()