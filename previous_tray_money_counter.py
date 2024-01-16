import cv2
import numpy as np
import os

def check_if_inside(bl, tr, point):
    return bl[0] < point[0] < tr[0] and tr[1] < point[1] < bl[1]

def detect_circles(gray_image):
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 5)
    return cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=60, param1=45, param2=21, minRadius=15, maxRadius=40)

def detect_edges(gray_image):
    return cv2.Canny(gray_image, 33, 155, apertureSize=3)

def get_tray_corners(lines, maxwidth, maxheight):
    x1min, y1max = 0, maxheight
    x2max, y2min = maxwidth, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 > x1min:
            x1min = x1
        if y1 < y1max:
            y1max = y1
        if x2 < x2max:
            x2max = x2
        if y2 > y2min:
            y2min = y2

    return (x1min, y1max), (x2max, y2min)

def add_legend(image):
    cv2.putText(image, 'Inside Tray - Green', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Outside Tray - Red', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, 'Small Coin - Yellow', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(image, 'Big Coin - Blue', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def process_image(path):
    image = cv2.imread(path)
    blur = cv2.GaussianBlur(image, (5, 5), 2)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    circles = detect_circles(gray)
    edges = detect_edges(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250)

    maxheight, maxwidth = image.shape[:2]
    right_top, left_bottom = get_tray_corners(lines, maxwidth, maxheight)
    value_inside = 0
    value_outside = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        big_coin_min_r = sorted(circles, key=lambda x: x[2], reverse=True)[1][2]

        for circle in circles:
            (x, y, r) = circle
            color = (255, 0, 0) if r >= big_coin_min_r else (0, 255, 255)  # Blue for big, yellow for small
            coin_value = 5 if r >= big_coin_min_r else 0.05

            if check_if_inside(left_bottom, right_top, (x, y)):
                cv2.circle(image, (x, y), r, color, 2)  # Draw the outer circle
                cv2.circle(image, (x, y), 2, (0, 255, 0), 3)  # Draw the center in green
                value_inside += coin_value
            else:
                cv2.circle(image, (x, y), r, color, 2)  # Draw the outer circle
                cv2.circle(image, (x, y), 2, (0, 0, 255), 3)  # Draw the center in red
                value_outside += coin_value

    print(f"Value of all coins: {value_inside + value_outside:.2f}")
    print(f"Value of inside coins: {value_inside:.2f}")
    print(f"Value of outside coins: {value_outside:.2f}")

    add_legend(image)  # Add a legend to the image
    cv2.imshow('Image: ', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = "resources/trays/"

    filenames = os.listdir(path)

    for filename in filenames:
        process_image(path + filename)
        print()