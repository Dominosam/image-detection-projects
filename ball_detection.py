import cv2
import numpy as np

def detect_ball(image, show_steps=False):
    """
    Detects a red ball in an image.
    :param image: Input image.
    :param show_steps: If True, shows intermediate steps.
    :return: Image with detected ball.
    """
    if image is None:
        raise ValueError("No image provided")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define colors dynamically or use a fixed range
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    if show_steps:
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)

    # Transformations for better detection
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours and get the largest one
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:  # Minimum size to avoid false positives
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(image, center, 5, (0, 0, 255), -1)

    if show_steps or center is None:
        cv2.imshow('Detected Ball', image)
        cv2.waitKey(0)

    return image

def main():
    image_path = "resources/ball.png"
    video_path = 'resources/movingball.mp4'

    # Image Detection
    image = cv2.imread(image_path)
    if image is not None:
        detect_ball(image, True)

    # Video Detection
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            detected_frame = detect_ball(frame, False)
            cv2.imshow('Moving Ball Detection', detected_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()