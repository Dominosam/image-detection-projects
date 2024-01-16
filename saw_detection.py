import cv2
import numpy as np
import os

def detect_chainsaw(train_image, image, filename):
    # Resize images for efficiency
    small_train_img = cv2.resize(train_image, (0,0), fx=0.3, fy=0.3)
    small_img = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

    # ORB feature detector and brute force matcher
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(small_train_img, None)
    kp2, des2 = orb.detectAndCompute(small_img, None)

    bruteforce_matcher = cv2.BFMatcher()
    matches = bruteforce_matcher.knnMatch(des1, des2, k=2)

    # Filter good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Dynamic thresholding based on input type
    threshold = 7 if filename.endswith('.mp4') else 25

    # Annotate if chainsaw is detected
    if len(good_matches) >= threshold:
        cv2.putText(small_img, 'CHAINSAW', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    # Draw matches
    matches_img = cv2.drawMatchesKnn(small_train_img, kp1, small_img, kp2, good_matches, None, flags=2)

    # Display result for images
    if not filename.endswith('.mp4'):
        cv2.imshow(f'matches_{filename}', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return matches_img

if __name__ == '__main__':
    # Set the path to the directory containing the images and videos
    path = "resources/saw/"

    # List all files in the specified directory
    filenames = os.listdir(path)

    # Read the training image used for chainsaw detection
    train_img = cv2.imread("resources/saw/train/saw1.jpg", 0)

    # Iterate over each file in the directory
    for filename in filenames:
        # Construct the full path to the file
        path_to_file = path + filename

        # Check if the current file is a video
        if filename.endswith(".mp4"):
            # Open the video file
            cap = cv2.VideoCapture(path_to_file)

            # Process each frame of the video
            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                # Detect chainsaw in the current frame
                matched_frame = detect_chainsaw(train_img, frame, filename)
                if matched_frame is not None:
                    cv2.imshow('sawmovie.mp4', matched_frame)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture object and close all OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

        # If the current file is an image
        else:
            # Read the image
            img = cv2.imread(path_to_file, 0)

            # Detect chainsaw in the image
            detect_chainsaw(train_img, img, filename)

        # Print a message indicating the file has been processed
        print(f"Processed: {filename}")