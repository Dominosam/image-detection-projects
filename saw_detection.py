import os
import cv2

def detect_chainsaw(train_image, image, filename):
    # Resize training and target images to reduce computation
    small_train_img = cv2.resize(train_image, (0,0), fx=0.3, fy=0.3)
    small_img = cv2.resize(image, (0,0), fx=0.3, fy=0.3)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Find keypoints and descriptors with ORB in both images
    kp1, des1 = orb.detectAndCompute(small_train_img, None)
    kp2, des2 = orb.detectAndCompute(small_img, None)

    # Create a brute force matcher object
    bruteforce_matcher = cv2.BFMatcher()

    # Match descriptors between both images
    matches = bruteforce_matcher.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Set dynamic threshold based on the type of input (image or video)
    threshold = 7 if filename.endswith('.mp4') else 25

    # If good matches exceed threshold, annotate the image with 'CHAINSAW'
    if len(good_matches) >= threshold:
        cv2.putText(small_img, 'CHAINSAW', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    # Draw the good matches on the image
    matches_img = cv2.drawMatchesKnn(small_train_img, kp1, small_img, kp2, good_matches, None, flags=2)

    # For images (not videos), display the result
    if not filename.endswith('.mp4'):
        cv2.imshow(f'matches_{filename}', matches_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # For videos, return the frame with the matches drawn
        return matches_img

if __name__ == '__main__':
    path = "resources/saw/"

    filenames = os.listdir(path)
    train_img = cv2.imread("resources/saw/train/saw1.jpg", 0)


    for filename in filenames:
        path_to_file = path + filename

        if filename.endswith(".mp4"):
            cap = cv2.VideoCapture(path_to_file)

            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                matched_frame = detect_chainsaw(train_img, frame, filename)
                if matched_frame is not None:
                    cv2.imshow('sawmovie.mp4', matched_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        # If the current file is an image
        else:
            img = cv2.imread(path_to_file, 0)
            detect_chainsaw(train_img, img, filename)

        print(f"Processed: {filename}")