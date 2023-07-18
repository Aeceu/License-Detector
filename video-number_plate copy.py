import cv2

harcascade = "model/haarcascade_russian_plate_number.xml"
video_path = "sample1.mp4"  # Specify the path to your video file

cap = cv2.VideoCapture(video_path)
min_area = 500
count = 0
plate_cascade = cv2.CascadeClassifier(harcascade)

# Set the desired output window size
output_width = 800
output_height = 600

while True:
    success, frame = cap.read()
    if not success:
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for x, y, w, h in plates:
        area = w * h

        if (
            area > min_area
        ):  # You can define the min_area threshold based on your requirements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Number Plate",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 0, 255),
                2,
            )

            img_roi = frame[y : y + h, x : x + w]
            cv2.imshow("ROI", img_roi)

    # Resize the frame to the desired output size
    frame = cv2.resize(frame, (output_width, output_height))

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
