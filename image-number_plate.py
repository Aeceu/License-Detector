import cv2

harcascade = "model/haarcascade_russian_plate_number.xml"
image_path = "sample.jpg"  # Specify the path to your image file

img = cv2.imread(image_path)
min_area = 500
plate_cascade = cv2.CascadeClassifier(harcascade)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

for x, y, w, h in plates:
    area = w * h

    if (
        area > min_area
    ):  # You can define the min_area threshold based on your requirements
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            "Number Plate",
            (x, y - 5),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 0, 255),
            2,
        )

        img_roi = img[y : y + h, x : x + w]
        cv2.imshow("ROI", img_roi)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
