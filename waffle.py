import cv2
import numpy as np
import os


image_path = "./"
image_extension = ".png"


image_files = [
    filename
    for filename in os.listdir(image_path)
    if filename.endswith(image_extension)
]


for image_file in image_files:
    RGBimage = cv2.imread(os.path.join(image_path, image_file))



    image_gray = cv2.cvtColor(RGBimage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image_gray",image_gray)


 
    _, binary_image = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary_image", binary_image)


    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening", opening)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    contour_image = RGBimage.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contour_image", contour_image)


    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)



    index = 0
    totalcontourArea = 0



    text = f"MAXArea: {area}"
    cv2.putText(contour_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if 200 < contour_area < 99000:
            totalcontourArea=totalcontourArea + contour_area
            index=index+1

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            index_text = f"Contour {index}: Area {contour_area:.2f}"
            cv2.putText(contour_image, index_text, (cX - 70, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    LOSS=totalcontourArea/98000*100

    print(totalcontourArea)
    print(LOSS)

    if 150 > LOSS > 96:
        cv2.putText(contour_image, "PASS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(contour_image, "FAIL", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(contour_image, "totalcontourArea:"+str(int(totalcontourArea)) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(contour_image, "LOSS:"+str(int(totalcontourArea/98000*100)) , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Contour Image with Area", contour_image)
    cv2.waitKey(0)



cv2.destroyAllWindows()