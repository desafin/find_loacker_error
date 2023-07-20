import cv2
import numpy as np
import os

# 사진 파일들의 경로와 확장자를 지정
image_path = "./"  # 사진 파일들이 있는 경로로 변경해야 합니다.
image_extension = ".png"

# 사진 파일들의 리스트 생성
image_files = [
    filename
    for filename in os.listdir(image_path)
    if filename.endswith(image_extension)
]


for image_file in image_files:
    RGBimage = cv2.imread(os.path.join(image_path, image_file))



    image_gray = cv2.cvtColor(RGBimage, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image_gray",image_gray)


    # 이진화를 위한 쓰레시홀드 함수 적용
    _, binary_image = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary_image", binary_image)

    # 모폴로지 연산을 사용하여 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening", opening)

    # 컨투어 찾기
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 컨투어 그리기
    contour_image = RGBimage.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contour_image", contour_image)

    # 가장 큰 컨투어의 넓이 계산
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)



    index = 0
    totalcontourArea = 0


    # 이미지에 넓이 출력
    text = f"MAXArea: {area}"
    cv2.putText(contour_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # 컨투어의 인덱스 출력
    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if 200 < contour_area < 99000:
            totalcontourArea=totalcontourArea + contour_area
            index=index+1
            # 컨투어 중앙 좌표 계산
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
    #손실률 계산

    cv2.putText(contour_image, "totalcontourArea:"+str(int(totalcontourArea)) , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(contour_image, "LOSS:"+str(int(totalcontourArea/98000*100)) , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow("Contour Image with Area", contour_image)
    cv2.waitKey(0)



cv2.destroyAllWindows()