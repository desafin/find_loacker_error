#include "../Common/Common.h"
#include <vector>
#include <filesystem>


int imgindex = 0;

int main()
{
    std::string image_path = "./";  // 사진 파일들이 있는 경로로 변경해야 합니다.
    std::string image_extension = ".png";

    std::vector<std::string> image_files;

    // 사진 파일들의 리스트 생성
    for (const auto& entry : std::filesystem::directory_iterator(image_path))
    {
        if (entry.path().extension() == image_extension)
        {
            image_files.push_back(entry.path().string());
        }
    }

    for (const auto& image_file : image_files)
    {
        cv::Mat RGBimage = cv::imread(image_file);

        cv::Mat image_gray;
        cv::cvtColor(RGBimage, image_gray, cv::COLOR_BGR2GRAY);

        // 이진화를 위한 쓰레시홀드 함수 적용
        cv::Mat binary_image;
        cv::threshold(image_gray, binary_image, 50, 255, cv::THRESH_BINARY);

        // 모폴로지 연산을 사용하여 노이즈 제거
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat opening;
        cv::morphologyEx(binary_image, opening, cv::MORPH_OPEN, kernel);

        // 컨투어 찾기
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(opening, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 컨투어 그리기
        cv::Mat contour_image = RGBimage.clone();
        cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 255, 0), 2);

        // 가장 큰 컨투어의 넓이 계산
        double max_area = 0;
        int max_contour_index = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > max_area)
            {
                max_area = area;
                max_contour_index = i;
            }
        }
        cv::putText(contour_image, "max_contourArea:"+ to_string(max_area), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

        double total_contour_area = 0;
        int index = 0;
        // 컨투어의 인덱스 출력
        for (int i = 0; i < contours.size(); i++)
        {
            
            double contour_area = cv::contourArea(contours[i]);
            if (contour_area > 100 && contour_area < 99000)
            {
                index = index + 1;
                total_contour_area += contour_area;

                // 컨투어 중앙 좌표 계산
                cv::Moments M = cv::moments(contours[i]);
                double cX = 0, cY = 0;
                if (M.m00 != 0)
                {
                    cX = M.m10 / M.m00;
                    cY = M.m01 / M.m00;
                }

                std::string index_text = "Contour " + std::to_string(i) + ": Area " + std::to_string(contour_area);
                cv::putText(contour_image, index_text, cv::Point(cX - 70, cY), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }
        }

        double LOSS = total_contour_area / 98000 * 100;

        std::cout << "Total Contour Area: " << total_contour_area << std::endl;
        std::cout << "LOSS: " << LOSS << std::endl;

        if (LOSS > 97 && LOSS < 150)
        {
            cv::putText(contour_image, "PASS", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }
        else
        {
            cv::putText(contour_image, "FAIL", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        cv::putText(contour_image, "Total Contour Area: " + std::to_string(static_cast<int>(total_contour_area)), cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(contour_image, "LOSS: " + std::to_string(static_cast<int>(total_contour_area / 98000 * 100)), cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        std::string result_filename = std::to_string(imgindex) + ".png";
        //cv::imwrite(result_filename, contour_image);


        cv::imshow("Contour Image with Area", contour_image);
        imgindex++;

        cv::waitKey(0);
    }

    cv::destroyAllWindows();

    return 0;
}
