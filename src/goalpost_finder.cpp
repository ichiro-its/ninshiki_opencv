// Copyright (c) 2021 ICHIRO ITS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <ninshiki_opencv/goalpost_finder.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define POLY_A 1.3173583158869139e+003
#define POLY_B -9.1141374926780969e+000
#define POLY_C 2.0379921266632382e-002

namespace ninshiki_opencv
{
GoalpostFinder::GoalpostFinder()
{
  draw_output = false;
  left_goal_found = false;
  right_goal_found = false;
}

double GoalpostFinder::find_area_roi(cv::Mat fr, cv::Point stR, cv::Point spR)
{
  cv::Point start_roi, stop_roi;
  if (stR.x < spR.x) {
    start_roi.x = stR.x;
    stop_roi.x = spR.x;
  } else {
    start_roi.x = spR.x;
    stop_roi.x = stR.x;
  }

  if (stR.y < spR.y) {
    start_roi.y = stR.y;
    stop_roi.y = spR.y;
  } else {
    start_roi.y = spR.y;
    stop_roi.y = stR.y;
  }

  start_roi.x = start_roi.x < 0 ? 0 : start_roi.x;
  start_roi.y = start_roi.y < 0 ? 0 : start_roi.y;
  start_roi.x = start_roi.x >= fr.cols ? fr.cols - 1 : start_roi.x;
  start_roi.y = start_roi.y >= fr.rows ? fr.rows - 1 : start_roi.y;

  stop_roi.x = stop_roi.x < 0 ? 0 : stop_roi.x;
  stop_roi.y = stop_roi.y < 0 ? 0 : stop_roi.y;
  stop_roi.x = stop_roi.x >= fr.cols ? fr.cols - 1 : stop_roi.x;
  stop_roi.y = stop_roi.y >= fr.rows ? fr.rows - 1 : stop_roi.y;
  if (stop_roi.x - start_roi.x < 0 || stop_roi.y - start_roi.y < 0) {return 0;}
  cv::Mat buff(fr, cv::Rect(
      start_roi.x, start_roi.y, stop_roi.x - start_roi.x,
      stop_roi.y - start_roi.y));
  double totalArea = buff.cols * buff.rows;
  
  // cv::cvtColor(buff, buff, cv::COLOR_BGR2GRAY);
  double count = countNonZero(buff);
  buff.release();
  return count * 100 / totalArea;
}

std::vector<cv::Point> GoalpostFinder::detect_goal(std::shared_ptr<SensorMeasurements> sensor)
{
  std::vector<cv::Point> coordinate(2);
  if (sensor.get()->cameras_size() > 0) {
    
  auto camera = sensor.get()->cameras(0);
    cv::Mat sensor_image(static_cast<int>(camera.height()),
      static_cast<int>(camera.width()), CV_8UC3, std::string(
        camera.image()).data());
    
    camera_height = camera.height();
    camera_width = camera.width();
    image.create(camera_height, camera_width, CV_8UC3);
    output_buffer.create(camera_height, camera_width, CV_8UC3);
    cv::Point empty_point(-1, -1);
    right_goal_coor = empty_point;
    left_goal_coor = empty_point;
    right_goal_height = 0;
    left_goal_height = 0;
    cv::Mat rgb(camera_height, camera_width, CV_8UC3);

    left_goal_found = false;
    right_goal_found = false;

    left_goal_distance = 0.0;
    right_goal_distance = 0.0;
    rgb = sensor_image.clone();

    image = rgb.clone();
    process_image(sensor_image);
    rgb.release();

    output_buffer = image.clone();

    coordinate[0] = left_goal_coor;
    coordinate[1] = right_goal_coor;
    
    if (left_goal_coor.x > 0 && left_goal_coor.y > 0) {
      left_goal_found = true;
    }
    if (right_goal_coor.x > 0 && right_goal_coor.y > 0) {
      right_goal_found = true;
    }
  }
  return coordinate;
}

void GoalpostFinder::process_image(cv::Mat binBuffer)
{
  cv::Mat bw(camera_height, camera_width, CV_8UC1);
  cv::Mat bw2(camera_height, camera_width, CV_8UC1);
  cv::Mat bw1(camera_height, camera_width, CV_8UC1);
  cv::Mat check(camera_height, camera_width, CV_8UC1);
  std::vector<cv::Vec2f> lines;

  bw = binBuffer.clone();
  cv::cvtColor(bw, bw, cv::COLOR_BGR2GRAY);
  if (countNonZero(check) > 10) {
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    morphologyEx(bw, bw, cv::MORPH_OPEN, element);
    morphologyEx(bw1, bw1, cv::MORPH_OPEN, element);
    Canny(bw, bw2, 100, 200);
    HoughLines(bw2, lines, 1, CV_PI / 180, 25);
    if (lines.size() > 0) {
      process_line(lines, bw);
    }
    element.release();
  }
  if (left_goal_height > 0) {
    left_goal_distance = estimate_distance(left_goal_height);
  }
  if (right_goal_height > 0) {
    right_goal_distance = estimate_distance(right_goal_height);
  }
  bw.release();
}

void GoalpostFinder::process_line(std::vector<cv::Vec2f> garis, cv::Mat bw)
{
  std::vector<cv::Vec2f> verticalLine = line_filter(garis, VERTICAL_FILTER),
    horizontalLine = line_filter(garis, HORIZONTAL_FILTER);

  if (verticalLine.size() <= 0) {
    return;
  }
  cv::Vec2f meanVertical = get_line_mean(verticalLine);
  double variance = get_variance(verticalLine, meanVertical[0]);
  if (variance >= 1500) {
    std::vector<std::vector<cv::Vec2f>> splittedMainLine = split_line_from_mean(
      verticalLine,
      meanVertical);
    cv::Vec2f meanLeft = get_line_mean(splittedMainLine[0]),
      meanRight = get_line_mean(splittedMainLine[1]);
    double varianceLeft = get_variance(splittedMainLine[0], meanLeft[0]),
      varianceRight = get_variance(splittedMainLine[1], meanRight[0]);

    if (varianceLeft > 40) {
      std::vector<std::vector<cv::Vec2f>> splittedLine = split_line_from_mean(
        splittedMainLine[0],
        meanLeft);
      cv::Vec2f meanA = get_line_mean(splittedLine[0]);
      cv::Vec2f meanB = get_line_mean(splittedLine[1]);

      if (draw_output) {draw_hough_line(meanA, cv::Scalar(255, 255, 255));}
      if (draw_output) {draw_hough_line(meanB, cv::Scalar(0, 0, 0));}

      std::vector<cv::Point> boundA = find_vertical_bound(bw, meanA, LARGE);
      std::vector<cv::Point> boundB = find_vertical_bound(bw, meanB, LARGE);
      std::vector<cv::Point> real_bound;

      if (boundA.size() == 2 && boundB.size() == 2) {
        int y = (boundA[1].y - boundA[0].y) / 2 + boundA[0].y;
        int x =
          (-1 * sin(meanA[1]) / cos(meanA[1]) * static_cast<double>(y) + meanA[0] / cos(meanA[1]));
        cv::Point centerA(x, y);
        y = (boundB[1].y - boundB[0].y) / 2 + boundB[0].y;
        x =
          (-1 * sin(meanB[1]) / cos(meanB[1]) * static_cast<double>(y) + meanB[0] / cos(meanB[1]));
        cv::Point centerB(x, y);
        cv::Point real_center;
        double areaA =
          find_area_roi(
          bw, cv::Point(centerA.x - 5, centerA.y - 1),
          cv::Point(centerA.x + 5, centerA.y + 1));
        double areaB =
          find_area_roi(
          bw, cv::Point(centerB.x - 5, centerB.y - 1),
          cv::Point(centerB.x + 5, centerB.y + 1));
        int heightA = sqrt(
          pow(boundA[0].x - boundA[1].x, 2) + pow(
            boundA[0].y - boundA[1].y,
            2));
        int heightB = sqrt(
          pow(boundB[0].x - boundB[1].x, 2) + pow(
            boundB[0].y - boundB[1].y,
            2));

        /* syarat tiang sebenarnya:
         * memiliki area tengah terluas pada tiang yang terdeteksi memiliki tinggi hampir sama
         * atau memiliki tinggi tertinggi dengan syarat beda tingginya sangat terlihat
         */
        if (areaA > areaB && abs(heightA - heightB) < 20) {
          real_bound = boundA;
          real_center = centerA;
        } else if (areaA < areaB && abs(heightA - heightB) < 20) {
          real_bound = boundB;
          real_center = centerB;
        } else if (abs(heightA - heightB) > 20 && heightA > heightB) {
          real_bound = boundA;
          real_center = centerA;
        } else if (abs(heightA - heightB) > 20 && heightA < heightB) {
          real_bound = boundB;
          real_center = centerB;
        }
      }
    } else {
      if (draw_output) {draw_hough_line(meanLeft, cv::Scalar(0, 255, 0));}
      std::vector<cv::Point> boundLeft = find_vertical_bound(bw, meanLeft, LARGE);
      if (boundLeft.size() == 2) {
        if (draw_output) {
          cv::line(image, boundLeft[0], boundLeft[1], cv::Scalar(255, 255, 255), 3, 8);
        }
        if (draw_output) {
          cv::line(image, boundLeft[1], boundLeft[1], cv::Scalar(255, 255, 255), 3, 8);
        }
        int cent_y = (boundLeft[1].y - boundLeft[0].y) / 2 + boundLeft[0].y;
        int cent_x =
          (-1 * sin(meanLeft[1]) / cos(meanLeft[1]) * static_cast<double>(cent_y) + meanLeft[0] /
          cos(meanLeft[1]));
        if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
        int height =
          sqrt(
          pow(boundLeft[0].x - boundLeft[1].x, 2) + pow(
            boundLeft[0].y - boundLeft[1].y,
            2));

        left_goal_height = height;
        left_goal_coor = cv::Point(cent_x, cent_y);
        goal_detected = false;
        if (draw_output) {
          cv::putText(image, "TLY", boundLeft[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BLY", boundLeft[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        }
      }
    }

    if (varianceRight > 40) {
      std::vector<std::vector<cv::Vec2f>> splittedLine = split_line_from_mean(
        splittedMainLine[1],
        meanRight);
      cv::Vec2f meanA = get_line_mean(splittedLine[0]);
      cv::Vec2f meanB = get_line_mean(splittedLine[1]);
      if (draw_output) {draw_hough_line(meanA, cv::Scalar(255, 255, 255));}
      if (draw_output) {draw_hough_line(meanB, cv::Scalar(0, 0, 0));}
      std::vector<cv::Point> boundA = find_vertical_bound(bw, meanA, LARGE);
      std::vector<cv::Point> boundB = find_vertical_bound(bw, meanB, LARGE);
      std::vector<cv::Point> real_bound;
      if (boundA.size() < 2 || boundB.size() < 2) {
        return;
      }
      int y = (boundA[1].y - boundA[0].y) / 2 + boundA[0].y;
      int x = (-1 * sin(meanA[1]) / cos(meanA[1]) * static_cast<double>(y) + meanA[0] / cos(
          meanA[1]));
      cv::Point centerA(x, y);
      y = (boundB[1].y - boundB[0].y) / 2 + boundB[0].y;
      x = (-1 * sin(meanB[1]) / cos(meanB[1]) * static_cast<double>(y) + meanB[0] / cos(meanB[1]));
      cv::Point centerB(x, y);
      cv::Point real_center;
      double areaA =
        find_area_roi(
        bw, cv::Point(centerA.x - 5, centerA.y - 1),
        cv::Point(centerA.x + 5, centerA.y + 1));
      double areaB =
        find_area_roi(
        bw, cv::Point(centerB.x - 5, centerB.y - 1),
        cv::Point(centerB.x + 5, centerB.y + 1));
      int heightA =
        sqrt(pow(boundA[0].x - boundA[1].x, 2) + pow(boundA[0].y - boundA[1].y, 2));
      int heightB =
        sqrt(pow(boundB[0].x - boundB[1].x, 2) + pow(boundB[0].y - boundB[1].y, 2));
      int real_height;

      /* syarat tiang sebenarnya:
       * memiliki area tengah terluas pada tiang yang terdeteksi memiliki tinggi hampir sama
       * atau memiliki tinggi tertinggi dengan syarat beda tingginya sangat terlihat
       */
      if (areaA > areaB && abs(heightA - heightB) < 20) {
        real_bound = boundA;
        real_center = centerA;
        real_height = heightA;
      } else if (areaA < areaB && abs(heightA - heightB) < 20) {
        real_bound = boundB;
        real_center = centerB;
        real_height = heightB;
      } else if (abs(heightA - heightB) > 20 && heightA > heightB) {
        real_bound = boundA;
        real_center = centerA;
        real_height = heightA;
      } else if (abs(heightA - heightB) > 20 && heightA < heightB) {
        real_bound = boundB;
        real_center = centerB;
        real_height = heightB;
      } else {
        return;
      }

      if (draw_output) {
        cv::line(image, real_bound[0], real_bound[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {
        cv::line(image, real_bound[1], real_bound[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {circle(image, real_center, 5, cv::Scalar(0, 0, 255), -1);}
      right_goal_height = real_height;
      right_goal_coor = real_center;
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TRY", real_bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BRY", real_bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    } else {
      if (draw_output) {draw_hough_line(meanRight, cv::Scalar(0, 255, 0));}
      std::vector<cv::Point> boundRight = find_vertical_bound(bw, meanRight, LARGE);
      if (boundRight.size() < 2) {
        return;
      }
      if (draw_output) {
        cv::line(image, boundRight[0], boundRight[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {
        cv::line(image, boundRight[1], boundRight[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      int cent_y = (boundRight[1].y - boundRight[0].y) / 2 + boundRight[0].y;
      int cent_x =
        (-1 * sin(meanRight[1]) / cos(meanRight[1]) * static_cast<double>(cent_y) + meanRight[0] /
        cos(meanRight[1]));
      if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
      int height =
        sqrt(
        pow(
          boundRight[0].x - boundRight[1].x,
          2) + pow(boundRight[0].y - boundRight[1].y, 2));

      right_goal_height = height;
      right_goal_coor = cv::Point(cent_x, cent_y);
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TRY", boundRight[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BRY", boundRight[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    }
  } else {
    if (draw_output) {draw_hough_line(meanVertical, cv::Scalar(0, 255, 0));}
    std::vector<cv::Point> bound = find_vertical_bound(bw, meanVertical, LARGE);
    if (bound.size() < 2) {
      return;
    }

    if (draw_output) {cv::line(image, bound[0], bound[1], cv::Scalar(255, 255, 255), 3, 8);}
    int cent_y = (bound[1].y - bound[0].y) / 2 + bound[0].y;
    int cent_x =
      (-1 * sin(meanVertical[1]) / cos(meanVertical[1]) * static_cast<double>(cent_y) +
      meanVertical[0] /
      cos(meanVertical[1]));
    if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
    int height = sqrt(pow(bound[0].x - bound[1].x, 2) + pow(bound[0].y - bound[1].y, 2));
    if (height < 20) {
      return;
    }

    double aArea = 0;
    int aReg = 0;
    for (int aX = bound[0].x; aX > bound[0].x - 30; aX -= 2) {
      if (aX < 0) {break;}
      int aY = bound[0].y + 2;
      aArea += find_area_roi(bw, cv::Point(aX - 1, aY - 3), cv::Point(aX, aY + 3));
      aReg++;
    }
    aArea = aArea / aReg;

    double bArea = 0;
    int bReg = 0;
    for (int bX = bound[0].x; bX < bound[0].x + 30; bX += 2) {
      if (bX > camera_width) {break;}
      int bY = bound[0].y + 2;
      bArea += find_area_roi(bw, cv::Point(bX - 1, bY - 3), cv::Point(bX, bY + 3));
      bReg++;
    }
    bArea = bArea / bReg;
    if (aArea - bArea > 20) {
      right_goal_height = height;
      right_goal_coor = cv::Point(cent_x, cent_y);
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TRY", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BRY", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    } else if (aArea - bArea < -20) {
      left_goal_height = height;
      left_goal_coor = cv::Point(cent_x, cent_y);
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TLY", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BLY", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    } else {
      goal_detected = true;
      if (draw_output) {
        cv::putText(image, "?", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "?", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    }
  }
}

void GoalpostFinder::process_line(std::vector<cv::Vec2f> garis, cv::Mat bw, cv::Mat inp)
{
  std::vector<cv::Vec2f> verticalLine = line_filter(garis, VERTICAL_FILTER),
    horizontalLine = line_filter(garis, HORIZONTAL_FILTER);

  std::vector<cv::Point> tmp_realBound;

  bool kiridapat = false;

  if (verticalLine.size() <= 0) {
    return;
  }
  cv::Vec2f meanVertical = get_line_mean(verticalLine);
  double variance = get_variance(verticalLine, meanVertical[0]);
  if (variance >= 1500) {
    std::vector<std::vector<cv::Vec2f>> splittedMainLine = split_line_from_mean(
      verticalLine,
      meanVertical);
    cv::Vec2f meanLeft = get_line_mean(splittedMainLine[0]),
      meanRight = get_line_mean(splittedMainLine[1]);
    double varianceLeft = get_variance(splittedMainLine[0], meanLeft[0]),
      varianceRight = get_variance(splittedMainLine[1], meanRight[0]);

    if (varianceLeft > 40) {
      std::vector<std::vector<cv::Vec2f>> splittedLine = split_line_from_mean(
        splittedMainLine[0],
        meanLeft);
      cv::Vec2f meanA = get_line_mean(splittedLine[0]);
      cv::Vec2f meanB = get_line_mean(splittedLine[1]);
      if (draw_output) {draw_hough_line(meanA, cv::Scalar(255, 255, 255));}
      if (draw_output) {draw_hough_line(meanB, cv::Scalar(0, 0, 0));}


      std::vector<cv::Point> boundA = find_vertical_bound(bw, meanA, LARGE, inp);
      std::vector<cv::Point> boundB = find_vertical_bound(bw, meanB, LARGE, inp);
      std::vector<cv::Point> real_bound;

      if (boundA.size() == 2 && boundB.size() == 2) {
        int y = (boundA[1].y - boundA[0].y) / 2 + boundA[0].y;
        int x =
          (-1 * sin(meanA[1]) / cos(meanA[1]) * static_cast<double>(y) + meanA[0] / cos(meanA[1]));
        cv::Point centerA(x, y);
        y = (boundB[1].y - boundB[0].y) / 2 + boundB[0].y;
        x =
          (-1 * sin(meanB[1]) / cos(meanB[1]) * static_cast<double>(y) + meanB[0] / cos(meanB[1]));
        cv::Point centerB(x, y);
        cv::Point real_center;
        double areaA =
          find_area_roi(
          bw, cv::Point(centerA.x - 5, centerA.y - 1),
          cv::Point(centerA.x + 5, centerA.y + 1));
        double areaB =
          find_area_roi(
          bw, cv::Point(centerB.x - 5, centerB.y - 1),
          cv::Point(centerB.x + 5, centerB.y + 1));
        int heightA = sqrt(
          pow(boundA[0].x - boundA[1].x, 2) + pow(
            boundA[0].y - boundA[1].y,
            2));
        int heightB = sqrt(
          pow(boundB[0].x - boundB[1].x, 2) + pow(
            boundB[0].y - boundB[1].y,
            2));

        /* syarat tiang sebenarnya:
         * memiliki area tengah terluas pada tiang yang terdeteksi memiliki tinggi hampir sama
         * atau memiliki tinggi tertinggi dengan syarat beda tingginya sangat terlihat
         */
        if (areaA > areaB && abs(heightA - heightB) < 20) {
          real_bound = boundA;
          real_center = centerA;
        } else if (areaA < areaB && abs(heightA - heightB) < 20) {
          real_bound = boundB;
          real_center = centerB;
        } else if (abs(heightA - heightB) > 20 && heightA > heightB) {
          real_bound = boundA;
          real_center = centerA;
        } else if (abs(heightA - heightB) > 20 && heightA < heightB) {
          real_bound = boundB;
          real_center = centerB;
        }
      } else {
        kiridapat = false;
      }
    } else {
      if (draw_output) {draw_hough_line(meanLeft, cv::Scalar(0, 255, 0));}
      std::vector<cv::Point> boundLeft = find_vertical_bound(bw, meanLeft, LARGE, inp);
      if (boundLeft.size() == 2) {
        if (draw_output) {
          cv::line(image, boundLeft[0], boundLeft[1], cv::Scalar(255, 255, 255), 3, 8);
        }
        if (draw_output) {
          cv::line(image, boundLeft[1], boundLeft[1], cv::Scalar(255, 255, 255), 3, 8);
        }
        int cent_y = (boundLeft[1].y - boundLeft[0].y) / 2 + boundLeft[0].y;
        int cent_x =
          (-1 * sin(meanLeft[1]) / cos(meanLeft[1]) * static_cast<double>(cent_y) + meanLeft[0] /
          cos(meanLeft[1]));
        if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
        int height =
          sqrt(
          pow(boundLeft[0].x - boundLeft[1].x, 2) + pow(
            boundLeft[0].y - boundLeft[1].y,
            2));

        left_goal_height = height;
        left_goal_coor = cv::Point(cent_x, cent_y);
        tmp_realBound = boundLeft;
        goal_detected = false;
        kiridapat = true;
      } else {
        kiridapat = false;
      }
    }

    if (varianceRight > 40) {
      std::vector<std::vector<cv::Vec2f>> splittedLine = split_line_from_mean(
        splittedMainLine[1],
        meanRight);
      cv::Vec2f meanA = get_line_mean(splittedLine[0]);
      cv::Vec2f meanB = get_line_mean(splittedLine[1]);
      if (draw_output) {draw_hough_line(meanA, cv::Scalar(255, 255, 255));}
      if (draw_output) {draw_hough_line(meanB, cv::Scalar(0, 0, 0));}
      std::vector<cv::Point> boundA = find_vertical_bound(bw, meanA, LARGE, inp);
      std::vector<cv::Point> boundB = find_vertical_bound(bw, meanB, LARGE, inp);
      std::vector<cv::Point> real_bound;
      if (boundA.size() < 2 || boundB.size() < 2) {
        if (kiridapat) {
          right_goal_height = left_goal_height;
          right_goal_coor = left_goal_coor;
          left_goal_height = -1;
          left_goal_coor = cv::Point(-1, -1);
          if (draw_output) {
            cv::putText(image, "TRY", tmp_realBound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
            cv::putText(image, "BRY", tmp_realBound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          }
        }
        return;
      }
      int y = (boundA[1].y - boundA[0].y) / 2 + boundA[0].y;
      int x = (-1 * sin(meanA[1]) / cos(meanA[1]) * static_cast<double>(y) + meanA[0] / cos(
          meanA[1]));
      cv::Point centerA(x, y);
      y = (boundB[1].y - boundB[0].y) / 2 + boundB[0].y;
      x = (-1 * sin(meanB[1]) / cos(meanB[1]) * static_cast<double>(y) + meanB[0] / cos(meanB[1]));
      cv::Point centerB(x, y);
      cv::Point real_center;
      double areaA =
        find_area_roi(
        bw, cv::Point(centerA.x - 5, centerA.y - 1),
        cv::Point(centerA.x + 5, centerA.y + 1)),
        areaB =
        find_area_roi(
        bw, cv::Point(centerB.x - 5, centerB.y - 1),
        cv::Point(centerB.x + 5, centerB.y + 1));
      int heightA =
        sqrt(pow(boundA[0].x - boundA[1].x, 2) + pow(boundA[0].y - boundA[1].y, 2)),
        heightB = sqrt(pow(boundB[0].x - boundB[1].x, 2) + pow(boundB[0].y - boundB[1].y, 2)),
        real_height;

      /* syarat tiang sebenarnya:
       * memiliki area tengah terluas pada tiang yang terdeteksi memiliki tinggi hampir sama
       * atau memiliki tinggi tertinggi dengan syarat beda tingginya sangat terlihat
       */
      if (areaA > areaB && abs(heightA - heightB) < 20) {
        real_bound = boundA;
        real_center = centerA;
        real_height = heightA;
      } else if (areaA < areaB && abs(heightA - heightB) < 20) {
        real_bound = boundB;
        real_center = centerB;
        real_height = heightB;
      } else if (abs(heightA - heightB) > 20 && heightA > heightB) {
        real_bound = boundA;
        real_center = centerA;
        real_height = heightA;
      } else if (abs(heightA - heightB) > 20 && heightA < heightB) {
        real_bound = boundB;
        real_center = centerB;
        real_height = heightB;
      } else {
        return;
      }

      if (draw_output) {
        cv::line(image, real_bound[0], real_bound[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {
        cv::line(image, real_bound[1], real_bound[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {circle(image, real_center, 5, cv::Scalar(0, 0, 255), -1);}


      if (kiridapat) {
        right_goal_height = real_height;
        right_goal_coor = real_center;
        goal_detected = false;
        if (draw_output) {
          cv::putText(image, "TRY", real_bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BRY", real_bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);

          cv::putText(image, "TLY", tmp_realBound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BLY", tmp_realBound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        }
      } else {
        left_goal_height = real_height;
        left_goal_coor = real_center;
        goal_detected = false;
        if (draw_output) {
          cv::putText(image, "TLY", real_bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BLY", real_bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        }
      }
    } else {
      if (draw_output) {draw_hough_line(meanRight, cv::Scalar(0, 255, 0));}
      std::vector<cv::Point> boundRight = find_vertical_bound(bw, meanRight, LARGE, inp);
      if (boundRight.size() < 2) {
        if (kiridapat) {
          right_goal_height = left_goal_height;
          right_goal_coor = left_goal_coor;
          left_goal_height = -1;
          left_goal_coor = cv::Point(-1, -1);
          std::cout << tmp_realBound[0].x << ", " << tmp_realBound[0].y << std::endl;
          std::cout << tmp_realBound[1].x << ", " << tmp_realBound[1].y << std::endl;
          if (draw_output) {
            cv::putText(image, "TRY", tmp_realBound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
            cv::putText(image, "BRY", tmp_realBound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          }
        }
        return;
      }
      if (draw_output) {
        cv::line(image, boundRight[0], boundRight[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      if (draw_output) {
        cv::line(image, boundRight[1], boundRight[1], cv::Scalar(255, 255, 255), 3, 8);
      }
      int cent_y = (boundRight[1].y - boundRight[0].y) / 2 + boundRight[0].y;
      int cent_x =
        (-1 * sin(meanRight[1]) / cos(meanRight[1]) * static_cast<double>(cent_y) + meanRight[0] /
        cos(meanRight[1]));
      if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
      int height =
        sqrt(
        pow(
          boundRight[0].x - boundRight[1].x,
          2) + pow(boundRight[0].y - boundRight[1].y, 2));

      if (kiridapat) {
        right_goal_height = height;
        right_goal_coor = cv::Point(cent_x, cent_y);
        goal_detected = false;
        if (draw_output) {
          cv::putText(image, "TRY", boundRight[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BRY", boundRight[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);

          cv::putText(image, "TLY", tmp_realBound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BLY", tmp_realBound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        }
      } else {
        left_goal_height = height;
        left_goal_coor = cv::Point(cent_x, cent_y);
        goal_detected = false;
        if (draw_output) {
          cv::putText(image, "TLY", boundRight[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
          cv::putText(image, "BLY", boundRight[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        }
      }
    }
  } else {
    if (draw_output) {draw_hough_line(meanVertical, cv::Scalar(0, 255, 0));}
    std::vector<cv::Point> bound = find_vertical_bound(bw, meanVertical, LARGE, inp);
    if (bound.size() < 2) {
      return;
    }

    if (draw_output) {cv::line(image, bound[0], bound[1], cv::Scalar(255, 255, 255), 3, 8);}
    int cent_y = (bound[1].y - bound[0].y) / 2 + bound[0].y;
    int cent_x =
      (-1 * sin(meanVertical[1]) / cos(meanVertical[1]) * static_cast<double>(cent_y) +
      meanVertical[0] /
      cos(meanVertical[1]));
    if (draw_output) {circle(image, cv::Point(cent_x, cent_y), 5, cv::Scalar(0, 0, 255), -1);}
    int height = sqrt(pow(bound[0].x - bound[1].x, 2) + pow(bound[0].y - bound[1].y, 2));
    if (height < 20) {
      return;
    }

    double aArea = 0;
    int aReg = 0;
    for (int aX = bound[0].x; aX > bound[0].x - 30; aX -= 2) {
      if (aX < 0) {break;}
      int aY = bound[0].y + 2;
      aArea += find_area_roi(bw, cv::Point(aX - 1, aY - 3), cv::Point(aX, aY + 3));
      aReg++;
    }
    aArea = aArea / aReg;

    double bArea = 0;
    int bReg = 0;
    for (int bX = bound[0].x; bX < bound[0].x + 30; bX += 2) {
      if (bX > camera_width) {break;}
      int bY = bound[0].y + 2;
      bArea += find_area_roi(bw, cv::Point(bX - 1, bY - 3), cv::Point(bX, bY + 3));
      bReg++;
    }
    bArea = bArea / bReg;
    if (aArea - bArea > 20) {
      right_goal_height = height;
      right_goal_coor = cv::Point(cent_x, cent_y);
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TRY", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BRY", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    } else if (aArea - bArea < -20) {
      left_goal_height = height;
      left_goal_coor = cv::Point(cent_x, cent_y);
      goal_detected = false;
      if (draw_output) {
        cv::putText(image, "TLY", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "BLY", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    } else {
      goal_detected = true;
      if (draw_output) {
        cv::putText(image, "?", bound[0], 1, 1.2, cv::Scalar(0, 255, 0), 1);
        cv::putText(image, "?", bound[1], 1, 1.2, cv::Scalar(0, 255, 0), 1);
      }
    }
  }
}

std::vector<cv::Vec2f> GoalpostFinder::line_filter(
  std::vector<cv::Vec2f> input_line,
  FilterLineOption option)
{
  std::vector<cv::Vec2f> garis;
  for (int i = 0; i < static_cast<int>(input_line.size()); i++) {
    double tetha = input_line[i][1];
    if ((tetha >= 0.785 && tetha <= 2.355) || (tetha >= 3.925 && tetha <= 5.495)) {
      if (option == HORIZONTAL_FILTER) {
        garis.push_back(input_line[i]);
      }
    } else {
      if (option == VERTICAL_FILTER) {
        garis.push_back(input_line[i]);
      }
    }
  }
  return garis;
}

std::vector<cv::Point> GoalpostFinder::find_vertical_bound(
  cv::Mat bw, cv::Vec2f garis,
  ScanArea option)
{
  std::vector<cv::Point> abc;
  bool whiteStart = false;
  int plus = option == LARGE ? 5 : 1;
  for (int y = 0; y < bw.rows; y += 1) {
    double x =
      (-1 * sin(garis[1]) / cos(garis[1]) * static_cast<double>(y) + garis[0] / cos(garis[1]));
    double whiteArea;
    if (option == LARGE) {
      whiteArea = find_area_roi(bw, cv::Point(x - plus, y - 5), cv::Point(x + plus, y));
    } else {
      whiteArea = find_area_roi(bw, cv::Point(x, y - 1), cv::Point(x + 1, y));
    }
    if (whiteArea > 10 && whiteStart == false) {
      abc.push_back(cv::Point(x, y));
      whiteStart = true;
    } else if ((whiteArea < 10 || y == bw.rows - 1) && whiteStart == true) {
      abc.push_back(cv::Point(x, y));
      break;
    }
  }
  return abc;
}

std::vector<cv::Point> GoalpostFinder::find_vertical_bound(
  cv::Mat bw, cv::Vec2f garis, ScanArea option,
  cv::Mat inp)
{
  std::vector<cv::Point> abc;
  bool whiteStart = false;
  int plus = option == LARGE ? 5 : 1;
  for (int y = bw.rows - 1; y >= 0; y -= 1) {
    double x =
      (-1 * sin(garis[1]) / cos(garis[1]) * static_cast<double>(y) + garis[0] / cos(garis[1]));
    double whiteArea;
    double greenArea;
    if (option == LARGE) {
      whiteArea = find_area_roi(bw, cv::Point(x - plus, y - 5), cv::Point(x + plus, y));
    } else {
      whiteArea = find_area_roi(bw, cv::Point(x, y - 1), cv::Point(x + 1, y));
    }

    if ((whiteArea > 10) && whiteStart == false) {
      greenArea = find_area_roi(inp, cv::Point(x - 30, y), cv::Point(x + 30, y + 30));
      if (greenArea > 10.0 || y == bw.rows - 1) {
        abc.push_back(cv::Point(x, y));
        whiteStart = true;
      }
    } else if ((whiteArea < 10) && whiteStart == true) {
      cv::Point tmp = abc[0];
      abc.pop_back();
      abc.push_back(cv::Point(x, y));
      abc.push_back(tmp);

      break;
    }
  }
  return abc;
}

cv::Vec2f GoalpostFinder::get_line_mean(std::vector<cv::Vec2f> lines)
{
  double mean_rho = 0.0, mean_alpha_sin = 0.0, mean_alpha_cos = 0.0;
  cv::Vec2f mean;
  for (size_t i = 0; i < lines.size(); i++) {
    mean_rho += abs(lines[i][0]);
    if (lines[i][0] < 0) {
      mean_alpha_sin += sin(lines[i][1] - CV_PI);
      mean_alpha_cos += cos(lines[i][1] - CV_PI);
    } else {
      mean_alpha_sin += sin(lines[i][1]);
      mean_alpha_cos += cos(lines[i][1]);
    }
  }
  mean[0] = mean_rho / lines.size();
  mean[1] = atan2((0.5 * mean_alpha_sin), (0.5 * mean_alpha_cos));
  return mean;
}

double GoalpostFinder::get_variance(std::vector<cv::Vec2f> lines, double mean_rho)
{
  double sum = 0.0;
  for (size_t i = 0; i < lines.size(); i++) {
    sum += pow((abs(lines[i][0]) - mean_rho), 2);
  }

  return sum / lines.size();
}

void GoalpostFinder::draw_hough_line(cv::Vec2f garis, cv::Scalar color)
{
  double rho = garis[0], theta = garis[1];
  double a = cos(theta), b = sin(theta);
  double x0 = a * rho, y0 = b * rho;
  cv::Point pt1, pt2;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  cv::line(image, pt1, pt2, color, 1, 8);
}

std::vector<std::vector<cv::Vec2f>> GoalpostFinder::split_line_from_mean(
  std::vector<cv::Vec2f> garis,
  cv::Vec2f mean_line)
{
  std::vector<std::vector<cv::Vec2f>> join(2);
  for (int i = 0; i < static_cast<int>(garis.size()); i++) {
    double cRho = abs(garis[i][0]);
    if (cRho < mean_line[0]) {
      join[0].push_back(garis[i]);
    } else {
      join[1].push_back(garis[i]);
    }
  }
  return join;
}

double GoalpostFinder::estimate_distance(double height)
{
  return (POLY_C * height * height) + (POLY_B * height) + POLY_A;
}

double GoalpostFinder::get_left_goal_distance()
{
  return left_goal_distance;
}

double GoalpostFinder::get_right_goal_distance()
{
  return right_goal_distance;
}

}  // namespace ninshiki_opencv
