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

#ifndef NINSHIKI_OPENCV__GOALPOST_FINDER_HPP_
#define NINSHIKI_OPENCV__GOALPOST_FINDER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <robocup_client/robocup_client.hpp>

#include <memory>
#include <vector>

#include "math/point.h"

namespace ninshiki_opencv
{
class GoalpostFinder
{
public:
  GoalpostFinder();
  std::vector<cv::Point> detect_goal(cv::Mat sensor_image);
  std::vector<cv::Point> detect_goal_by_hough(cv::Mat sensor_image);

  double get_left_goal_distance();
  double get_right_goal_distance();

  void set_detect_goal_post_by(int i) {detect_goal_post_by = i;}

private:
  enum FilterLineOption
  {
    HORIZONTAL_FILTER,
    VERTICAL_FILTER
  };

  enum ScanArea
  {
    LARGE,
    SMALL
  };

  int detect_goal_post_by;

  bool goal_detected;
  bool left_goal_found;
  bool right_goal_found;
  int left_goal_height;
  int right_goal_height;
  int bawah_tiang;

  cv::Mat output_buffer;
  cv::Mat image;

  float camera_height;
  float camera_width;

  bool draw_output;
  double left_goal_distance;
  double right_goal_distance;

  cv::Point left_goal_coor;
  cv::Point right_goal_coor;

  double get_variance(std::vector<cv::Vec2f> lines, double mean_rho);
  cv::Vec2f get_line_mean(std::vector<cv::Vec2f> lines);

  void process_image(cv::Mat binBuffer);
  void process_line(std::vector<cv::Vec2f> garis, cv::Mat bw);
  void process_line(std::vector<cv::Vec2f> garis, cv::Mat bw, cv::Mat inp_lap);

  double find_area_roi(cv::Mat fr, cv::Point stR, cv::Point spR);
  std::vector<cv::Vec2f> line_filter(std::vector<cv::Vec2f> inputLine, FilterLineOption option);
  std::vector<cv::Point> find_vertical_bound(cv::Mat bw, cv::Vec2f garis, ScanArea option);
  std::vector<cv::Point> find_vertical_bound(
    cv::Mat bw, cv::Vec2f garis, ScanArea option,
    cv::Mat inp_lap);
  void draw_hough_line(cv::Vec2f garis, cv::Scalar color);
  std::vector<std::vector<cv::Vec2f>> split_line_from_mean(
    std::vector<cv::Vec2f> garis,
    cv::Vec2f meanLine);
  double estimate_distance(double height);
};

}  // namespace ninshiki_opencv

#endif  // NINSHIKI_OPENCV__GOALPOST_FINDER_HPP_
