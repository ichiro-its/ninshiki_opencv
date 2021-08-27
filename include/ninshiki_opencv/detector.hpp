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

#ifndef NINSHIKI_OPENCV__DETECTOR_HPP_
#define NINSHIKI_OPENCV__DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <ninshiki_opencv/goalpost_finder.hpp>

#include <memory>

#include "./vision.h"

namespace ninshiki_opencv
{
class Detector
{
public:
  Detector();

  void vision_process(cv::Mat image_hsv, cv::Mat image_rgb);

  // cv::Mat get_image(std::shared_ptr<SensorMeasurements> sensor);
  const Contours & get_field_contours() const;

  float get_ball_pos_x() {return ball_pos_x;}
  float get_ball_pos_y() {return ball_pos_y;}

  float get_post_left_x() {return post_left_x;}
  float get_post_left_y() {return post_left_y;}
  float get_post_right_x() {return post_right_x;}
  float get_post_right_y() {return post_right_y;}

  int get_detect_goal_post_by() {return detect_goal_post_by;}

  void detect_goal_by_threshold() {detect_goal_post_by = BY_THRESHOLD;}
  void detect_goal_by_threshold_and_hough() {detect_goal_post_by = BY_THRESHOLD_AND_HOUGH;}

  void set_detect_goal_post(const bool & detect);

private:
  enum
  {
    BY_THRESHOLD,
    BY_THRESHOLD_AND_HOUGH
  };

  int detect_goal_post_by;

  std::shared_ptr<ColorClassifier> field_classifier;
  std::shared_ptr<LBPClassifier> lbp_classifier;
  GoalpostFinder goal_post;
  Contours field_contours;

  float ball_pos_x;
  float ball_pos_y;

  float post_left_x;
  float post_left_y;
  float post_right_x;
  float post_right_y;

  bool detect_goal_post;
};

}  // namespace ninshiki_opencv

#endif  // NINSHIKI_OPENCV__DETECTOR_HPP_
