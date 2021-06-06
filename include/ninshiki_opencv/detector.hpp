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
#include <robocup_client/robocup_client.hpp>

#include <memory>

#include "./vision.h"

namespace ninshiki_opencv
{
class Detector
{
public:
  Detector();

  void vision_process(cv::Mat image_hsv, cv::Mat image_rgb);

  cv::Mat get_image(std::shared_ptr<SensorMeasurements> sensor);

  float get_ball_pos_x() {return ball_pos_x;}
  float get_ball_pos_y() {return ball_pos_y;}

private:
  std::shared_ptr<ColorClassifier> field_classifier;

  float ball_pos_x;
  float ball_pos_y;
};

}  // namespace ninshiki_opencv

#endif  // NINSHIKI_OPENCV__DETECTOR_HPP_
