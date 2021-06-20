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

#include <ninshiki_opencv/detector.hpp>
#include <memory>
#include <string>

namespace ninshiki_opencv
{
Detector::Detector()
: field_classifier(std::make_shared<ColorClassifier>(ColorClassifier::CLASSIFIER_TYPE_FIELD)),
  lbp_classifier(std::make_shared<LBPClassifier>(LBPClassifier::CLASSIFIER_TYPE_BALL))
{
  field_classifier->setHue(71);
  field_classifier->setHueTolerance(25);
  field_classifier->setMinSaturation(40);
  field_classifier->setMaxSaturation(100);
  field_classifier->setMinValue(0);
  field_classifier->setMaxValue(100);

  ball_pos_x = -1;
  ball_pos_y = -1;

  post_left_x = -1;
  post_left_y = -1;
  post_right_x = -1;
  post_right_y = -1;

  detect_goal_post = false;
}

cv::Mat Detector::get_image(std::shared_ptr<SensorMeasurements> sensor)
{
  cv::Mat temp;

  auto camera = sensor.get()->cameras(0);
  cv::Mat sensor_image(static_cast<int>(camera.height()),
    static_cast<int>(camera.width()), CV_8UC3, std::string(
      camera.image()).data());

  temp = sensor_image.clone();
  return temp;
}

const Contours & Detector::get_field_contours() const
{
  return field_contours;
}

void Detector::set_detect_goal_post(const bool & detect)
{
  detect_goal_post = detect;
}

void Detector::vision_process(cv::Mat image_hsv, cv::Mat image_rgb)
{
  cv::Size mat_size = image_hsv.size();
  cv::Mat field_binary_mat = field_classifier->classify(image_hsv);

  field_contours = Contours(field_binary_mat);
  field_contours.filterLargerThen(700.0);
  field_contours.joinAll();
  field_contours.convexHull();
  field_contours.getContours();

  cv::Mat field_contours_mat = field_contours.getBinaryMatLine(mat_size, 4);
  cv::Mat lbp_input = image_rgb.clone();
  cv::cvtColor(lbp_input, lbp_input, cv::COLOR_BGR2GRAY);

  cv::bitwise_and(field_contours.getBinaryMat(mat_size), lbp_input, lbp_input);

  cv::cvtColor(lbp_input, lbp_input, cv::COLOR_GRAY2BGR);

  Rects ball_rects;
  ball_rects = lbp_classifier->classify(lbp_input);
  ball_rects.filterLargest();

  goal_post.set_detect_goal_post_by(detect_goal_post_by);
  if (detect_goal_post) {
    auto coordinate = goal_post.detect_goal(image_rgb);
    post_left_x = coordinate[0].x;
    post_left_y = coordinate[0].y;
    post_right_x = coordinate[1].x;
    post_right_y = coordinate[1].y;
  }

  ball_pos_x = ball_rects.getFirstRectCenter().x;
  ball_pos_y = ball_rects.getFirstRectCenter().y;
}

}  // namespace ninshiki_opencv
