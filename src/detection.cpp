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

#include <ninshiki_opencv/detection.hpp>
#include <memory>
#include <string>

namespace ninshiki_opencv
{
Detection::Detection()
: field_classifier(new ColorClassifier(ColorClassifier::CLASSIFIER_TYPE_FIELD))
{
  // field_classifier->setHue(89);
  // field_classifier->setHueTolerance(54);
  // field_classifier->setMinSaturation(20);
  // field_classifier->setMaxSaturation(100);
  // field_classifier->setMinValue(28);
  // field_classifier->setMaxValue(100);

  field_classifier->set_hue_min(33);
  field_classifier->set_hue_max(68);
  field_classifier->set_sat_min(102);
  field_classifier->set_sat_max(255);
  field_classifier->set_val_min(0);
  field_classifier->set_val_max(255);

  ball_pos_x = -1;
  ball_pos_y = -1;
}

cv::Mat Detection::get_image(std::shared_ptr<SensorMeasurements> sensor)
{
  cv::Mat temp;

  auto camera = sensor.get()->cameras(0);
  cv::Mat coba(static_cast<int>(camera.height()),
    static_cast<int>(camera.width()), CV_8UC3, std::string(
      camera.image()).data());

  temp = coba.clone();
  return temp;
}

void Detection::vision_process(cv::Mat image_hsv, cv::Mat image_rgb)
{
  LBPClassifier * lbp_classifier = new LBPClassifier(LBPClassifier::CLASSIFIER_TYPE_BALL);

  cv::Size mat_size = image_hsv.size();
  cv::Mat field_binary_mat = field_classifier->classify_hsv(image_hsv);
  // cv::Mat ball_binary_mat = ball_classifier->classify(image_hsv);

  Contours field_contours(field_binary_mat);
  field_contours.filterLargerThen(700.0);
  field_contours.joinAll();
  field_contours.convexHull();
  field_contours.getContours();
  // std::cout << field_contours.minX() << std::endl;

  cv::Mat field_contours_mat = field_contours.getBinaryMatLine(mat_size, 4);
  cv::Mat lbp_input = image_rgb.clone();
  cv::cvtColor(lbp_input, lbp_input, cv::COLOR_BGR2GRAY);

  cv::bitwise_and(field_contours.getBinaryMat(mat_size), lbp_input, lbp_input);
  cv::cvtColor(lbp_input, lbp_input, cv::COLOR_GRAY2BGR);

  Rects ball_rects;
  ball_rects = lbp_classifier->classify(lbp_input);
  ball_rects.filterLargest();

  ball_pos_x = ball_rects.getFirstRectCenter().x;
  ball_pos_y = ball_rects.getFirstRectCenter().y;

  // return field_binary_mat;
}

}  // namespace ninshiki_opencv
