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

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <ninshiki_opencv/detector.hpp>
#include <robocup_client/robocup_client.hpp>

#include <unistd.h>
#include <fstream>

#include <iostream>
#include <string>

int main(int argc, char * argv[])
{
  if (argc < 3) {
    std::cerr << "Please specify the host and the port!" << std::endl;
    return 0;
  }
  std::string host = argv[1];
  int port = std::stoi(argv[2]);

  robocup_client::RobotClient client(host, port);
  if (!client.connect()) {
    std::cerr << "Failed to connect to server on port " <<
      client.get_port() << "!" << std::endl;

    return 1;
  }

  robocup_client::MessageHandler message;
  message.add_sensor_time_step("Camera", 16);

  cv::Mat frame;
  cv::Mat frame_hsv;
  cv::Mat field_mask;

  while (client.get_tcp_socket()->is_connected()) {
    client.send(*message.get_actuator_request());
    auto sensors = client.receive();

    if (sensors.get()->cameras_size() > 0) {
      std::cout << sensors.get()->cameras(0).height() << std::endl;

      ninshiki_opencv::Detection detection;
      cv::Mat temp = detection.get_image(sensors);

      frame = temp.clone();
      frame_hsv = temp.clone();
      cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

      detection.vision_process(frame_hsv, frame);

      std::cout << "position x = " << detection.get_ball_pos_x() << std::endl;
      std::cout << "position y = " << detection.get_ball_pos_y() << std::endl;

      // draw red circle on ball
      cv::circle(
        frame, cv::Point(detection.get_ball_pos_x(), detection.get_ball_pos_y()), 12.5, cv::Scalar(
          0, 0,
          255), cv::FILLED,
        cv::LINE_8);
    }

    if (!frame.empty()) {
      cv::imshow("Live", frame);
      if (cv::waitKey(5) >= 0) {
        break;
      }
    }
  }
  return 0;
}