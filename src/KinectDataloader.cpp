//
// Created by Komorowicz David on 2020. 06. 25..
//

#include "bundleadjust/KinectDataloader.h"
#include "VirtualSensor.h"

KinectDataloader::KinectDataloader(const std::string &datasetDir) {
    VirtualSensor sensor;
    sensor.Init(datasetDir);

    while (sensor.ProcessNextFrame()) {
        auto color = sensor.GetColorRGBX();
        auto depth = sensor.GetDepth();

    }
}
