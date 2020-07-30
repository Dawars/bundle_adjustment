//
// Created by Komorowicz David on 2020. 07. 25..
//
#pragma once

#include <ceres/ceres.h>
#include <fmt/core.h>

#include <bundleadjust/BundleAdjustment.h>

class MeshWriterCallback : public ceres::IterationCallback {
public:
    explicit MeshWriterCallback(BundleAdjustment* ba, std::string prefix):ba(ba), prefix(prefix){}

    ~MeshWriterCallback() {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary &summary) {
        if(summary.iteration % 1 == 0) {
            ba->WriteToPLYFile(fmt::format("mesh_{}_{}.ply", prefix, summary.iteration));
//            fmt::print("Saving iteration {}\n", summary.iteration);
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    BundleAdjustment* ba;
    std::string prefix;
};