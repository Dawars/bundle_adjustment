//
// Created by Komorowicz David on 2020. 06. 15..
//
#include <iostream>

#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BalDataloader.h"
#include "bundleadjust/KinectDataloader.h"


int main(){
    
    
    KinectDataloader* data = new KinectDataloader("/home/andrew/repos/tum/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");



    auto solvers = {ceres::SPARSE_NORMAL_CHOLESKY, ceres::DENSE_SCHUR, ceres::SPARSE_SCHUR, ceres::ITERATIVE_SCHUR, ceres::CGNR};
    for (auto & solver : solvers) {
        ceres::Solver::Options options;

        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.use_nonmonotonic_steps = false;
        options.linear_solver_type = solver;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 600;
        options.num_threads = 4;

        BundleAdjustment ba{data, options};

        ba.createProblem();
        ba.solve();
    }

//
//    ba.writeMesh("veniceGroundTruth.off");
//    ba.writeCamerasMesh("veniceCameraGroundTruth.off");



    delete data;

    return 0;
}