
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "bundleadjust/ProcrustesAligner.h"
#include "bundleadjust/BundleAdjustment.h"
#include "bundleadjust/BalDataloader.h"
#include "bundleadjust/KinectDataloader.h"
#include "bundleadjust/MeshWriter.h"


int main() {

    KinectDataloader* data = new KinectDataloader("/home/andrew/repos/tum/bundle_adjustment/data/rgbd_dataset_freiburg1_xyz/");

    auto solver = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = false;
    options.linear_solver_type = solver;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 600;
    options.num_threads = 4;

    BundleAdjustment ba{data, options};
    ba.WriteToPLYFile("Test.ply");
    ba.createProblem();
    MeshWriter m;

    std::string dir = "init_meshes";

    //data->visualizeMatch(0,1);

    delete data;

    return -1;
}