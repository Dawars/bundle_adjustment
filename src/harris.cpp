//
// Created by Komorowicz David on 2020. 06. 08..
//
#include <iostream>

//#include <omp.h>
//#include <opencv2/imgproc.hpp>
//#include <sophus/geometry.hpp>

//#include <ceres/ceres.h>
//#include <ceres/c_api.h>
#include <torch/torch.h>
#include "harris/harris.h"

Harris::Harris(unsigned *image, int width, int height) {
//    cv::Mat M(2,2, CV_8UC3, cv::Scalar(0,0,255));
//
//    Sophus::SO3<float> rotation{};
//
//    auto this_is_a_problem = ceres_create_problem();
//
//    ceres::sin(1);
}

int Harris::getNumCores() {
    // Set floating point output precision
//    std::cout << std::fixed << std::setprecision(4);
//
//
//    // ================================================================ //
//    //                     BASIC AUTOGRAD EXAMPLE 1                     //
//    // ================================================================ //
//    // Device
//    auto cuda_available = torch::cuda::is_available();
//    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//
//    std::cout << device << std::endl;
//
//    std::cout << "---- BASIC AUTOGRAD EXAMPLE 1 ----\n";
//
//    // Create Tensors
//    torch::Tensor x = torch::tensor(1.0, torch::requires_grad()).to(device);
//    torch::Tensor w = torch::tensor(2.0, torch::requires_grad()).to(device);
//    torch::Tensor b = torch::tensor(3.0, torch::requires_grad()).to(device);
//
//    // Build a computational graph
//    auto y = w * x + b;  // y = 2 * x + 3
//    std::cout << y << std::endl;
//
//    // Compute the gradients
//    y.backward();
//    x.retain_grad();
//    std::cout << "x device " << x.device() << std::endl;
//
//    // Print out the gradients
//    std::cout << x.grad() << '\n';  // x.grad() = 2
//    std::cout << w.grad() << '\n';  // w.grad() = 1
//    std::cout << b.grad() << "\n\n";  // b.grad() = 1

//    auto grad = x.grad();
//    std::cout << grad << '\n';  // x.grad() = 2
//
//    auto x_ = grad.to(torch::kCPU);
//    std::cout << x_ << '\n';  // x.grad() = 2

    return 0;//omp_get_num_procs();
}
