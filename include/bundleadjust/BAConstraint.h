//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

#include "BalDataloader.h"

/**
 * Optimization constraints.
 */
class BAConstraint {

public:
    BAConstraint(const Eigen::Vector3f& observation);

    template <typename T>
    bool operator()(const T* const point, const T* const rot, const T* const tr, const T* const intrinsics, T* residuals) const;

    static ceres::CostFunction* create(const Eigen::Vector3f& observation);

protected:
    const Eigen::Vector3f observation;
};


