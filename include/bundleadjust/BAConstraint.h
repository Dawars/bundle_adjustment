//
// Created by Komorowicz David on 2020. 06. 20..
//

#pragma once

#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <Eigen/src/Core/Matrix.h>

#include "BalDataloader.h"

/**
 * Optimization constraints.
 */
class BAConstraint {

public:
    BAConstraint(const Eigen::Vector3f& observation, const Camera& camera);

    template <typename T>
    bool operator()(const T* const point, const T* const rot, const T* const tr, T* residuals) const;

    static ceres::CostFunction* create(const Eigen::Vector3f& observation, const Camera& camera);

protected:
    const Camera& camera; // use only intrinsics
    const Eigen::Vector3f observation;
};


