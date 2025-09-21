#include <chrono>
#include <iostream>

#include <Eigen/Dense>
#include <daqp/daqp.hpp>
#include "lexls_interface.hpp"
#include "nipm_interface.hpp"

int main() {
    // Task 0: -1 <= x <= 1
    Eigen::MatrixXd matrix0 = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd upper0 = Eigen::VectorXd::Ones(3);
    Eigen::VectorXd lower0 = -Eigen::VectorXd::Ones(3);

    // Task 1: x1+x2+x3 <= 1
    Eigen::MatrixXd matrix1 = (Eigen::MatrixXd(1, 3) << 1, 1, 1).finished();
    Eigen::VectorXd upper1 = Eigen::VectorXd::Ones(1);
    Eigen::VectorXd lower1 = Eigen::VectorXd::Constant(1, -1e9);

    // Task 2: x1 - x2 == 0.5
    Eigen::MatrixXd matrix2 = (Eigen::MatrixXd(1, 3) << 1, -1, 0).finished();
    Eigen::VectorXd upper2 = 0.5 * Eigen::VectorXd::Ones(1);
    Eigen::VectorXd lower2 = 0.5 * Eigen::VectorXd::Ones(1);

    // Task 3: 10 <= 3*x1+x2-x3 <= 20
    Eigen::MatrixXd matrix3 = (Eigen::MatrixXd(1, 3) << 3, 1, -1).finished();
    Eigen::VectorXd upper3 = 20 * Eigen::VectorXd::Ones(1);
    Eigen::VectorXd lower3 = 10 * Eigen::VectorXd::Ones(1);

    // Stack the tasks
    Eigen::MatrixXd matrix = (Eigen::MatrixXd(6, 3) << matrix0, matrix1, matrix2, matrix3).finished();
    Eigen::VectorXd upper = (Eigen::VectorXd(6) << upper0, upper1, upper2, upper3).finished();
    Eigen::VectorXd lower = (Eigen::VectorXd(6) << lower0, lower1, lower2, lower3).finished();
    Eigen::VectorXi breaks = (Eigen::VectorXi(4) << 3, 4, 5, 6).finished();

    // DAQP
    DAQP daqp(3, 50, 5);
    auto t_start = std::chrono::high_resolution_clock::now();
    daqp.solve(matrix, upper, lower, (Eigen::VectorXi(1 + breaks.size()) << 0, breaks).finished());
    auto solution = daqp.get_primal();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto daqp_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::cout << "Solution DAQP: " << solution.transpose() << std::endl;
    std::cout << "DAQP execution time: " << daqp_time.count() << " μs" << std::endl;

    // LexLS
    auto lexls = lexls_from_stack(matrix, upper, lower, breaks);
    t_start = std::chrono::high_resolution_clock::now();
    auto status = lexls.solve();
    solution = lexls.get_x();
    t_end = std::chrono::high_resolution_clock::now();
    auto lexls_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::cout << "Solution LexLS: " << solution.transpose() << std::endl;
    std::cout << "LexLS execution time: " << lexls_time.count() << " μs" << std::endl;

    // NIPM-HLSP
    auto nipm = nipm_from_stack(matrix, upper, lower, breaks);
    t_start = std::chrono::high_resolution_clock::now();
    nipm.solve();
    solution = nipm.get_x();
    t_end = std::chrono::high_resolution_clock::now();
    auto nipm_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::cout << "Solution NIPM-HLSP: " << solution.transpose() << std::endl;
    std::cout << "NIPM-HLSP execution time: " << nipm_time.count() << " μs" << std::endl;

    double precision = 1e-5;
    return daqp.get_primal().isApprox(lexls.get_x(), precision) &&
                   daqp.get_primal().isApprox(nipm.get_x(), precision)
               ? 0
               : 1;
}
