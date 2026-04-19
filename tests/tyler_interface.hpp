#pragma once

#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include "Highs.h"

#include "hierarchy_utils.hpp"

struct TylerResult {
    Eigen::VectorXd primal;
    int first_failed_level = -1;
    double violation = 0.0;
    bool all_levels_satisfied = false;
};

namespace tyler_detail {

constexpr double kInf = 1.0e30;

struct HighsLpResult {
    bool optimal = false;
    std::string status;
    Eigen::VectorXd primal;
    double objective = 0.0;
};

inline HighsLpResult solve_lp(Eigen::MatrixXd const& A,
                              Eigen::VectorXd const& row_lower,
                              Eigen::VectorXd const& row_upper,
                              Eigen::VectorXd const& col_cost,
                              Eigen::VectorXd const& col_lower,
                              Eigen::VectorXd const& col_upper) {
    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.addCols(static_cast<HighsInt>(A.cols()),
                  col_cost.data(),
                  col_lower.data(),
                  col_upper.data(),
                  0,
                  nullptr,
                  nullptr,
                  nullptr);

    std::vector<HighsInt> indices;
    std::vector<double> values;
    indices.reserve(static_cast<std::size_t>(A.cols()));
    values.reserve(static_cast<std::size_t>(A.cols()));

    for (Eigen::Index row = 0; row < A.rows(); ++row) {
        indices.clear();
        values.clear();
        for (Eigen::Index col = 0; col < A.cols(); ++col) {
            double const value = A(row, col);
            if (value == 0.0) {
                continue;
            }
            indices.push_back(static_cast<HighsInt>(col));
            values.push_back(value);
        }

        highs.addRow(row_lower(row),
                     row_upper(row),
                     static_cast<HighsInt>(indices.size()),
                     indices.empty() ? nullptr : indices.data(),
                     values.empty() ? nullptr : values.data());
    }

    highs.run();

    HighsLpResult result;
    result.status = highs.modelStatusToString(highs.getModelStatus());
    result.optimal = result.status == "Optimal";
    if (!result.optimal) {
        return result;
    }

    HighsSolution const& solution = highs.getSolution();
    result.objective = highs.getInfo().objective_function_value;
    result.primal = Eigen::Map<Eigen::VectorXd const>(solution.col_value.data(),
                                                      static_cast<Eigen::Index>(solution.col_value.size()));
    return result;
}

inline HighsLpResult solve_exact_prefix(Eigen::MatrixXd const& matrix,
                                        Eigen::VectorXd const& upper,
                                        Eigen::VectorXd const& lower,
                                        std::vector<HierarchyLevelRange> const& ranges,
                                        int last_level) {
    if (last_level < 0) {
        HighsLpResult result;
        result.optimal = true;
        result.status = "Optimal";
        result.primal = Eigen::VectorXd::Zero(matrix.cols());
        return result;
    }

    int const prefix_end = ranges[static_cast<std::size_t>(last_level)].end;
    Eigen::MatrixXd A = matrix.topRows(prefix_end);
    Eigen::VectorXd row_upper = upper.head(prefix_end);
    Eigen::VectorXd row_lower = lower.head(prefix_end);
    Eigen::VectorXd col_cost = Eigen::VectorXd::Zero(matrix.cols());
    Eigen::VectorXd col_lower = Eigen::VectorXd::Constant(matrix.cols(), -kInf);
    Eigen::VectorXd col_upper = Eigen::VectorXd::Constant(matrix.cols(), kInf);
    return solve_lp(A, row_lower, row_upper, col_cost, col_lower, col_upper);
}

inline HighsLpResult solve_relaxed_level(Eigen::MatrixXd const& matrix,
                                         Eigen::VectorXd const& upper,
                                         Eigen::VectorXd const& lower,
                                         std::vector<HierarchyLevelRange> const& ranges,
                                         int level) {
    HierarchyLevelRange const range = ranges[static_cast<std::size_t>(level)];
    int const exact_rows = range.start;
    int const relaxed_rows = range.end - range.start;
    int const total_rows = exact_rows + 2 * relaxed_rows;
    int const n = matrix.cols();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(total_rows, n + 1);
    Eigen::VectorXd row_lower = Eigen::VectorXd::Constant(total_rows, -kInf);
    Eigen::VectorXd row_upper = Eigen::VectorXd::Constant(total_rows, kInf);

    if (exact_rows > 0) {
        A.block(0, 0, exact_rows, n) = matrix.topRows(exact_rows);
        row_lower.head(exact_rows) = lower.head(exact_rows);
        row_upper.head(exact_rows) = upper.head(exact_rows);
    }

    for (int i = 0; i < relaxed_rows; ++i) {
        int const src = range.start + i;

        A.block(exact_rows + i, 0, 1, n) = matrix.row(src);
        A(exact_rows + i, n) = -1.0;
        row_upper(exact_rows + i) = upper(src);

        A.block(exact_rows + relaxed_rows + i, 0, 1, n) = matrix.row(src);
        A(exact_rows + relaxed_rows + i, n) = 1.0;
        row_lower(exact_rows + relaxed_rows + i) = lower(src);
    }

    Eigen::VectorXd col_cost = Eigen::VectorXd::Zero(n + 1);
    col_cost(n) = 1.0;

    Eigen::VectorXd col_lower = Eigen::VectorXd::Constant(n + 1, -kInf);
    Eigen::VectorXd col_upper = Eigen::VectorXd::Constant(n + 1, kInf);
    col_lower(n) = 0.0;

    return solve_lp(A, row_lower, row_upper, col_cost, col_lower, col_upper);
}

} // namespace tyler_detail

inline TylerResult tyler_from_stack(Eigen::MatrixXd const& matrix,
                                    Eigen::VectorXd const& upper,
                                    Eigen::VectorXd const& lower,
                                    Eigen::VectorXi const& breaks) {
    auto const ranges = hierarchy_level_ranges(breaks, matrix.rows());
    TylerResult result;
    result.primal = Eigen::VectorXd::Zero(matrix.cols());

    if (ranges.empty()) {
        result.all_levels_satisfied = true;
        return result;
    }

    tyler_detail::HighsLpResult last_feasible;
    bool have_last_feasible = false;

    for (int level = 0; level < static_cast<int>(ranges.size()); ++level) {
        auto prefix = tyler_detail::solve_exact_prefix(matrix, upper, lower, ranges, level);
        if (prefix.optimal) {
            last_feasible = std::move(prefix);
            have_last_feasible = true;
            continue;
        }

        auto relaxed = tyler_detail::solve_relaxed_level(matrix, upper, lower, ranges, level);
        if (!relaxed.optimal) {
            throw std::runtime_error("Tyler/HiGHS solve failed for level " + std::to_string(level) +
                                     " with status: " + relaxed.status);
        }

        result.primal = relaxed.primal.head(matrix.cols());
        result.first_failed_level = level;
        result.violation = relaxed.primal(matrix.cols());
        result.all_levels_satisfied = false;
        return result;
    }

    if (have_last_feasible) {
        result.primal = last_feasible.primal;
    }
    result.all_levels_satisfied = true;
    return result;
}
