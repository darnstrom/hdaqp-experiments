#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Highs.h>

#include "hierarchy_utils.hpp"

struct TylerResult {
    Eigen::VectorXd primal;
    Eigen::VectorXd mip_primal;
    int first_failed_level = -1;
    double violation = 0.0;
    bool all_levels_satisfied = false;
};

namespace tyler_detail {

constexpr double kPracticalInfinity = 1e20;

inline void require_highs_ok(HighsStatus status, std::string const& context) {
    if (status != HighsStatus::kOk) {
        throw std::runtime_error("HiGHS error while " + context);
    }
}

inline void require_model_solution(Highs const& highs, std::string const& context) {
    HighsModelStatus const model_status = highs.getModelStatus();
    if (model_status != HighsModelStatus::kOptimal &&
        model_status != HighsModelStatus::kObjectiveBound &&
        model_status != HighsModelStatus::kObjectiveTarget) {
        throw std::runtime_error("HiGHS failed while " + context +
                                 " (model status " +
                                 std::to_string(static_cast<int>(model_status)) + ")");
    }
}

inline double max_abs(Eigen::VectorXd const& vector) {
    return vector.size() == 0 ? 0.0 : vector.cwiseAbs().maxCoeff();
}

inline bool has_finite_upper(double value) {
    return std::isfinite(value) && value < kPracticalInfinity;
}

inline bool has_finite_lower(double value) {
    return std::isfinite(value) && value > -kPracticalInfinity;
}

inline bool has_usable_upper_rhs(double value) {
    return std::isfinite(value) && value < kPracticalInfinity;
}

inline bool has_usable_lower_rhs(double value) {
    return std::isfinite(value) && value < kPracticalInfinity;
}

inline double bounded_max_abs(Eigen::VectorXd const& vector, bool upper_like) {
    double max_value = 0.0;
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        double const value = vector(i);
        bool const usable = upper_like ? has_finite_upper(value) : has_finite_lower(value);
        if (usable) {
            max_value = std::max(max_value, std::abs(value));
        }
    }
    return max_value;
}

inline double tyler_variable_bound(Eigen::MatrixXd const& matrix,
                                   Eigen::VectorXd const& upper,
                                   Eigen::VectorXd const& lower) {
    double const bound_scale =
        std::max({1.0, bounded_max_abs(upper, true), bounded_max_abs(lower, false)});
    double const dimension_scale = std::sqrt(static_cast<double>(std::max<Eigen::Index>(1, matrix.cols())));
    return std::max(10.0, 10.0 * bound_scale * dimension_scale + 1.0);
}

inline std::vector<double> row_relaxation_bounds(Eigen::MatrixXd const& matrix,
                                                 Eigen::VectorXd const& upper,
                                                 Eigen::VectorXd const& lower,
                                                 double variable_bound) {
    std::vector<double> row_bounds(static_cast<std::size_t>(matrix.rows()), 0.0);
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        double const activity_bound = matrix.row(row).cwiseAbs().sum() * variable_bound;
        double const upper_violation =
            has_finite_upper(upper(row)) ? std::max(0.0, activity_bound - upper(row)) : 0.0;
        double const lower_violation =
            has_finite_lower(lower(row)) ? std::max(0.0, activity_bound + lower(row)) : 0.0;
        row_bounds[static_cast<std::size_t>(row)] = std::max(upper_violation, lower_violation);
    }
    return row_bounds;
}

inline std::vector<double> level_relaxation_bounds(std::vector<HierarchyLevelRange> const& ranges,
                                                   std::vector<double> const& row_bounds) {
    std::vector<double> level_bounds(ranges.size(), 0.0);
    for (std::size_t level = 0; level < ranges.size(); ++level) {
        double bound = 0.0;
        for (int row = ranges[level].start; row < ranges[level].end; ++row) {
            bound = std::max(bound, row_bounds[static_cast<std::size_t>(row)]);
        }
        level_bounds[level] = bound;
    }
    return level_bounds;
}

}  // namespace tyler_detail

inline TylerResult tyler_from_stack(Eigen::MatrixXd const& matrix,
                                    Eigen::VectorXd const& upper,
                                    Eigen::VectorXd const& lower,
                                    Eigen::VectorXi const& breaks) {
    TylerResult result;
    result.primal = Eigen::VectorXd::Zero(matrix.cols());
    result.mip_primal = Eigen::VectorXd::Zero(matrix.cols());

    auto const ranges = hierarchy_level_ranges(breaks, matrix.rows());
    if (ranges.empty()) {
        result.all_levels_satisfied = true;
        return result;
    }

    double const x_bound = tyler_detail::tyler_variable_bound(matrix, upper, lower);
    std::vector<double> const row_bounds = tyler_detail::row_relaxation_bounds(matrix, upper, lower, x_bound);
    std::vector<double> const level_bounds = tyler_detail::level_relaxation_bounds(ranges, row_bounds);
    double const phi_upper =
        std::max(1.0, *std::max_element(level_bounds.begin(), level_bounds.end()));
    double const objective_weight = phi_upper + 1.0;

    Highs highs;
    tyler_detail::require_highs_ok(highs.setOptionValue("output_flag", false), "disabling output");
    tyler_detail::require_highs_ok(highs.setOptionValue("log_to_console", false), "disabling console log");
    tyler_detail::require_highs_ok(highs.setOptionValue("mip_rel_gap", 0.0), "setting mip_rel_gap");
    tyler_detail::require_highs_ok(highs.setOptionValue("mip_abs_gap", 0.0), "setting mip_abs_gap");

    int const n = static_cast<int>(matrix.cols());
    int const levels = static_cast<int>(ranges.size());

    std::vector<int> x_index(static_cast<std::size_t>(n));
    std::vector<int> l_index(static_cast<std::size_t>(levels));
    for (int col = 0; col < n; ++col) {
        x_index[static_cast<std::size_t>(col)] = static_cast<int>(highs.getNumCol());
        tyler_detail::require_highs_ok(highs.addCol(0.0, -x_bound, x_bound, 0, nullptr, nullptr),
                                       "adding primal variables");
    }
    for (int level = 0; level < levels; ++level) {
        l_index[static_cast<std::size_t>(level)] = static_cast<int>(highs.getNumCol());
        tyler_detail::require_highs_ok(highs.addCol(0.0, 0.0, 1.0, 0, nullptr, nullptr),
                                       "adding level indicators");
        tyler_detail::require_highs_ok(
            highs.changeColIntegrality(l_index[static_cast<std::size_t>(level)], HighsVarType::kInteger),
            "setting level indicators to binary");
    }
    int const phi_index = static_cast<int>(highs.getNumCol());
    tyler_detail::require_highs_ok(highs.addCol(0.0, 0.0, phi_upper, 0, nullptr, nullptr),
                                   "adding phi variable");

    for (int level = 0; level + 1 < levels; ++level) {
        std::vector<HighsInt> indices = {
            l_index[static_cast<std::size_t>(level)],
            l_index[static_cast<std::size_t>(level + 1)],
        };
        std::vector<double> values = {-1.0, 1.0};
        tyler_detail::require_highs_ok(
            highs.addRow(-kHighsInf, 0.0, static_cast<HighsInt>(indices.size()), indices.data(), values.data()),
            "adding precedence constraints");
    }

    for (std::size_t level = 0; level < ranges.size(); ++level) {
        HierarchyLevelRange const& range = ranges[level];
        double const level_bound = level_bounds[level];
        for (int row = range.start; row < range.end; ++row) {
            Eigen::VectorXd const coeffs = matrix.row(row).transpose();
            bool const has_upper = tyler_detail::has_finite_upper(upper(row));
            bool const has_lower = tyler_detail::has_finite_lower(lower(row));
            double const satisfaction_upper_rhs = upper(row) + level_bound;
            double const satisfaction_lower_rhs = -lower(row) + level_bound;
            double const failed_upper_rhs = upper(row) + level_bound * static_cast<double>(level);
            double const failed_lower_rhs = -lower(row) + level_bound * static_cast<double>(level);

            std::vector<HighsInt> sat_indices(static_cast<std::size_t>(n + 1));
            std::vector<double> sat_values(static_cast<std::size_t>(n + 1));
            for (int col = 0; col < n; ++col) {
                sat_indices[static_cast<std::size_t>(col)] = x_index[static_cast<std::size_t>(col)];
                sat_values[static_cast<std::size_t>(col)] = coeffs(col);
            }
            sat_indices[static_cast<std::size_t>(n)] = l_index[level];
            sat_values[static_cast<std::size_t>(n)] = level_bound;
            if (has_upper && tyler_detail::has_usable_upper_rhs(satisfaction_upper_rhs)) {
                HighsStatus const status =
                    highs.addRow(-kHighsInf, satisfaction_upper_rhs,
                                 static_cast<HighsInt>(sat_indices.size()), sat_indices.data(), sat_values.data());
                if (status != HighsStatus::kOk) {
                    throw std::runtime_error("HiGHS error while adding satisfaction upper constraints"
                                             " at level " + std::to_string(level) +
                                             ", row " + std::to_string(row) +
                                             ", upper " + std::to_string(upper(row)) +
                                             ", rhs " + std::to_string(satisfaction_upper_rhs) +
                                             ", level_bound " + std::to_string(level_bound));
                }
            }
            if (has_lower && tyler_detail::has_usable_lower_rhs(satisfaction_lower_rhs)) {
                for (int col = 0; col < n; ++col) {
                    sat_values[static_cast<std::size_t>(col)] = -coeffs(col);
                }
                tyler_detail::require_highs_ok(
                    highs.addRow(-kHighsInf, satisfaction_lower_rhs,
                                 static_cast<HighsInt>(sat_indices.size()), sat_indices.data(), sat_values.data()),
                    "adding satisfaction lower constraints");
            }

            std::vector<HighsInt> phi_indices;
            std::vector<double> phi_values;
            phi_indices.reserve(static_cast<std::size_t>(n + 2 + level));
            phi_values.reserve(static_cast<std::size_t>(n + 2 + level));
            for (int col = 0; col < n; ++col) {
                phi_indices.push_back(x_index[static_cast<std::size_t>(col)]);
                phi_values.push_back(coeffs(col));
            }
            phi_indices.push_back(phi_index);
            phi_values.push_back(-1.0);
            phi_indices.push_back(l_index[level]);
            phi_values.push_back(-level_bound);
            for (std::size_t prev = 0; prev < level; ++prev) {
                phi_indices.push_back(l_index[prev]);
                phi_values.push_back(level_bound);
            }
            if (has_upper && tyler_detail::has_usable_upper_rhs(failed_upper_rhs)) {
                tyler_detail::require_highs_ok(
                    highs.addRow(-kHighsInf, failed_upper_rhs,
                                 static_cast<HighsInt>(phi_indices.size()), phi_indices.data(), phi_values.data()),
                    "adding first-failed upper constraints");
            }
            if (has_lower && tyler_detail::has_usable_lower_rhs(failed_lower_rhs)) {
                for (int col = 0; col < n; ++col) {
                    phi_values[static_cast<std::size_t>(col)] = -coeffs(col);
                }
                tyler_detail::require_highs_ok(
                    highs.addRow(-kHighsInf, failed_lower_rhs,
                                 static_cast<HighsInt>(phi_indices.size()), phi_indices.data(), phi_values.data()),
                    "adding first-failed lower constraints");
            }
        }
    }

    for (int level = 0; level < levels; ++level) {
        tyler_detail::require_highs_ok(
            highs.changeColCost(l_index[static_cast<std::size_t>(level)], -objective_weight),
            "setting Tyler objective weights");
    }
    tyler_detail::require_highs_ok(highs.changeColCost(phi_index, 1.0), "setting phi objective weight");

    tyler_detail::require_highs_ok(highs.run(), "solving Tyler MILP");
    tyler_detail::require_model_solution(highs, "solving Tyler MILP");
    HighsSolution const& solution = highs.getSolution();

    int const satisfied_levels = static_cast<int>(
        std::llround(std::accumulate(solution.col_value.begin() + n,
                                     solution.col_value.begin() + n + levels,
                                     0.0)));
    result.all_levels_satisfied = satisfied_levels == levels;
    result.violation = solution.col_value[static_cast<std::size_t>(phi_index)];
    result.first_failed_level = result.all_levels_satisfied ? -1 : satisfied_levels;
    for (int col = 0; col < n; ++col) {
        result.mip_primal(col) =
            solution.col_value[static_cast<std::size_t>(x_index[static_cast<std::size_t>(col)])];
    }
    result.primal = result.mip_primal;
    return result;
}
