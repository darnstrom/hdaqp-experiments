#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

struct HierarchyLevelRange {
    int start;
    int end;
};

inline Eigen::VectorXi normalized_breaks(Eigen::VectorXi const& breaks, Eigen::Index total_rows) {
    if (breaks.size() == 0) {
        if (total_rows != 0) {
            throw std::invalid_argument("Non-empty hierarchy must provide break points");
        }
        return Eigen::VectorXi::Zero(1);
    }

    Eigen::VectorXi normalized;
    if (breaks(0) == 0) {
        normalized = breaks;
    } else {
        normalized.resize(breaks.size() + 1);
        normalized(0) = 0;
        normalized.tail(breaks.size()) = breaks;
    }

    for (int i = 1; i < normalized.size(); ++i) {
        if (normalized(i) < normalized(i - 1)) {
            throw std::invalid_argument("Break points must be nondecreasing");
        }
    }
    if (normalized(normalized.size() - 1) != total_rows) {
        throw std::invalid_argument("Final break point must equal the number of constraint rows");
    }
    return normalized;
}

inline std::vector<HierarchyLevelRange> hierarchy_level_ranges(Eigen::VectorXi const& breaks,
                                                               Eigen::Index total_rows) {
    Eigen::VectorXi const normalized = normalized_breaks(breaks, total_rows);
    std::vector<HierarchyLevelRange> ranges;
    ranges.reserve(normalized.size() > 0 ? static_cast<std::size_t>(normalized.size() - 1) : 0U);

    for (int k = 0; k + 1 < normalized.size(); ++k) {
        if (normalized(k + 1) == normalized(k)) {
            continue;
        }
        ranges.push_back(HierarchyLevelRange{normalized(k), normalized(k + 1)});
    }
    return ranges;
}

inline Eigen::VectorXi task_endpoints(Eigen::VectorXi const& breaks, Eigen::Index total_rows) {
    auto const ranges = hierarchy_level_ranges(breaks, total_rows);
    Eigen::VectorXi endpoints(static_cast<Eigen::Index>(ranges.size()));
    for (Eigen::Index k = 0; k < endpoints.size(); ++k) {
        endpoints(k) = ranges[static_cast<std::size_t>(k)].end;
    }
    return endpoints;
}

inline int max_constraints_in_level(Eigen::VectorXi const& breaks, Eigen::Index total_rows) {
    auto const ranges = hierarchy_level_ranges(breaks, total_rows);
    int max_constraints = 0;
    for (auto const& range : ranges) {
        max_constraints = std::max(max_constraints, range.end - range.start);
    }
    return max_constraints;
}

inline Eigen::VectorXd compute_band_slacks(Eigen::MatrixXd const& matrix,
                                           Eigen::VectorXd const& upper,
                                           Eigen::VectorXd const& lower,
                                           Eigen::VectorXd const& primal) {
    Eigen::VectorXd const values = matrix * primal;
    Eigen::VectorXd slacks(values.size());
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        slacks(i) = std::max({0.0, values(i) - upper(i), lower(i) - values(i)});
    }
    return slacks;
}

inline double compute_lexdiff(Eigen::VectorXd const& s1,
                              Eigen::VectorXd const& s2,
                              Eigen::VectorXi const& breaks,
                              double tol = 1e-9) {
    auto const ranges = hierarchy_level_ranges(breaks, s1.size());
    for (auto const& range : ranges) {
        double const w1 = s1.segment(range.start, range.end - range.start).squaredNorm();
        double const w2 = s2.segment(range.start, range.end - range.start).squaredNorm();
        double const diff = w1 - w2;
        if (std::abs(diff) > tol) {
            return diff;
        }
    }
    return 0.0;
}
