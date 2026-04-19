#include <Eigen/Dense>
#include <daqp.hpp>

#include "hierarchy_utils.hpp"

// Solve an HQP naively using DAQP:
//   - No warm starting between levels (fresh DAQP solves per level)
//   - Explicit slack variables introduced for each level's constraints
Eigen::VectorXd naive_daqp_from_stack(Eigen::MatrixXd const& matrix,
                                       Eigen::VectorXd const& upper,
                                       Eigen::VectorXd const& lower,
                                       Eigen::VectorXi const& breaks) {
    int n  = matrix.cols();
    Eigen::VectorXi const normalized = normalized_breaks(breaks, matrix.rows());
    int nh = normalized.size() - 1;
    // Small weight on z in phase 1 so the problem is non-degenerate while
    // still driving the solver to minimise the violation first.
    const double eps = 1e-9;

    Eigen::VectorXd upper_mod = upper;
    Eigen::VectorXd lower_mod = lower;
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    Eigen::VectorXi no_breaks(0);

    for (int k = 0; k < nh; k++) {
        int level_start = normalized(k);
        int level_end   = normalized(k + 1);
        int m_k         = level_end - level_start;
        int n_ext       = n + m_k;   // [z; s_k]

        // Extended constraint matrix:
        //   rows 0..level_start-1 : [A_i | 0_{m_i x m_k}]  (previous levels)
        //   rows level_start..end-1: [A_k | I_{m_k}]        (current level)
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            A_ext(level_end, n_ext);
        A_ext.setZero();
        if (level_start > 0)
            A_ext.block(0, 0, level_start, n) = matrix.topRows(level_start);
        A_ext.block(level_start, 0,   m_k, n)   = matrix.middleRows(level_start, m_k);
        A_ext.block(level_start, n,   m_k, m_k) = Eigen::MatrixXd::Identity(m_k, m_k);

        // Previous levels use modified bounds; current level uses original bounds.
        Eigen::VectorXd bu_ext(level_end), bl_ext(level_end);
        bu_ext.head(level_start) = upper_mod.head(level_start);
        bl_ext.head(level_start) = lower_mod.head(level_start);
        bu_ext.tail(m_k) = upper.segment(level_start, m_k);
        bl_ext.tail(m_k) = lower.segment(level_start, m_k);

        // DAQP solves min 0.5 x'Hx + f'x, so set H = diag(2eps,...,2eps,2,...,2)
        // to obtain min eps*||z||^2 + ||s_k||^2.
        Eigen::MatrixXd H1 = Eigen::MatrixXd::Zero(n_ext, n_ext);
        H1.diagonal().head(n).setConstant(2.0 * eps);
        H1.diagonal().tail(m_k).setConstant(2.0);
        Eigen::VectorXd f1 = Eigen::VectorXd::Zero(n_ext);
        Eigen::VectorXi sense(0);

        EigenDAQPResult res1 = daqp_solve(H1, f1, A_ext, bu_ext, bl_ext, sense, no_breaks);
        Eigen::VectorXd s_k  = res1.get_primal().tail(m_k);

        // Shift level-k bounds by s_k so the constraint on z becomes tight.
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            A_p2(level_end, n);
        A_p2.setZero();
        if (level_start > 0)
            A_p2.topRows(level_start) = matrix.topRows(level_start);
        A_p2.bottomRows(m_k) = matrix.middleRows(level_start, m_k);

        Eigen::VectorXd bu_p2(level_end), bl_p2(level_end);
        bu_p2.head(level_start) = upper_mod.head(level_start);
        bl_p2.head(level_start) = lower_mod.head(level_start);
        bu_p2.tail(m_k) = upper.segment(level_start, m_k) - s_k;
        bl_p2.tail(m_k) = lower.segment(level_start, m_k) - s_k;

        EigenDAQPResult res2 = daqp_solve(A_p2, bu_p2, bl_p2, no_breaks);
        z = res2.get_primal();

        // Lock in level k's violation budget for subsequent levels.
        upper_mod.segment(level_start, m_k) -= s_k;
        lower_mod.segment(level_start, m_k) -= s_k;
    }

    return z;
}
