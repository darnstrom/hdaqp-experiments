#include <nipm_hlsp/nipm_hlsp.h>

nipmhlsp::NIpmHLSP nipm_from_stack(Eigen::MatrixXd const &matrix,
                                   Eigen::VectorXd const &upper,
                                   Eigen::VectorXd const &lower,
                                   Eigen::VectorXi const &breaks) {
    int n_tasks = breaks.size();
    int n_variables = matrix.cols();
    nipmhlsp::NIpmHLSP solver{n_tasks, n_variables};

    for (int start = 0, k = 0; k < n_tasks; ++k) {
        int n_constraints = breaks(k) - start;
        
        // Check equality row by row using Eigen array
        Eigen::Array<bool, Eigen::Dynamic, 1> is_equality(n_constraints);
        for (int i = 0; i < n_constraints; ++i) {
            is_equality(i) = std::abs(lower(start + i) - upper(start + i)) < 1e-10;
        }

        // Count equality and inequality constraints
        int n_equalities = is_equality.count();
        int n_inequalities = n_constraints - n_equalities;

        // Prepare matrices for equality constraints
        Eigen::MatrixXd A_eq(n_equalities, n_variables);
        Eigen::VectorXd b_eq(n_equalities);
        
        // Prepare matrices for inequality constraints (double-sided become single-sided)
        Eigen::MatrixXd A_ineq(2 * n_inequalities, n_variables);
        Eigen::VectorXd b_ineq(2 * n_inequalities);

        // Fill equality and inequality matrices row by row
        int eq_idx = 0, ineq_idx = 0;
        for (int i = 0; i < n_constraints; ++i) {
            if (is_equality(i)) {
                // Equality constraint: A_eq * x = b_eq
                A_eq.row(eq_idx) = matrix.row(start + i);
                b_eq(eq_idx) = upper(start + i);  // Since lower ≈ upper for equality
                eq_idx++;
            } else {
                // Inequality constraint: lower <= A * x <= upper
                // Convert to: A * x >= lower  and  -A * x >= -upper
                A_ineq.row(ineq_idx) = matrix.row(start + i);
                b_ineq(ineq_idx) = lower(start + i);
                A_ineq.row(ineq_idx + n_inequalities) = -matrix.row(start + i);
                b_ineq(ineq_idx + n_inequalities) = -upper(start + i);
                ineq_idx++;
            }
        }

        // Set data with separated equality and inequality constraints
        solver.setData(k, A_eq, b_eq, A_ineq, b_ineq);
        
        start = breaks(k);
    }

    return solver;
}
