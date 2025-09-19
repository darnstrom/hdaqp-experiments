#include <lexls/lexlsi.h>

LexLS::internal::LexLSI lexls_from_stack(Eigen::MatrixXd const& matrix,
                                         Eigen::VectorXd const& upper,
                                         Eigen::VectorXd const& lower,
                                         Eigen::VectorXi const& breaks) {
    int n_tasks = breaks.size();
    std::vector<LexLS::Index> number_of_constraints(n_tasks);
    std::vector<LexLS::ObjectiveType> types_of_objectives(n_tasks);
    std::vector<Eigen::MatrixXd> objectives(n_tasks);

    for (int start = 0, k = 0; k < n_tasks; ++k) {
        int n_constraints = breaks(k) - start;
        Eigen::MatrixXd objective(n_constraints, matrix.cols() + 2);
        objective << matrix.middleRows(start, n_constraints), lower.segment(start, n_constraints),
          upper.segment(start, n_constraints);
        number_of_constraints[k] = n_constraints;
        types_of_objectives[k]   = LexLS::ObjectiveType::GENERAL_OBJECTIVE;
        objectives[k]            = objective;
        start                    = breaks(k);
    }

    LexLS::internal::LexLSI lexls(matrix.cols(), n_tasks, &number_of_constraints[0], &types_of_objectives[0]);
    LexLS::ParametersLexLSI parameters;
    // parameters.output_file_name = "lexls_log.txt";
    lexls.setParameters(parameters);

    for (int k = 0; k < n_tasks; ++k) {
        lexls.setData(k, objectives[k]);
    }
    return lexls;
}
