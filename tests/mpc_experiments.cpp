#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <H5Cpp.h>
#include <daqp.hpp>

#include "lexls_interface.hpp"
#include "nipm_interface.hpp"
#include "tyler_interface.hpp"
#include "hierarchy_utils.hpp"

namespace {

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct MPCProblem {
    RowMajorMatrix matrix;
    Eigen::VectorXd upper;
    Eigen::VectorXd lower;
    Eigen::VectorXi breaks;
    double t;
};

std::filesystem::path result_directory() {
    return std::filesystem::path(HDAQP_EXPERIMENTS_SOURCE_DIR) / "mpc-example" / "result";
}

bool is_scenario_file(std::filesystem::path const& path) {
    std::string stem = path.stem().string();
    return path.extension() == ".h5" && stem.rfind("scenario", 0) == 0;
}

std::vector<std::filesystem::path> scenario_files(std::filesystem::path const& dir) {
    if (!std::filesystem::exists(dir)) {
        throw std::runtime_error("Missing MPC result directory " + dir.string() +
                                 ". Generate scenario*.h5 files from mpc-example first.");
    }
    if (!std::filesystem::is_directory(dir)) {
        throw std::runtime_error(dir.string() + " is not a directory");
    }

    std::vector<std::filesystem::path> paths;
    for (auto const& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_scenario_file(entry.path())) {
            paths.push_back(entry.path());
        }
    }

    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("No scenario*.h5 files found in " + dir.string());
    }
    return paths;
}

Eigen::VectorXd read_vector(H5::Group const& group, std::string const& dataset_name) {
    H5::DataSet dataset = group.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1] = {0};
    dataspace.getSimpleExtentDims(dims);

    std::vector<double> buffer(dims[0]);
    dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

    Eigen::Map<Eigen::VectorXd const> mapped(buffer.data(), static_cast<Eigen::Index>(buffer.size()));
    return mapped;
}

Eigen::VectorXi read_int_vector(H5::Group const& group, std::string const& dataset_name) {
    H5::DataSet dataset = group.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1] = {0};
    dataspace.getSimpleExtentDims(dims);

    std::vector<int> buffer(dims[0]);
    dataset.read(buffer.data(), H5::PredType::NATIVE_INT);

    Eigen::Map<Eigen::VectorXi const> mapped(buffer.data(), static_cast<Eigen::Index>(buffer.size()));
    return mapped;
}

RowMajorMatrix read_matrix(H5::Group const& group, std::string const& dataset_name) {
    H5::DataSet dataset = group.openDataSet(dataset_name);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[2] = {0, 0};
    dataspace.getSimpleExtentDims(dims);

    std::vector<double> buffer(dims[0] * dims[1]);
    dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

    Eigen::Map<RowMajorMatrix const> mapped(buffer.data(),
                                            static_cast<Eigen::Index>(dims[0]),
                                            static_cast<Eigen::Index>(dims[1]));
    return mapped;
}

double read_scalar(H5::Group const& group, std::string const& dataset_name) {
    H5::DataSet dataset = group.openDataSet(dataset_name);
    double value = 0.0;
    dataset.read(&value, H5::PredType::NATIVE_DOUBLE);
    return value;
}

int count_problems(H5::H5File const& file) {
    int n = 0;
    for (;; ++n) {
        std::string group_name = "/problem" + std::to_string(n);
        if (H5Lexists(file.getId(), group_name.c_str(), H5P_DEFAULT) <= 0)
            break;
    }
    return n;
}

MPCProblem load_problem(H5::H5File const& file, int problem_idx) {
    std::string group_name = "/problem" + std::to_string(problem_idx);
    H5::Group group = file.openGroup(group_name);
    return MPCProblem{
        read_matrix(group, "matrix"),
        read_vector(group, "upper"),
        read_vector(group, "lower"),
        read_int_vector(group, "break_points"),
        read_scalar(group, "t"),
    };
}

double elapsed_seconds(std::chrono::high_resolution_clock::time_point start,
                       std::chrono::high_resolution_clock::time_point end) {
    auto dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return std::chrono::duration<double>(dt).count();
}

void write_scenario_times(std::filesystem::path const& output_path,
                          std::vector<double> const& t_values,
                          Eigen::MatrixXd const& times,
                          Eigen::VectorXd const& daqp_tyler_diff) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error("Failed to open " + output_path.string());
    }

    output << std::setprecision(17);
    output << "problem t daqp lexls nipm tyler daqp_tyler_diff\n";
    for (Eigen::Index i = 0; i < times.rows(); ++i) {
        output << i << " " << t_values[static_cast<std::size_t>(i)] << " " << times(i, 0) << " "
               << times(i, 1) << " " << times(i, 2) << " " << times(i, 3) << " "
               << daqp_tyler_diff(i) << "\n";
    }
    if (!output) {
        throw std::runtime_error("Failed to write to " + output_path.string());
    }
}

} // namespace

int main() {
    std::filesystem::path const results_dir = result_directory();
    std::vector<std::filesystem::path> const scenarios = scenario_files(results_dir);

    // Run each scenario, skipping ones whose output already exists so the
    // binary can safely be re-launched after an interrupted run.
    for (std::size_t scenario_idx = 0; scenario_idx < scenarios.size(); ++scenario_idx) {
        std::filesystem::path const& scenario_path = scenarios[scenario_idx];
        std::string const scenario_name = scenario_path.stem().string();
        std::filesystem::path const timings_path = results_dir / (scenario_name + "_timings.dat");

        if (std::filesystem::exists(timings_path)) {
            std::cout << "Skipping " << scenario_name << " (output already exists)" << std::endl;
            continue;
        }

        std::cout << "Running " << scenario_name << " (" << (scenario_idx + 1) << "/"
                  << scenarios.size() << ")" << std::flush;

        H5::H5File h5file(scenario_path.string(), H5F_ACC_RDONLY);
        int const n_problems = count_problems(h5file);
        if (n_problems == 0) {
            throw std::runtime_error("No problem groups found in " + scenario_path.string());
        }

        Eigen::MatrixXd times(n_problems, 4);
        Eigen::VectorXd daqp_tyler_diff(n_problems);
        std::vector<double> t_values(static_cast<std::size_t>(n_problems));

        for (int problem_idx = 0; problem_idx < n_problems; ++problem_idx) {
            // Load one problem at a time so peak memory stays bounded.
            MPCProblem const problem = load_problem(h5file, problem_idx);
            t_values[static_cast<std::size_t>(problem_idx)] = problem.t;
            Eigen::VectorXi const normalized = normalized_breaks(problem.breaks, problem.matrix.rows());
            Eigen::VectorXd daqp_primal(problem.matrix.cols());

            {
                DAQP daqp(problem.matrix.cols(),
                          problem.upper.size(),
                          max_constraints_in_level(problem.breaks, problem.matrix.rows()));
                auto t_start = std::chrono::high_resolution_clock::now();
                daqp.solve(problem.matrix, problem.upper, problem.lower, normalized);
                auto t_end = std::chrono::high_resolution_clock::now();
                daqp_primal = daqp.get_primal();
                times(static_cast<Eigen::Index>(problem_idx), 0) = elapsed_seconds(t_start, t_end);
            }

            {
                auto lexls = lexls_from_stack(problem.matrix, problem.upper, problem.lower, problem.breaks);
                auto t_start = std::chrono::high_resolution_clock::now();
                lexls.solve();
                auto t_end = std::chrono::high_resolution_clock::now();
                times(static_cast<Eigen::Index>(problem_idx), 1) = elapsed_seconds(t_start, t_end);
            }

            {
                auto nipm = nipm_from_stack(problem.matrix, problem.upper, problem.lower, problem.breaks);
                auto t_start = std::chrono::high_resolution_clock::now();
                nipm.solve();
                auto t_end = std::chrono::high_resolution_clock::now();
                times(static_cast<Eigen::Index>(problem_idx), 2) = elapsed_seconds(t_start, t_end);
            }

            TylerResult tyler;
            {
                auto t_start = std::chrono::high_resolution_clock::now();
                try {
                    tyler = tyler_from_stack(problem.matrix, problem.upper, problem.lower, problem.breaks);
                } catch (std::exception const& error) {
                    throw std::runtime_error("Tyler failed for " + scenario_name + " problem" +
                                             std::to_string(problem_idx) + ": " + error.what());
                }
                auto t_end = std::chrono::high_resolution_clock::now();
                times(static_cast<Eigen::Index>(problem_idx), 3) = elapsed_seconds(t_start, t_end);
            }

            daqp_tyler_diff(static_cast<Eigen::Index>(problem_idx)) =
                compute_lexdiff(compute_band_slacks(problem.matrix, problem.upper, problem.lower, daqp_primal),
                                compute_band_slacks(problem.matrix, problem.upper, problem.lower, tyler.primal),
                                problem.breaks);
        }

        write_scenario_times(timings_path, t_values, times, daqp_tyler_diff);
        std::cout << " done." << std::endl;
    }

    // Rebuild summary from all per-scenario timing files so it is always
    // consistent regardless of which run produced the individual files.
    std::ofstream summary_file(results_dir / "mpc_timings_summary.dat");
    if (!summary_file) {
        throw std::runtime_error("Failed to open summary output file");
    }
    summary_file << std::setprecision(17);
    summary_file << "scenario daqpmin lexlsmin nipmmin tylermin daqpmean lexlsmean nipmmean tylermean "
                    "daqpmax lexlsmax nipmmax tylermax\n";

    for (auto const& scenario_path : scenarios) {
        std::string const scenario_name = scenario_path.stem().string();
        std::filesystem::path const timings_path = results_dir / (scenario_name + "_timings.dat");

        std::ifstream in(timings_path);
        if (!in) {
            std::cerr << "Warning: missing " << timings_path << ", skipping from summary.\n";
            continue;
        }
        std::string header;
        std::getline(in, header);

        std::vector<double> daqp_times, lexls_times, nipm_times, tyler_times;
        int prob;
        double t, d, l, n, y, diff;
        while (in >> prob >> t >> d >> l >> n >> y >> diff) {
            daqp_times.push_back(d);
            lexls_times.push_back(l);
            nipm_times.push_back(n);
            tyler_times.push_back(y);
        }
        if (in.bad()) {
            std::cerr << "Warning: I/O error reading " << timings_path << ", skipping from summary.\n";
            continue;
        }

        if (daqp_times.empty()) {
            std::cerr << "Warning: no data rows in " << timings_path << ", skipping from summary.\n";
            continue;
        }

        auto stats = [](std::vector<double>& v) -> std::tuple<double, double, double> {
            double mn   = *std::min_element(v.begin(), v.end());
            double mx   = *std::max_element(v.begin(), v.end());
            double mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
            return {mn, mean, mx};
        };

        auto [dmin, dmean, dmax] = stats(daqp_times);
        auto [lmin, lmean, lmax] = stats(lexls_times);
        auto [nmin, nmean, nmax] = stats(nipm_times);
        auto [ymin, ymean, ymax] = stats(tyler_times);

        summary_file << scenario_name
                     << " " << dmin  << " " << lmin  << " " << nmin  << " " << ymin
                     << " " << dmean << " " << lmean << " " << nmean << " " << ymean
                     << " " << dmax  << " " << lmax  << " " << nmax  << " " << ymax
                     << "\n";
    }

    std::cout << "Done. Summary written to mpc_timings_summary.dat" << std::endl;
    return 0;
}
