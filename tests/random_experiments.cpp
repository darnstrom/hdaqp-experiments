#include <chrono>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <daqp/daqp.hpp>
#include "lexls_interface.hpp"
#include "nipm_interface.hpp"

struct Polytope {
    Eigen::MatrixXd A;
    Eigen::VectorXd ub;
    Eigen::VectorXd lb;
};

struct Hierarchy{
    Eigen::MatrixXd matrix;
    Eigen::VectorXd upper;
    Eigen::VectorXd lower;
    Eigen::VectorXi breaks;
};

std::vector<Polytope> generate_random_polytopes(int nh, int n, int mmax, double equality_prob) {
    // Seed the random number generator
    std::vector<Polytope> polytopes(nh);

    for (int i = 0; i < nh; ++i) {
        int m = rand() % mmax + 1;
        // Random elements between [0, 1]
        polytopes[i].A = Eigen::MatrixXd::Random(m, n).cwiseAbs();
        // Normalize each row to have a L2 norm of 1
        for (int row = 0; row < m; ++row) {
            polytopes[i].A.row(row) /= polytopes[i].A.row(row).norm();
        }
        // Random lements between [0, 1]
        polytopes[i].ub = Eigen::VectorXd::Random(m, 1).cwiseAbs();

        // Do coordinate change \tilde{z} = z + rand()
        polytopes[i].ub += polytopes[i].A *(Eigen::VectorXd::Random(n, 1)); 

        if ((double)rand() / RAND_MAX < equality_prob)
            polytopes[i].lb = polytopes[i].ub; // Equality constraints
        else // randomly generate lb such that lb < ub            
            polytopes[i].lb = polytopes[i].ub - Eigen::VectorXd::Random(m, 1).cwiseAbs();
    }
    return polytopes;
}

Hierarchy create_random_hierarchy(int nh, int n, int mmax, double equality_prob){
    std::vector<Polytope> polytopes = generate_random_polytopes(nh,n,mmax,equality_prob);
    // Extract the rows
    Eigen::VectorXi rows(nh); 
    for (int i = 0; i <  nh; i++)  rows(i) = polytopes[i].ub.size();

    Hierarchy h;
    h.matrix = Eigen::MatrixXd(rows.sum(), n);
    h.upper = Eigen::VectorXd(rows.sum());
    h.lower = Eigen::VectorXd(rows.sum());
    h.breaks = Eigen::VectorXi(nh+1);

    int break_point = 0;
    h.breaks(0) = 0;
    for (int i = 0; i <  nh; i++){
        h.matrix.block(break_point, 0, rows(i), n) = polytopes[i].A;
        h.upper.block(break_point,0,rows(i),1) = polytopes[i].ub;
        h.lower.block(break_point,0,rows(i),1) = polytopes[i].lb;

        break_point += rows(i);
        h.breaks(i+1) = break_point;
    }
    return h;
}

double compute_lexdiff(Eigen::VectorXd s1, Eigen::VectorXd s2, Eigen::VectorXi breaks, double tol = 1e-9){
    double diff,w1,w2;
    for(int i = 0; i < breaks.size() - 1; i++){
        w1 = s1.segment(breaks(i),breaks(i+1)-breaks(i)).squaredNorm();
        w2 = s2.segment(breaks(i),breaks(i+1)-breaks(i)).squaredNorm();
        //std::cout << w1 << " | " << w2 << std::endl; 
        diff = w1-w2; 
        if(abs(diff) > tol)
            return diff;
    }
    return 0.0;
} 


int main() {

    std::ofstream output_file("ratio.dat");
    output_file << "ratio "; 
    output_file << "daqpmin " << "lexlsmin " << "nipmmin ";
    output_file << "daqpmean " << "lexlsmean " << "nipmmean ";
    output_file << "daqpmax " << "lexlsmax " << "nipmmax";
    output_file << std::endl;

    srand(123);
    int nh = 10;
    int n = 50;
    int mmax = 20;

    Eigen::VectorXd eq_probs = (Eigen::VectorXd(11) <<0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0).finished();

    int Nruns = 100;
    Eigen::MatrixXd times(Nruns,3);
    Eigen::MatrixXd daqp_diff_lexls(Nruns,eq_probs.size());
    Eigen::MatrixXd daqp_diff_nipm(Nruns,eq_probs.size());

    for (int i = 0; i < eq_probs.size(); i++){
        printf("Running %d / %d \n",i+1, static_cast<int>(eq_probs.size())); 
        Eigen::MatrixXd times(Nruns,3);
        double equality_prob = eq_probs(i);
        for (int run = 0; run < Nruns; run++){
            Hierarchy h = create_random_hierarchy(nh, n, mmax, equality_prob);

            // DAQP
            DAQP daqp(n, h.upper.size(), mmax);
            auto t_start = std::chrono::high_resolution_clock::now();
            daqp.solve(h.matrix, h.upper, h.lower, h.breaks);
            auto t_end = std::chrono::high_resolution_clock::now();
            auto daqp_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
            times(run,0) = std::chrono::duration<double>(daqp_time).count();

            // LexLS
            auto lexls = lexls_from_stack(h.matrix, h.upper, h.lower, h.breaks);
            t_start = std::chrono::high_resolution_clock::now();
            auto status = lexls.solve();
            t_end = std::chrono::high_resolution_clock::now();
            auto lexls_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
            times(run,1) = std::chrono::duration<double>(lexls_time).count();

            //// NIPM-HLSP
            auto nipm = nipm_from_stack(h.matrix, h.upper, h.lower, h.breaks);
            t_start = std::chrono::high_resolution_clock::now();
            nipm.solve();
            t_end = std::chrono::high_resolution_clock::now();
            auto nipm_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
            times(run,2) = std::chrono::duration<double>(nipm_time).count();


            // Compute slacks
            Eigen::VectorXd w_daqp = compute_slacks(h.matrix,h.upper,h.lower,daqp.get_primal());
            Eigen::VectorXd w_lexls = compute_slacks(h.matrix,h.upper,h.lower,lexls.get_x());
            Eigen::VectorXd w_nipm = compute_slacks(h.matrix,h.upper,h.lower,nipm.get_x());

            daqp_diff_lexls(run,i) = compute_lexdiff(w_daqp,w_lexls,h.breaks);
            daqp_diff_nipm(run,i) = compute_lexdiff(w_daqp,w_nipm,h.breaks);
        }

        // Append to file
        output_file << equality_prob << " ";
        output_file << times.colwise().minCoeff() << " ";
        output_file << times.colwise().mean() << " ";
        output_file << times.colwise().maxCoeff();
        output_file << std::endl;

    }

    output_file.close();

    std::ofstream output_file_diff_lexls("daqp_diff_lexls.dat");
    output_file_diff_lexls << daqp_diff_lexls;
    output_file_diff_lexls.close();
    std::ofstream output_file_diff_nipm("daqp_diff_nipm.dat");
    output_file_diff_nipm << daqp_diff_nipm;
    output_file_diff_nipm.close();
}
