#include <iostream>
#include <iomanip>
#include <vector>

#include "include/laplace.h" 
#include "include/wse3.h"

using namespace exafmm;

namespace exafmm {
    int P;
    int NTERM;
    int NCRIT;
    int IMAGES;
    int IX[3];
    real_t CYCLE;
    real_t THETA;
    real_t R0;
    vec3 X0;
    complex_t WAVEK;
}

int main() {

    P = 8; 
    initKernel(); 
    NTERM = (P + 1) * (P + 2) / 2;

    // --- EXAFMM MEMORY ---

    // Setup cells (father and child)
    Cell parent_cell;
    parent_cell.X = vec3(0.0, 0.0, 0.0);
    parent_cell.M.resize(NTERM, complex_t(0.0, 0.0));
    
    Cell child_cell;
    child_cell.X = vec3(0.5, -0.2, 0.8); 
    child_cell.M.resize(NTERM, complex_t(0.0, 0.0));

    // Link child to father cell
    parent_cell.child = &child_cell;
    parent_cell.numChilds = 1;


    // --- WSE3 MEMORY ---
    
    int buf_size = ((P + 1) * (P + 2) / 2) * 2;
    std::vector<double> wse3_father(buf_size, 0.0);
    std::vector<double> wse3_child(buf_size, 0.0);
    std::vector<double> wse3_V(buf_size, 0.0);

    // Precomputed tables generation
    M2MTables tables(P);


    // --- DATA GENERATION ---

    for (int l = 0; l <= P; l++) {
        for (int m = 0; m <= l; m++) {

            // Random complex data generation
            double dummy_r = (l + 1.0) * 1.1;
            double dummy_i = (m + 1.0) * 0.5;

            // Write in WSE3 buffers
            int wse3_idx = get_idx_wse3(l, m, P);
            wse3_child[wse3_idx] = dummy_r;
            wse3_child[wse3_idx + 1] = dummy_i;

            // Write in EXAFMM structs
            int nms = l * (l + 1) / 2 + m; 
            double sign_l = (l % 2 == 0) ? 1.0 : -1.0;
            child_cell.M[nms] = complex_t(dummy_r * sign_l, -dummy_i * sign_l);
        }
    }


    // --- EXAFMM EXECUTION ---
    
    M2M(&parent_cell);

   
    // --- WSE3 EXECUTION ---

    // Calculate distance vector
    double dx_wse3 = child_cell.X[0] - parent_cell.X[0];
    double dy_wse3 = child_cell.X[1] - parent_cell.X[1];
    double dz_wse3 = child_cell.X[2] - parent_cell.X[2];

    compute_solid_harmonics_wse3(dx_wse3, dy_wse3, dz_wse3, wse3_V.data(), P);
    execute_m2m_wse3(wse3_father.data(), wse3_child.data(), wse3_V.data(), P, tables);

    
    // --- RESULTS ---

    std::cout << "M2M Translation Comparison (P=" << P << ")\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left << std::setw(20) << "(l, m)" 
              << std::setw(40) << "ExaFMM" 
              << std::setw(40) << "WSE3" << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(6);
    
    for (int l = 0; l < P; l++) {
        for (int m = 0; m <= l; m++) {
            
            // Same "object", different indexes
            // Calculate both

            int nms = l * (l + 1) / 2 + m; 
            double exa_r = parent_cell.M[nms].real();
            double exa_i = parent_cell.M[nms].imag();

            int wse3_idx = get_idx_wse3(l, m, P);
            double sign_l = (l % 2 == 0) ? 1.0 : -1.0;

            // ExaFMM usa rotazioni inverse per convenzione
            // Simulate ExaFMM notation in WSE3 results 
            double wse3_r = wse3_father[wse3_idx] * sign_l;
            double wse3_i = -wse3_father[wse3_idx + 1] * sign_l;

            std::cout << "l=" << l << ", m=" << m << "\t"
                      << std::showpos << exa_r << " " << exa_i << "i" << "\t\t"
                      << std::showpos << wse3_r << " " << wse3_i << "i\n";
        }
    }

    return 0;
}
