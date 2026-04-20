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


    // --- EXAFMM ---

    // Setup cell
    Cell cell;
    cell.X = vec3(0.0, 0.0, 0.0);
    cell.M.resize(NTERM, complex_t(0.0, 0.0));
    
    // Create particle with position and charge
    Body particle;
    particle.X = vec3(0.5, -0.2, 0.8); 
    particle.q = 1.5;

    // Link particle to cell
    cell.body = &particle;
    cell.numBodies = 1;

    // EXAFMM execution
    P2M(&cell);


    // --- WSE3 SIMULATION ---

    // Create buffer (real + complex)
    int wse3_buf_size = ((P + 1) * (P + 2) / 2) * 2;
    std::vector<double> wse3_buf(wse3_buf_size, 0.0);
    
    // Calculate distance vector
    double dx = particle.X[0] - cell.X[0];
    double dy = particle.X[1] - cell.X[1];
    double dz = particle.X[2] - cell.X[2];
    
    // WSE3 execution
    compute_solid_harmonics_wse3(dx, dy, dz, wse3_buf.data(), P);


    // --- RESULTS ---

    std::cout << "P2P Coefficents Comparison (P=" << P << ")\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left << std::setw(20) << "(l, m)" 
              << std::setw(40) << "ExaFMM " 
              << std::setw(40) << "WSE3 " << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(6);
    
    for (int l = 0; l < P; l++) {
        for (int m = 0; m <= l; m++) {
            
            // Same "object", different indexes
            // Calculate both

            int nms = l * (l + 1) / 2 + m; 
            double exa_r = cell.M[nms].real();
            double exa_i = cell.M[nms].imag();

            int wse3_idx = get_idx_wse3(l, m, P);
            double sign_l = (l % 2 == 0) ? 1.0 : -1.0;      

            // ExaFMM usa rotazioni inverse per convenzione
            // Simulate ExaFMM notation in WSE3 results 
            double wse3_r = wse3_buf[wse3_idx] * particle.q * sign_l;
            double wse3_i = wse3_buf[wse3_idx + 1] * particle.q * sign_l;

            std::cout << "l=" << l << ", m=" << m << "\t"
                      << std::showpos << exa_r << " " << exa_i << "i" << "\t\t"
                      << std::showpos << wse3_r << " " << wse3_i << "i\n";
        }
    }

    return 0;
}
