#ifndef exafmm_h
#define exafmm_h
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <vector>

namespace exafmm {

  //! Basic type definitions
  typedef double real_t;                        //!< Floating point type is double precision
  typedef std::complex<real_t> complex_t;       //!< Complex type
  const complex_t I(0.,1.);                     //!< Imaginary unit

  //! Custom lightweight vector class to replace vec.h
  struct vec3 {
    real_t d[3];
    
    // Costruttori
    vec3() { d[0]=0; d[1]=0; d[2]=0; }
    vec3(real_t a) { d[0]=a; d[1]=a; d[2]=a; }  //!< Permette operazioni come "vec3 v = 0;"
    vec3(real_t x, real_t y, real_t z) { d[0]=x; d[1]=y; d[2]=z; }
    
    // Accesso agli elementi
    real_t& operator[](int i) { return d[i]; }
    const real_t& operator[](int i) const { return d[i]; }
    
    // Operatori matematici necessari in laplace.h
    vec3 operator-(const vec3& b) const {
      return vec3(d[0]-b.d[0], d[1]-b.d[1], d[2]-b.d[2]);
    }
    vec3 operator*(real_t b) const {
      return vec3(d[0]*b, d[1]*b, d[2]*b);
    }
    vec3& operator+=(const vec3& b) {
      d[0]+=b.d[0]; d[1]+=b.d[1]; d[2]+=b.d[2];
      return *this;
    }
  };

  //! Funzione norm necessaria per cart2sph
  inline real_t norm(const vec3& a) {
      return a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
  }

  typedef vec3 ivec3;                           //!< Vector of 3 int types (fallback)

  //! Structure of bodies
  struct Body {
    vec3 X;                                     //!< Position
    real_t q;                                   //!< Charge
    real_t p;                                   //!< Potential
    vec3 F;                                     //!< Force
    uint64_t key;                               //!< Hilbert key
  };
  typedef std::vector<Body> Bodies;             //!< Vector of bodies

  //! Base components of cells
  struct CellBase {
    int numChilds;                              //!< Number of child cells
    int numBodies;                              //!< Number of descendant bodies
    vec3 X;                                     //!< Cell center
    real_t R;                                   //!< Cell radius
    uint64_t key;                               //!< Hilbert key
  };

  //! Structure of cells
  struct Cell : public CellBase {
    Cell * child;                               //!< Pointer of first child cell
    Body * body;                                //!< Pointer of first body
    std::vector<complex_t> M;                   //!< Multipole expansion coefs
    std::vector<complex_t> L;                   //!< Local expansion coefs
    using CellBase::operator=;                  //!< Substitution to derived class
  };
  typedef std::vector<Cell> Cells;              //!< Vector of cells

  //! Global variables
  extern int P;                                 //!< Order of expansions
  extern int NTERM;                             //!< Number of coefficients
  extern int NCRIT;                             //!< Number of bodies per leaf cell
  extern int IMAGES;                            //!< Number of periodic image sublevels
  extern int IX[3];                             //!< 3-D periodic index
  extern real_t CYCLE;                          //!< Cycle of periodic boundary condition
  extern real_t THETA;                          //!< Multipole acceptance criterion
  extern real_t R0;                             //!< Radius of the bounding box
  extern vec3 X0;                               //!< Center of the bounding box
  extern complex_t WAVEK;                       //!< Wave number

} // chiusura namespace exafmm
#endif
