/**
 * \file shape_fcn_1d.h
 *
 * This is a header for the 1-D shape function which makes up an element. 
 * It is templated based on type so that complex can be passed to
 * it.  This is only for linear shape_fcn_1d functions.  Also, only the 
 * 1D shape functions are here.  The tensor product must be used
 * to form the shape functions in higher dimensions.  The code could
 * be extended to higher order by adding higher order bases here.
 *
 * \author James Grisham
 * \date 05/22/2015
 */

#ifndef ONE_D_HEADER
#define ONE_D_HEADER

#include <iostream>
#include <cstdlib>

/**
  The psi function takes an integer index and coordinate in the 
  computational plane as arguments.  It returns the value of the 
  linear, 1-D shape function at the given point.

  @param[in]  i is an integer index which must be 0 or 1 for linear basis.
  @param[in]  xi is the coordinate in the computational plane.
  @returns    The value of the 1D shape function at the given xi value.
 */

template <typename T> T psi(const int i, T xi) {

  // Declaring variables
  T psi_val;

  // Computing shape_fcn_1d
  // psi_i(xi) = < (1 - xi)/2, (1 + xi)/2>
  if (i==0) {
    psi_val = ((T) 1 - xi)/((T) 2);
  }
  else if (i==1) {
    psi_val = ((T) 1 + xi)/((T) 2);
  }
  else {
    std::cerr << "\nERROR in function psi()." << std::endl;
    std::cerr << "Indices permitted for linear shape functions are 0 or 1." << std::endl;
    std::cerr << "i = " << i << std::endl;
    exit(-1);
  }
  
  return psi_val;
}

/**
  The dpsi function takes an integer index and coordinate in the 
  computational plane as arguments.  It returns the value of the 
  derivative of the linear, 1-D shape function at the given point.

  @param[in]  i is an integer index which must be 0 or 1 for linear basis.
  @param[in]  xi is the coordinate in the computational plane.
  @returns    The value of the derivative of the 1D shape function at the given xi value.
 */

template <typename T> T dpsi(const int i, T xi) {
  
  // Declaring variables
  T dpsi_val;

  // Computing the derivative of the shape_fcn_1d
  if (i==0) {
    dpsi_val = -((T) 1)/((T) 2);
  }
  else if (i==1) {
    dpsi_val = ((T) 1)/((T) 2);
  }
  else {
    std::cerr << "\nERROR in function dpsi()." << std::endl;
    std::cerr << "Indices permitted for linear shape functions are 0 or 1." << std::endl;
    std::cerr << "i = " << i << std::endl;
    exit(-1);
  }

  return dpsi_val;
}

#endif
