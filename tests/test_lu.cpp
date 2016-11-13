#include <iostream>
#include "linalg.h"

int main() {

  // Declaring variables
  arma::Mat<double> A(5,5,arma::fill::randu);
  arma::Col<double> x(5,arma::fill::randu),b(5);
  arma::Mat<double> L,U,P;
  arma::Col<double> y,x_computed;
  
  // Finding rhs
  b = A*x;

  // Finding LU decomposition
  arma::lu(L,U,P,A);
  P.print("\nP=");
  L.print("\nL=");
  std::cout << "\n\nP^T * L = \n" << P.t()*L << std::endl;

  // Doing forward and back solve to recompute x
  y = forward_solve<double>(L,b);
  x_computed = backward_solve<double>(U,y);
  
  // Printing differences
  x.print("\nx = ");
  x_computed.print("\nx_computed = ");
  std::cout.setf(std::ios_base::scientific);
  std::cout << "|x - x_computed| = " << arma::norm(x-x_computed) << std::endl;

}
