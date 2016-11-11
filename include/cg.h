/**
 * \file cg.h
 *
 * This header contains functions for solving systems of equations which
 * have positive definite, symmetric matrices.  The cg function is just
 * an implementation of plain conjugate gradient, and pcg is 
 * preconditioned conjugate gradient.
 *
 * \author Ashkan Akbariyeh and James Grisham
 * \date 07/24/2015
 */

#ifndef CGHEADER
#define CGHEADER

#include <cmath>
#include <iostream>
#include "armadillo"

/**
  This struct contains solution information.
*/
template <typename T>
struct solution {
  arma::Col<T> x;
  arma::Col<T> r;
  arma::Col<T> norm;
};

/**
  This function is for solving a system of equations using conjugate
  gradient.

  @param[in] A Armadillo matrix in the A x = b system.
  @param[in] b Armadillo column vector which represents the RHS.
  @param[in] x0 Armadillo column vector which contains the initial guess.
  @param[in] tol tolerance used to monitor convergence.
  @param[in] max_iter max number of iterations.
  @param[out] s structure which contains solution information.

*/

template <typename T> 
void cg(arma::Mat<T> A, arma::Col<T> b, arma::Col<T> x0, const T tol, const unsigned int max_iter, solution<T>& s) {

  // Delcaring some variables
  arma::Col<T> r, p, x, Ap, norm(max_iter);
  T alpha, rr, beta, dot_b, dot_r;
  unsigned int j;
  
  x = x0;
  r = b - A*x;
  p = r;
  dot_b = arma::dot(b,b);
  for (j=0; j<max_iter; ++j) {
  
    rr = arma::dot(r,r);
    Ap = A*p;
    alpha = rr/arma::dot(Ap,p);
    x += alpha*p;
    r -= alpha*Ap;
    dot_r = arma::dot(r,r);
    beta = dot_r/rr;
    p = r + beta*p;
    norm(j) = sqrt(dot_r/dot_b);
#ifdef DEBUG
    std::cout << "norm(" << j << ") = " << norm(j) << std::endl;
#endif

    if (norm(j) < tol) {
#ifdef DEBUG
      std::cout << "Conjugate gradient converged." << std::endl;
#endif
      ++j;
      break;
    }
  }

  s.x = x;
  s.r = r;
  s.norm = norm.rows(0,j-1);

}

template <typename T> 
arma::Col<T> solveL(const arma::Mat<T>& L, const arma::Col<T>& rhs) {
  arma::Col<T> result(rhs.n_elem);
  result(0) = rhs(0) / L(0,0);
  for (unsigned int i=1; i<rhs.n_elem; i++) {
    result(i) = (rhs(i) - arma::dot(L(i,arma::span(0,i-1)).t() , result.rows(0,i-1)))/L(i,i);
  }
  return result;
}

//PROBLEM HERE!!!!!!!!!!
template <typename T>
arma::Col<T> solveU(const arma::Mat<T>& U, const arma::Col<T>& rhs) {
  arma::Col<T> result(rhs.n_elem);
  unsigned int max = rhs.n_elem-1;
  result(max) = rhs(max) / U(max,max);
  for (unsigned int i=max-1; i>=0; i--) {
    result(i) = (rhs(i) - arma::dot(U(i,arma::span(i+1,max)).t() , result.rows(i+1,max)))/U(i,i);
  }
  return result;
}

/**
  This function is for solving a system of equations using preconditioned
  conjugate gradient.

  @param[in] A Armadillo matrix in the A x = b system.
  @param[in] b Armadillo column vector which represents the RHS.
  @param[in] x0 Armadillo column vector which contains the initial guess.
  @param[in] tol tolerance used to monitor convergence.
  @param[in] max_iter max number of iterations.
  @param[out] s structure which contains solution information.
  @param[in] L lower triangular matrix used for preconditioning.
  @param[in] U upper triangular matrix used for preconditioning.

*/

template <typename T>
void pcg(arma::Mat<T> A, arma::Col<T> b, arma::Col<T> x0, const T tol, const unsigned int max_iter, solution<T>& s, const arma::Mat<T>& L, const arma::Mat<T>& U) {

  // Delcaring some variables
  arma::Col<T> r, p, x, Ap, z, norm(max_iter), u;
  T alpha, rz, beta, dot_b, dot_rz;
  unsigned int j;
  
  x = x0;
  r = b - A*x;
  //z = inv(LU)*r;
  u = solveL(L,r);
  z = solveU(U,u);
  p = z;
  dot_b = arma::dot(b,b);
  for (j=0; j<max_iter; ++j) {
  
    rz = arma::dot(r,z);
    Ap = A*p;
    alpha = rz/arma::dot(Ap,p);
    x += alpha*p;
    r -= alpha*Ap;
    u = solveL(L,r);
    z = solveU(U,u);
    dot_rz = arma::dot(r,z);
    beta = dot_rz/rz;
    p = z + beta*p;
    norm(j) = sqrt(arma::dot(r,r)/dot_b);
    std::cout << "norm(" << j << ") = " << norm(j) << std::endl;

    if (norm(j) < tol) {
      std::cout << "Converged." << std::endl;
      ++j;
      break;
    }
  }

  s.x = x;
  s.r = r;
  s.norm = norm.rows(0,j-1);

}

template <typename T>
void pcg(arma::Mat<T> A, arma::Col<T> b, arma::Col<T> x0, const T tol, const unsigned int max_iter, solution<T>& s, const arma::Mat<T>& Minv) {

  // Delcaring some variables
  arma::Col<T> r, p, x, Ap, z, norm(max_iter), u;
  T alpha, rz, beta, dot_b, dot_rz;
  unsigned int j;
  
  x = x0;
  r = b - A*x;
  //z = inv(LU)*r;
  z = Minv*r;
  p = z;
  dot_b = arma::dot(b,b);
  for (j=0; j<max_iter; ++j) {
  
    rz = arma::dot(r,z);
    Ap = A*p;
    alpha = rz/arma::dot(Ap,p);
    x += alpha*p;
    r -= alpha*Ap;
    z = Minv * r;
    dot_rz = arma::dot(r,z);
    beta = dot_rz/rz;
    p = z + beta*p;
    norm(j) = sqrt(arma::dot(r,r)/dot_b);
    std::cout << "norm(" << j << ") = " << norm(j) << std::endl;

    if (norm(j) < tol) {
      std::cout << "Converged." << std::endl;
      ++j;
      break;
    }
  }

  s.x = x;
  s.r = r;
  s.norm = norm.rows(0,j-1);

}
#endif
