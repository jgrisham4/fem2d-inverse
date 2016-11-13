#ifndef LINALGHEADERDEF
#define LINALGHEADERDEF

#include "armadillo"

/**
 * Function for performing the forward solution
 * given a lower triangular system of equations
 * and an rhs, b.
 *
 * @param[in] L lower triangular matrix.
 * @param[in] b rhs of Ax=b system where A=LU.
 * @return solution of lower triangular system.
 */
template <typename T>
arma::Col<T> forward_solve(const arma::Mat<T>& L, const arma::Col<T>& b) {

  // Creating data structures
  arma::Col<T> y(b.n_elem);
  T sum;
  int N = b.n_elem;

  // Computing the result, row-by-row
  for (int i=0; i<N; ++i) {
    sum = T{0.0};
    for (int j=0; j<i-1; ++j) {
      sum += L(i,j)*y(j);
    }
    y(i) = (b(i) - sum)/L(i,i);
  }

  return y;

}

/**
 * Function for performing the backward solution
 * given an upper triangular system of equations
 * and an rhs, y.
 *
 * @param[in] U lower triangular matrix.
 * @param[in] y rhs of Ax=b system where A=LU.
 * @return solution of upper triangular system.
 */
template <typename T>
arma::Col<T> backward_solve(const arma::Mat<T>& U, const arma::Col<T>& y) {

  // Creating data structures
  arma::Col<T> x(y.n_elem);
  T sum;
  int N = y.n_elem;

  // Computing result, row-by-row
  for (int i=N-1; i>=0; i--) {
    sum = T{0.0};
    for (int j=i+1; j<N; ++j) {
      sum += U(i,j)*x(j);
    }
    x(i) = (y(i) - sum)/U(i,i);
  }

  return x;

}

#endif
