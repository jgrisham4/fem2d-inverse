
#ifndef FORMULATIONDIVCURL
#define FORMULATIONDIVCURL

#include <iostream>
#include <vector>
#include "node.h"
#include "armadillo"
#include "element.h"

/***************************************************************\
 * Definitions for element stiffness and element load -- 2D    *
\***************************************************************/

template <typename T> 
void sample_integrands(const element<T>& elem, const T xi, const T eta, const arma::Col<T>& source, arma::Mat<T>& K_local, arma::Col<T>& F_local) {
  
  // Figuring out the size of the arrays
  // The below is dim^(dim+1)*basis_order
  // It will be different for 3D
  int b = 1;
  int d = 2;
  int maxind = pow(d,d+1)*b;

  // Declaring some variables
  T detJ;
  arma::Col<T> tmp(d);
  arma::Mat<T> dN(maxind/d,d);
  arma::Mat<T> A(d,maxind);

  // Computing Jacobian determinant
  detJ = elem.get_detJ(xi,eta);

  // Getting derivatives of the shape functions in each dimension
  // The below method dN returns a column which contains
  // {dN_j/dx, dN_j/dy}
  for (int j=0; j<maxind/d; ++j) {
    tmp = elem.dN(j,xi,eta);
    dN(j,0) = tmp(0);
    dN(j,1) = tmp(1);
  }

  // This is the row for the divergence
  for (int j=0; j<maxind/d; ++j) {

    // The below is complicated because this is an inner product
    // of two 2nd rank tensors.  This flattens a 2nd rank tensor
    // to first rank.
    A(0,j) = dN(j,0);
    A(0,j+maxind/d) = dN(j,1);

  }

  // This is the row for the curl
  // T sum1, sum2;
  for (int l=0; l<maxind/d; ++l) {
    /*  There is no need to do the below.  It can be easily
        observed what j must be for the permutation tensor
        to be nonzero.
    sum1 = (T) 0;
    sum2 = (T) 0;
    for (int j=0; j<dim; j++) {
      sum1 += tnsr::levi(2,j,0)*dN(l,j);
      sum2 += tnsr::levi(2,j,1)*dN(l,j);
    }
    */
    A(1,l) = tnsr::levi<T>(2,1,0)*dN(l,1);
    A(1,l+maxind/d) = tnsr::levi<T>(2,0,1)*dN(l,0);
  }

  // Multiplying by detJ
  A *= detJ;

  // Projecting the source term onto the basis
  T local_source = (T) 0;
  std::vector<int> conn = elem.get_connectivity();
  for (int i=0; i<maxind/d; ++i) {
    local_source += source(conn[i])*N(i,xi,eta);
  }
  arma::Col<T> F(d);
  F(0) = local_source*detJ;
  F(1) = (T) 0.0;

  // Forming element stiffness matrix and load vector at the given
  // Gauss points
  K_local = A.t()*A;
  F_local = A.t()*F;

}

template <typename T> 
void sample_integrands(const element<T>& elem, const T xi, const T eta, T (*f)(T,T), arma::Mat<T>& K_local, arma::Col<T>& F_local) {
  
  // Figuring out the size of the arrays
  // The below is dim^(dim+1)*basis_order
  // It will be different for 3D
  int b = 1;
  int d = 2;
  int maxind = pow(d,d+1)*b;

  // Declaring some variables
  T detJ;
  arma::Col<T> tmp(d);
  arma::Mat<T> dN(maxind/d,d);
  arma::Mat<T> A(d,maxind);
  std::vector<node<T> > local_nodes = elem.get_nodes();

  // Computing Jacobian determinant
  detJ = elem.get_detJ(xi,eta);

  // Recovering the x- and y-coordinates for the given Gauss points
  T x = (T) 0;
  T y = (T) 0;
  T Nval;
  for (int j=0; j<maxind/d; ++j) {
    Nval = N(j, xi, eta);
    x += Nval*local_nodes[j].get_x();
    y += Nval*local_nodes[j].get_y();
  }

  // Getting derivatives of the shape functions in each dimension
  // The below method dN returns a column which contains
  // {dN_j/dx, dN_j/dy}
  for (int j=0; j<maxind/d; ++j) {
    tmp = elem.dN(j,xi,eta);
    dN(j,0) = tmp(0);
    dN(j,1) = tmp(1);
  }

  // This is the row for the divergence
  for (int j=0; j<maxind/d; ++j) {

    // The below is complicated because this is an inner product
    // of two 2nd rank tensors.  This flattens a 2nd rank tensor
    // to first rank.
    A(0,j) = dN(j,0);
    A(0,j+maxind/d) = dN(j,1);

  }

  // This is the row for the curl
  // T sum1, sum2;
  for (int l=0; l<maxind/d; ++l) {
    A(1,l) = tnsr::levi<T>(2,1,0)*dN(l,1);
    A(1,l+maxind/d) = tnsr::levi<T>(2,0,1)*dN(l,0);
  }

  // Multiplying by detJ
  A *= detJ;

  // Projecting the source term onto the basis
  arma::Col<T> F(d);
  F(0) = f(x,y)*detJ;
  F(1) = (T) 0.0;

  // Forming element stiffness matrix and load vector at the given
  // Gauss points
  K_local = A.t()*A;
  F_local = A.t()*F;

}


/***************************************************************\
 * Definitions for element stiffness and element load -- 3D    *
\***************************************************************/

template <typename T> 
void sample_integrands(const element<T>& elem, const T xi, const T eta, const T zeta, const arma::Col<T> source, arma::Mat<T>& K_local, arma::Col<T>& F_local) {

  // Figuring out the size of the arrays
  // This works for b = 1,2 for sure, not sure about higher
  int b = 1; // order of the basis
  int d = 3; // dimension
  int maxind = (int) (-72.0 + 48.0*b + 16.0*d);

  std::cerr << "3D LSFEM not implemented yet." << std::endl;
  exit(-1);
  

}


#endif
