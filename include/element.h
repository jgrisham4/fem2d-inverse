/**
 * \file element.h
 * \class element
 *
 * This class represents individual elements in the mesh.  There
 * are methods for getting the Jacobian determinant, finding 
 * derivatives of the i-th shape function with respect to either
 * xi or eta, and others.
 *
 * \author James Grisham
 * \date 07/23/2015
 */

#ifndef ELEMENTHEADER
#define ELEMENTHEADER

#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>
#include "armadillo"
#include "node.h"
#include "shape_fcns_2d.h"

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

template <typename T>
class element {
  
  public:
    element() {};
    element(const std::vector<int>& conn) : connectivity{conn} {};
    element(const std::vector<int>& conn, const std::vector<node<T> >& elem_nodes) : connectivity{conn}, nodes{elem_nodes} {};
    void assign_nodes(const std::vector<node<T> >& mesh_nodes);
    std::vector<node<T> > get_nodes() const;
    std::vector<int> get_connectivity() const;
    T get_detJ(const T xi, const T eta) const;
    void compute_jacobian(const T xi, const T eta, arma::Mat<T>& Jinv, T& detJ, arma::Mat<T>& A) const;
    arma::Col<T> dN(const int i, const T xi, const T eta) const;

  private:
    std::vector<int> connectivity;
    std::vector<node<T> > nodes;
    
};

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

/**
  Method for assigning nodes to the element.

  @param[in] mesh_nodes a vector of all the nodes in the mesh.  This vector and the connectivity information are used to assign the correct nodes to the element.
*/

template <typename T> void element<T>::assign_nodes(const std::vector<node<T> >& mesh_nodes) {
  nodes.resize(connectivity.size());
  for (unsigned int i=0; i<connectivity.size(); ++i) {
    nodes[i] = mesh_nodes[connectivity[i]];
  }
}
  
/**
  Method for returning the connectivity.

  @return connectivity a vector of global node numbers for the current element.
*/
template <typename T> 
std::vector<int> element<T>::get_connectivity() const {
  return connectivity;
}

/**
  This method is for computing the Jacobian for the element.
  The assign nodes method must be called before computing the Jacobian

  @param[in] xi the first coordinate in the computational domain.
  @param[in] eta the second coordinate in the computational domain.
  @param[out] Jinv an Armadillo matrix which will contain the inverse of the Jacobian.
  @param[out] detJ the determinant of the Jacobian matrix.
  @param[out] A an Armadillo matrix which contains the derivatives of the shape functions wrt xi and eta.
*/
template <typename T> 
void element<T>::compute_jacobian(const T xi, const T eta, arma::Mat<T>& Jinv, T& detJ, arma::Mat<T>& A) const {
  
  // Declaring variables
  A.zeros(2,4);
  arma::Mat<T> B(4,2);
  arma::Mat<T> J;

  // Forming A matrix 
  for (int i=0; i<4; ++i) {
    A(0,i) = dNdxi<T>(i,xi,eta);
    A(1,i) = dNdeta<T>(i,xi,eta);
  }

  // Forming B matrix
  for (int i=0; i<4; ++i) {
    B(i,0) = nodes[i].get_x();
    B(i,1) = nodes[i].get_y();
  }

  // Computing the Jacobian matrix and determinant
  J = A*B;
  detJ = arma::det(J);

  // Sanity check
  /*  ---- commented out because I'm using complex numbers ----
  if (detJ <= (T) 0) {
    std::cerr << "\nERROR: Negative cell volume." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
    std::cerr << "detJ = " << detJ << std::endl;
    std::cerr << "Coordinates of nodes:" << std::endl;
    for (int i=0; i<4; ++i) {
      std::cerr << B(i,0) << " " << B(i,1) << std::endl;
    }
    exit(-1);
  }
  */

  // Finding inverse
  Jinv = arma::inv(J);

}

/**
  This method is for computing dN/dx and dN/dy for the i-th shape function.
  
  @param[in] i integer index for the i-th shape function.
  @param[in] xi first coordinate in the computational plane.
  @param[in] eta second coordinate in the computational plane.
  @return an Armadillo column vector which contains dNdX (i.e., {dN/dxi, dNdeta}).
*/

template <typename T> arma::Col<T> element<T>::dN(const int i, const T xi, const T eta) const {

  arma::Col<T> dNdX;
  arma::Mat<T> dNdXi;
  arma::Mat<T> Jinv(2,2);
  T detJ;
  compute_jacobian(xi,eta,Jinv,detJ,dNdXi);
  dNdX = Jinv*dNdXi.col(i);

  return dNdX;

}

/**
  Method for getting only the determinant of the Jacobian.

  @param[in] xi first coordinate in the computational plane.
  @param[in] eta second coordinate in the computational plane.
  @return The determinant of the Jacobian.
*/
template <typename T> T element<T>::get_detJ(const T xi, const T eta) const {

  arma::Mat<T> dNdXi;
  arma::Mat<T> Jinv;
  T detJ;
  compute_jacobian(xi,eta,Jinv,detJ,dNdXi);

  return detJ;

}

/**
  Method for getting an STL vector of node objects for the 
  current element.

  @return An STL vector which contains node objects for the current element.
 */

template <typename T> std::vector<node<T> > element<T>::get_nodes() const {
  return nodes;
}

#endif
