#ifndef FORMULATIONLAPLACE
#define FORMULATIONLAPLACE


/***************************************************************\
 * Definitions for element stiffness and element load -- 2D    *
\***************************************************************/

template <typename T>
void sample_integrands(const element<T>& elem, const T xi, const T eta, arma::Mat<T>& Ki, arma::Mat<T>& Fi) {

  // Declaring some variables
  T detJ;
  arma::Col<T> dN_i(2);
  arma::Col<T> dN_j(2);

  // Computing the Jacobian determinant
  detJ = elem.get_detJ(xi,eta);
  
  // Finding contribution to element stiffness matrix and element load vector
  for (int i=0; i<4; ++i) {
    dN_i = elem.dN(i,xi,eta);
    for (int j=0; j<4; ++j) {
      dN_j = elem.dN(j,xi,eta);
      //Ki(i,j) = (dN_i(0)*dN_j(0) + dN_i(1)*dN_j(1))*detJ;
      Ki(i,j) = (dN_i(0)*dN_j(0) + dN_i(1)*dN_j(1));
    }
    Fi(i) = T(0.0);
  }

}

#endif
