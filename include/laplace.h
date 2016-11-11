/**
 * \file laplace.h
 *
 * This header contains a class named laplace. It is a specialization for an 
 * steady-state heat conduction problem.  Specifically, the problem
 * involves an infinitely long pipe with known temperatures on the outside
 * and inside.  Given internal temperature, external temperature and 
 * external heat flux, find the shape of the interior.  This is accomplished 
 * externally.
 *
 * \author James Grisham
 * \date 10/18/2016
 */

#ifndef LAPLACEHEADER
#define LAPLACEHEADER

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <type_traits>
#include <complex>
#include "armadillo"
#include "node.h"
#include "element.h"
#include "mesh.h"
#include "gdata.h"
#include "shape_fcns_2d.h"
#include "formulation_laplace.h"
#include "boundary.h"

template <typename P> class optimization_driver;

/***************************************************************\
 * This is the 2D, GFEM case.                                  *
\***************************************************************/

template <typename T,typename C=T>
class laplace {

  template <typename P> friend class optimization_driver;

  protected:
    arma::Mat<T> K;
    arma::Mat<T> L;
    arma::Mat<T> U;
    arma::Mat<T> Kinv;
    arma::Col<T> u;
    arma::Col<T> F;
    mesh<C> grid;
    void integrate_element(const int npts, const element<T>& elem, arma::Mat<T>& Kelem, arma::Col<T>& Felem);
    void integrate_element_perturbed(const int npts, const element<C>& elem, arma::Mat<C>& Kelem, arma::Col<C>& Felem);

  public:
    typedef T VT;
    laplace() {};
    void discretize(const int imax, const int jmax, const std::vector<T>& r_i, const T r_o);
    void assemble();
    void discretize_perturbed(const int imax, const int jmax, const std::vector<C>& r_i, const T r_o);
    void solve();
    void write_tecplot(const std::string& tec_file) const;
    void write_mesh(const std::string& mesh_file) const { grid.write_mesh(mesh_file); };
    void apply_bc_dirichlet(const int boundary_index, const int variable_index, T value);
    void extract_slice(const std::string& slice_file) const;
    void set_problem_specific_data(const T d);
    inline void set_K(arma::Mat<T>& Kmat) { K = Kmat; };
    inline void set_f(arma::Col<T>& fvec) { F = fvec; };
    inline arma::Col<T> get_u() const { return u; };
    inline arma::Mat<T> get_K() const { return K; };
    inline arma::Col<T> get_f() const { return F; };
    inline arma::Mat<T> get_Kinv() const { return Kinv; };
    inline mesh<C> get_grid() const { return grid; };
    inline void set_grid(const mesh<C>& m) { grid = m; };

  private:
    T k;
};

/**
 * Method for generating mesh and assembling system of equations.
 *
 * @param[in] imax number of grid points in i-direction.
 * @param[in] jmax number of grid points in j-direction.
 * @param[in] r_i vector of inner radius locations.
 * @param[in] r_o outer radius.
 */
template <typename T,typename C>
void laplace<T,C>::discretize(const int imax, const int jmax, const std::vector<T>& r_i, const T r_o) {

#ifdef VERBOSE
  std::cout << "Generating o-grid." << std::endl;
#endif

  // Generating mesh
  grid = mesh<T>(imax,jmax,r_i,r_o);

#ifdef VERBOSE
  std::cout << "Done generating mesh." << std::endl;
#endif

  // Assembling the system of equations
  assemble();

}

/**
 * Method for assembling the linear system of equations.
 */
template <typename T,typename C>
void laplace<T,C>::assemble() {

#ifdef VERBOSE
  std::cout << "Assembling global stiffness matrix and load vector." << std::endl;
#endif 

  // Setting up some variables
  int nnodes = grid.get_num_nodes();
  int nelem  = grid.get_num_elements();
  arma::Mat<T> Ke;
  arma::Col<T> Fe;
  K.zeros(nnodes,nnodes);
  F.zeros(nnodes);

  // Assembling global stiffness matrix and global load vector for normal case
  std::vector<int> con;
  std::vector<element<T> > elements = grid.get_elements();
  for (int en=0; en<nelem; ++en) {

    // Getting element connectivity
    con = elements[en].get_connectivity();

    // Integrating element
    integrate_element(1,elements[en],Ke,Fe);

    // Adding element stiffness matrix and element load vector to global
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        K(con[i],con[j]) += Ke(i,j);
      }
      F(con[i]) += Fe(i);
    }
  }

#ifdef VERBOSE
  std::cout << "Done assembling." << std::endl;
#endif

}

/**
 * Method for generating mesh and assembling system of equations for perturbed case.
 *
 * @param[in] imax number of grid points in i-direction.
 * @param[in] jmax number of grid points in j-direction.
 * @param[in] r_i vector of inner radius locations.
 * @param[in] r_o outer radius.
 */
template <typename T,typename C>
void laplace<T,C>::discretize_perturbed(const int imax, const int jmax, const std::vector<C>& r_i, const T r_o) {

  // Generating mesh
  grid = mesh<C>(imax,jmax,r_i,C(r_o,T{0.0}));

  // Setting up some variables
  int nnodes = grid.get_num_nodes();
  int nelem  = grid.get_num_elements();
  arma::Mat<C> Ke;
  arma::Col<C> Fe;
  K.zeros(nnodes,nnodes);
  F.zeros(nnodes);

  // Assembling global stiffness matrix and global load vector for normal case
  std::vector<int> con;
  std::vector<element<C> > elements = grid.get_elements();
  for (int en=0; en<nelem; ++en) {

    // Getting element connectivity
    con = elements[en].get_connectivity();

    // Integrating element
    integrate_element_perturbed(1,elements[en],Ke,Fe);

    // Adding element stiffness matrix and element load vector to global
    for (int i=0; i<4; ++i) {
      for (int j=0; j<4; ++j) {
        K(con[i],con[j]) += Ke(i,j).imag();
      }
      F(con[i]) += Fe(i).imag();
    }
  }
}


/**
 * This method is for computing the element stiffness matrix and element load
 * vector.
 */
template <typename T,typename C>
void laplace<T,C>::integrate_element(const int npts, const element<T>& elem, arma::Mat<T>& Kelem, arma::Col<T>& Felem) {

  // Declaring some variables
  std::vector<T> gpts(npts),w(npts);
  arma::Mat<T> Ktmp = arma::zeros<arma::Mat<T> >(4,4);
  arma::Col<T> Ftmp = arma::zeros<arma::Col<T> >(4);

  // Initializing matrices and vectors
  Kelem.zeros(4,4);
  Felem.zeros(4);

  // Getting weights and Gauss points
  gdata<T>(npts,gpts,w);

  // Numerically integrating using Gaussian quadrature
  for (int i=0; i<npts; ++i) {
    for (int j=0; j<npts; ++j) {
      sample_integrands<T>(elem,gpts[i],gpts[j],Ktmp,Ftmp);
      Kelem += k*w[i]*w[j]*Ktmp;
      Felem += w[i]*w[j]*Ftmp;
    }
  }

}

/**
 * This method is for computing the element stiffness matrix and element load
 * vector for the perturbed case.
 */
template <typename T,typename C>
void laplace<T,C>::integrate_element_perturbed(const int npts, const element<C>& elem, arma::Mat<C>& Kelem, arma::Col<C>& Felem) {

  // Declaring some variables
  std::vector<C> gpts(npts),w(npts);
  arma::Mat<C> Ktmp = arma::zeros<arma::Mat<C> >(4,4);
  arma::Col<C> Ftmp = arma::zeros<arma::Col<C> >(4);

  // Initializing matrices and vectors
  Kelem = arma::zeros<arma::Mat<C> >(4,4);
  Felem = arma::zeros<arma::Col<C> >(4);

  // Getting weights and Gauss points
  gdata<C>(npts,gpts,w);

  // Numerically integrating using Gaussian quadrature
  for (int i=0; i<npts; ++i) {
    for (int j=0; j<npts; ++j) {
      sample_integrands<C>(elem,gpts[i],gpts[j],Ktmp,Ftmp);
      Kelem += C(k)*w[i]*w[j]*Ktmp;
      Felem += w[i]*w[j]*Ftmp;
    }
  }

}

/**
 * Method for applying Dirichlet boundary conditions.
 */
template <typename T,typename C>
void laplace<T,C>::apply_bc_dirichlet(const int boundary_index, const int variable_index, T value) {

#ifdef VERBOSE
  std::cout << "Applying Dirichlet boundary conditions to boundary " << boundary_index << "." << std::endl;
#endif

  // Getting boundary 
  boundary bnd(grid.get_boundary(boundary_index));
  std::vector<std::vector<int> > bcons = bnd.get_boundary_connectivity();

  // Getting a list of unique boundary nodes.  
  std::vector<int> unique_bnodes;
  for (auto &b : bcons) {
    for (auto nn : b ) {
      unique_bnodes.push_back(nn);
    }
  }
  std::sort(unique_bnodes.begin(),unique_bnodes.end());
  auto last = std::unique(unique_bnodes.begin(),unique_bnodes.end());
  unique_bnodes.erase(last, unique_bnodes.end());
#ifdef BC_DEBUG
  std::cout << "Found " << unique_bnodes.size() << " unique boundary nodes." << std::endl;
#endif

  // Enforcing Dirichlet BCs
  int num_nodes = grid.get_num_nodes();
  for (int bn : unique_bnodes) {
    for (int i=0; i<num_nodes; ++i) {
      K(bn,i) = T(0.0);
    }
    K(bn,bn) = T(1.0);
    F(bn) = value;
  }

#ifdef VERBOSE
  std::cout << "Done applying Dirichlet boundary conditions to boundary " << boundary_index << "." << std::endl;
#endif

}

template <typename T,typename C>
void laplace<T,C>::set_problem_specific_data(const T d) {
  k = d;
}

/**
 * Method to solve the system of equations.
 */
template <typename T,typename C>
void laplace<T,C>::solve() {

  // Using arma inverse
  //Kinv = K.i();
  //u = Kinv * F;

  // Using arma solve
  //arma::solve(u,K,F);

  // Using LU decomposition to solve the system
  // The below is used to check whether or not the matrix is upper or lower triangular
  //bool all_zero = all( X.elem(find(trimatl(X))) == 0 );
  // Need to do forward solve and back solve
  arma::lu(L,U,K);
  

}

template <typename T,typename C>
void laplace<T,C>::write_tecplot(const std::string& tec_file) const {

#ifdef VERBOSE
  std::cout << "Writing solution to file named " << tec_file << "." << std::endl;
#endif

  // Setting up stream
  std::ofstream tecfile(tec_file.c_str());
  if (!tecfile.is_open()) {
    std::cerr << "\nERROR: Can't open " << tec_file << " for writing data." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
    exit(-1);
  }
  tecfile.setf(std::ios_base::scientific);
  tecfile.precision(16);

  // Writing header information
  tecfile << "variables=\"x\",\"y\",\"T\"" << std::endl;
  tecfile << "zone t=\"Temperature\" n=" << grid.get_num_nodes() << " e=" << grid.get_num_elements() << " et=quadrilateral f=fepoint" << std::endl;

  // Writing nodes and the solution at each node
  std::vector<node<T> > nodes = grid.get_nodes();
  for (unsigned int n=0; n<nodes.size(); ++n) {
    tecfile << nodes[n].get_x() << " " << nodes[n].get_y() << " " << u(n) << "\n";
  }

  // Writing connectivity information
  std::vector<int> con_tmp;
  std::vector<element<T> > elements = grid.get_elements();
  for (unsigned int en=0; en<elements.size(); ++en) {
    con_tmp = elements[en].get_connectivity();
    for (int node_num : con_tmp) {
      tecfile << node_num + 1 << " ";
    }
    tecfile << "\n";
  }

  // Closing tecplot file
  tecfile.close();

}

/**
 * This method extracts a radial slice from the structured o-grid.
 *
 * @param[in] slice_file std::string which holds name of file.
 */
template <typename T,typename C>
void laplace<T,C>::extract_slice(const std::string& slice_file) const {

  // Opening file stream
  std::ofstream sfile(slice_file.c_str());
  if (!sfile.is_open()) {
    std::cerr << "\nError: Can't open " << slice_file << "." << std::endl;
    std::cerr << "Exiting.\n\n";
    exit(-1);
  }
  sfile.precision(16);
  sfile.setf(std::ios_base::scientific);

  // Writing data to file
  int jmax = grid.get_jmax();
  std::vector<node<T> > nodes = grid.get_nodes();
  for (int j=0; j<jmax; ++j) {
    sfile << nodes[j].get_x() << " " << u(j) << std::endl;
  }

  // Closing file
  sfile.close();

}

#endif
