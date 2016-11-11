/**
 * \file divcurl.h
 *
 * This header contains a class named divcurl, which is derived from the 
 * abstract base class named problem. It is templated based on type,
 * dimension, and discretization type.  The functions for sampling the
 * integrands are overloaded based on the number of coordinates in the 
 * computational plane which are provided as arguments.
 *
 * \author James Grisham
 * \date 07/25/2015
 */

#ifndef DIVCURLHEADER
#define DIVCURLHEADER

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <iterator>
#include <initializer_list>
#include "armadillo"
#include "node.h"
#include "element.h"
#include "gdata.h"
#include "problem.h"
#include "mesh.h"
#include "shape_fcns_2d.h"
#include "util.h"
#include "formulation_divcurl.h"
#include "cg.h"
#include "boundary.h"

template <typename T> 
T dummysource(T x, T y) {
  return (T) 0.0;
}

/***************************************************************\
 * This is the general derived class definition.               *
\***************************************************************/

template <typename T, const int D, const int DI> 
class divcurl : public problem<T,D,DI> {

  protected:
    std::vector<bool> transformed_boundary;
    arma::Mat<T> K;
    arma::Col<T> u;
    arma::Col<T> F;
    arma::Col<T> source;
    mesh<T> grid;
    solution<T> cgsol;
    std::vector<arma::Cube<T> > boundary_transformation;
    void integrate_element(const int npts, const element<T>& elem, T (*f)(T,T), arma::Mat<T>& Kelem, arma::Col<T>& Felem) override;

  public:
    void discretize(const std::string mesh_file) override;
    void discretize(const std::string mesh_file, T (*f)(T,T)) override;
    void solve() override;
    void write_tecplot(const std::string tec_file) const override;
    void apply_bc_dirichlet(const int boundary_index, const int variable_index, T value) override;
    void apply_bc_dirichlet_normal(const int boundary_index, T value) override;

};

/***************************************************************\
 * This is the 2D, LSFEM case.                                 *
\***************************************************************/

template <typename T> 
class divcurl<T,D2,LSFEM> : public problem<T,D2,LSFEM> {

  protected:
    std::vector<bool> transformed_boundary;
    arma::Mat<T> K;
    arma::Col<T> u;
    arma::Col<T> F;
    arma::Col<T> source;
    mesh<T> grid;
    solution<T> cgsol;
    std::vector<arma::Cube<T> > boundary_transformation;
    void integrate_element(const int npts, const element<T>& elem, T (*f)(T,T), arma::Mat<T>& Kelem, arma::Col<T>& Felem) override;

  public:
    void discretize(const std::string mesh_file) override;
    void discretize(const std::string mesh_file, T (*f)(T,T)) override;
    void solve() override;
    void write_tecplot(const std::string tec_file) const override;
    void apply_bc_dirichlet(const int boundary_index, const int variable_index, T value) override;
    void apply_bc_dirichlet_normal(const int boundary_index, T value) override;

};

template<typename T>
void divcurl<T,D2,LSFEM>::discretize(const std::string mesh_file) {

  // D2 -> dimension and nn -> number of nodes per element
  int nn = 4;

  // Reading mesh
  grid.set_filename(mesh_file);
  grid.read_ugrid2d();

  // Setting transformed boundary flag to false
  int nboundaries = grid.get_nboundaries();
  int nng = grid.get_num_nodes();
  for (int i=0; i<grid.get_nboundaries(); ++i) {
    transformed_boundary.push_back(false);
  }
  boundary_transformation.resize(nboundaries);

  // Assigning a dummy source term for now
  source.set_size(2*nng);
  source.fill((T) 0);

  // Setting up
  arma::Mat<T> Ke;
  arma::Col<T> Fe;
  K.zeros(2*nng,2*nng);
  F.zeros(2*nng);
  std::vector<int> con;

  // Assembling global stiffness matrix and global load vector
  int numel = grid.get_num_elements();
  for (int en=0; en<numel; ++en) {

    // Getting connectivity
    con = grid.elements[en].get_connectivity();

    // Integrating
    integrate_element(4,grid.elements[en],&dummysource,Ke,Fe);

    // Putting in the global structures
    for (unsigned int i=0; i<nn; ++i) {
      for (unsigned int j=0; j<D2; ++j) {
        for (unsigned int k=0; k<nn; ++k) {
          for (unsigned int l=0; l<D2; ++l) {
            K(con[i]*D2+j,con[k]*D2+l) += Ke(i+nn*j,nn*l+k);
          }
        }
        F(con[i]*D2+j) += Fe(j*nn+i);
      }
    }

  }

}

template<typename T>
void divcurl<T,D2,LSFEM>::discretize(const std::string mesh_file, T (*f)(T,T)) {

  // D2 -> dimension and nn -> number of nodes per element
  int nn = 4;

  // Reading mesh
  grid.set_filename(mesh_file);
  grid.read_ugrid2d();

  // Setting transformed boundary flag to false
  int nboundaries = grid.get_nboundaries();
  int nng = grid.get_num_nodes();
  for (int i=0; i<grid.get_nboundaries(); ++i) {
    transformed_boundary.push_back(false);
  }
  boundary_transformation.resize(nboundaries);

  // Setting up
  arma::Mat<T> Ke;
  arma::Col<T> Fe;
  K.zeros(2*nng,2*nng);
  F.zeros(2*nng);
  std::vector<int> con;

  // Assembling global stiffness matrix and global load vector
  int numel = grid.get_num_elements();
  for (int en=0; en<numel; ++en) {

    // Getting connectivity
    con = grid.elements[en].get_connectivity();

    // Integrating
    integrate_element(4,grid.elements[en],f,Ke,Fe);

    // Putting in the global structures
    for (unsigned int i=0; i<nn; ++i) {
      for (unsigned int j=0; j<D2; ++j) {
        for (unsigned int k=0; k<nn; ++k) {
          for (unsigned int l=0; l<D2; ++l) {
            K(con[i]*D2+j,con[k]*D2+l) += Ke(i+nn*j,nn*l+k);
          }
        }
        F(con[i]*D2+j) += Fe(j*nn+i);
      }
    }

  }

}


template <typename T>
void divcurl<T,D2,LSFEM>::integrate_element(const int npts, const element<T>& elem, T (*f)(T,T), arma::Mat<T>& Kelem, arma::Col<T>& Felem) {
  
  // Declaring variables
  std::vector<T> gpts(npts), w(npts);

  // Figuring out the size of the arrays
  // The below is dim^(dim+1)*basis_order
  // It will be different for 3D
  int b = 1;
  int maxind = pow(D2,D2+1)*b;

  // Making sure matrix and vector have been initialized
  arma::Mat<T> Ktmp = arma::zeros<arma::Mat<T> >(maxind,maxind);
  arma::Col<T> Ftmp = arma::zeros<arma::Col<T> >(maxind);
  Kelem.zeros(maxind,maxind);
  Felem.zeros(maxind);

  // Getting weights and Gauss points
  gdata<T>(npts,gpts,w);

  // Numerically integrating
  for (int i=0; i<npts; ++i) {
    for (int j=0; j<npts; ++j) {
      sample_integrands<T>(elem,gpts[i],gpts[j],f,Ktmp,Ftmp);
      Kelem += w[i]*w[j]*Ktmp;
      Felem += w[i]*w[j]*Ftmp;
    }
  }

}

/**
  This method is for applying zero normal velocity boundary conditions.
  This task is accomplished by transforming the elements which are 
  connected to the boundary elements, so that the boundary element is oriented 
  with its face normal pointing in the positive x-direction.  Then,
  the velocity components are v_normal and v_tangential.  The first is
  set to zero, and the new element stiffness matrix is assembled.  The 
  implementation is complicated.  The transformation is given by:
  \f[
    \beta_{ij} = \begin{bmatrix}
      \cos\theta & \sin \theta \\
      -\sin\theta & \cos\theta
    \end{bmatrix}
  \f]
  
  @param[in] boundary_index Integer index used to identify different boundaries.

*/

template <typename T>
void divcurl<T,D2,LSFEM>::apply_bc_dirichlet_normal(const int boundary_index, T value) {

  // Making sure that the boundary hasn't already been transformed
  if (transformed_boundary[boundary_index]) {
    std::cerr << "ERROR: Boundary " << boundary_index << " has already been transformed." << std::endl;
    std::cerr << "Exiting." << std::endl;
    exit(-1);
  }
  transformed_boundary[boundary_index] = true;

  // Declaring some variables
  int variable_index = 0;  // 0 -- normal component, 1 -- tangential component
  unsigned int nn;
  std::map<int,arma::Col<T> > bnode_normals;
  boundary bndry(grid.get_boundary(boundary_index));
  std::vector<std::vector<int> > bcons = bndry.get_boundary_connectivity();
  std::vector<int> con;
  arma::Mat<T> M(D2,D2,arma::fill::zeros);
  T x1,x2,y1,y2,dx,dy,nx,ny,mag;
  arma::Mat<T> Ke, Ke_tmp, Trnsfrm;
  arma::Col<T> Fe, Fe_tmp;

  // Getting unique nodes
  // The keys for the bnode_normals map are the unique boundary node numbers
  // Given a boundary node index, this map can be used to find its normal vector.
  for (std::vector<int> &b : bcons) {
    for (int i : b) {
      bnode_normals[i] = arma::Col<T>(D2,arma::fill::zeros);
    }
  }

  // Creating Cube-type array for rotation matrices for each node
  boundary_transformation[boundary_index] = arma::Cube<T>(D2,D2,bnode_normals.size(),arma::fill::zeros);
  std::cout << "size of bnode_normals map: " << bnode_normals.size() << std::endl;

  // Calculating and aggregating the boundary node normals
  for (std::vector<int> &b : bcons) {
    
    x1=grid.nodes[b[0]].get_x();
    x2=grid.nodes[b[1]].get_x();
    y1=grid.nodes[b[0]].get_y();
    y2=grid.nodes[b[1]].get_y();
    dx = x2 - x1;
    dy = y2 - y1;
    mag = sqrt(dx*dx + dy*dy);
    nx =  dy/mag;
    ny = -dx/mag;
    bnode_normals[b[0]] += arma::Col<T>({nx,ny}); 
    bnode_normals[b[1]] += arma::Col<T>({nx,ny});

  }

  // Renormalizing the normal vectors so that they are unit normals
  for (auto it = bnode_normals.begin(); it!=bnode_normals.end(); ++it) {
    it->second = arma::normalise(it->second);
  }

  // Determining node-element connectivity
  std::vector<std::set<int> > node_elem_con = grid.get_node_elem_con();
  if (node_elem_con.size()==0){
    grid.determine_node_element_con();
    node_elem_con = grid.get_node_elem_con();
  }
    
  // 
  int Kesize;
  for (auto bn = bnode_normals.begin(); bn!=bnode_normals.end(); ++bn) {
    nx = bn->second(0);
    ny = bn->second(1);
    //std::cout << "nvec["<< bn->first <<"] = " << nx << " " << ny << std::endl;
    M(0,0) =  nx;
    M(0,1) = -ny;
    M(1,0) =  ny;
    M(1,1) =  nx;
    (boundary_transformation[boundary_index]).slice(std::distance(bnode_normals.begin(), bn)) = M;
    std::cout << "slice for bnode "<< bn->first << std::endl;
    std::cout << boundary_transformation[boundary_index].slice(std::distance(bnode_normals.begin(),bn));
    
    // bn->first is the node number of a boundary node
    for (auto econ = node_elem_con[bn->first].begin(); econ != node_elem_con[bn->first].end(); ++econ ){

      // Finding the element stiffness and load again (for the elements
      // which have at least one edge on the boundary)
      integrate_element(4,grid.elements[*econ],&dummysource,Ke_tmp,Fe_tmp);
      Kesize = Ke_tmp.n_rows;
      Ke = arma::Mat<T>(Kesize,Kesize,arma::fill::zeros);
      Fe = arma::Col<T>(Kesize,arma::fill::zeros);

      // Need to figure out which rows and columns to subtract
      // Find out which local node number in the element corresponds to the 
      // global bnode index
      // vector<int> con = grid.elements[*econ] <-- element connectivity
      con = grid.elements[*econ].get_connectivity();
      unsigned int idx = 0;
      for (idx=0; idx < con.size(); ++idx){
        if (con[idx] != bn->first) {
          break;
        }
      }
      
      Ke = - Ke_tmp;
      Fe = - Fe_tmp;
      // Now need to apply the transformation
      Trnsfrm = arma::Mat<T>(Kesize,Kesize,arma::fill::eye);
      Trnsfrm(arma::span(D2*idx,D2*idx+D2-1),arma::span(D2*idx,D2*idx+D2-1)) = M;
      Ke += Trnsfrm.t() * Ke_tmp * Trnsfrm;
      Fe += Trnsfrm.t() * Fe_tmp;
      // Assembling into the global matrix
      nn = con.size();
      for (unsigned int i=0; i<nn; ++i) {
        for (unsigned int j=0; j<D2; ++j) {
          for (unsigned int k=0; k<nn; ++k) {
            for (unsigned int l=0; l<D2; ++l) {
              K(con[i]*D2+j,con[k]*D2+l) += Ke(i+nn*j,nn*l+k);
            }
          }
          F(con[i]*D2+j) += Fe(j*nn+i);
        }
      }
    }

    // Using the penalty method to apply the boundary condition
    auto node_num = bn->first;
    F(node_num*D2+variable_index) = value*1e6*K(node_num*D2+variable_index,node_num*D2+variable_index); 
    K(node_num*D2+variable_index,node_num*D2+variable_index) *= 1e6;
//      K(node_num*D2+variable_index,node_num*D2+variable_index) = 1*1e10;
//      F(node_num*D2+variable_index) = value*1e10;

  }
  std::cout << "boundary_index = " << boundary_index << std::endl;
  std::cout << "The size is set to " << boundary_transformation[boundary_index].n_slices << std::endl;

}

template <typename T>
void divcurl<T,D2,LSFEM>::apply_bc_dirichlet(const int boundary_index, const int variable_index, T value) {
  
  // Declaring some variables
  boundary bndry(grid.get_boundary(boundary_index));
  std::vector<std::vector<int> > bcons = bndry.get_boundary_connectivity();

  for (std::vector<int> &b : bcons) {
    for (int node_num : b) {

      // Putting one on the diagonal and assigning the load
      K(node_num*D2+variable_index,node_num*D2+variable_index) = 1*1e6;
      F(node_num*D2+variable_index) = value*1e6;

    }
  }


}

template <typename T>
void divcurl<T,D2,LSFEM>::solve() {

  std::cout << "Solving the system." << std::endl;
  arma::Col<T> u0 = arma::zeros<arma::Col<T> >(F.n_elem);
  arma::Mat<T> Minv = arma::Mat<T>(K.n_rows,K.n_rows,arma::fill::zeros);
  for (int i=0; i<Minv.n_rows; ++i){
    Minv(i,i)=1/K(i,i);
  }
//  cg<T>(K,F,u0,(T) 3e-14, 10000, cgsol);
  //pcg<T>(K,F,u0,(T) 1e-26, 10000, cgsol,Minv);
  //u = cgsol.x;
  u = arma::solve(K,F);

  // Transforming boundaries back, if necessary
  int nbndrs = grid.get_nboundaries();
  for (int i=0; i<nbndrs; ++i) {
    if (transformed_boundary[i]) {

      std::set<int> ubnode;
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << "Transforming boundary " << i << " back." << std::endl;

      boundary bndry(grid.get_boundary(i));
      std::vector<std::vector<int> > bcons = bndry.get_boundary_connectivity();
      
      // Getting unique nodes
      for (std::vector<int> &b : bcons) {
        for (int ii : b) {
          ubnode.insert(ii);
        }
      }

      // b is a pointer to unique boundary node numbers
      // We want to multiply the varibales of this node number by the transformation matrix
      // which is stored in an arma::cube.
      // boundary_transformation is a vector of length nbndrs of arma::cube 
      std::cout << "size of cube " << i << " is " << boundary_transformation[i].n_slices << std::endl;
      for (auto b=ubnode.begin(); b!=ubnode.end(); ++b){
        std::cout << "slice for bnode "<< *b << std::endl;
        std::cout << boundary_transformation[i].slice(std::distance(ubnode.begin(),b));

        u.rows(*b*D2,*b*D2+D2-1) = boundary_transformation[i].slice(std::distance(ubnode.begin(),b))*u.rows(*b*D2,*b*D2+D2-1);
      }

    }
  }
  
}

template <typename T>
void divcurl<T,D2,LSFEM>::write_tecplot(const std::string tec_file) const {

  // Figuring out the size of the arrays
  // The below is dim^(dim+1)*basis_order
  // It will be different for 3D
  int b = 1;
  int maxind = pow(D2,D2+1)*b;

  std::cout << "Writing solution to file named " << tec_file << "." << std::endl;

  // Opening up stream and writing header
  std::ofstream tecfile;
  std::vector<int> con_tmp;
  tecfile.open(tec_file.c_str());
  if (!tecfile.is_open()) {
    std::cerr << "\nERROR: Can't open " << tec_file << " for writing data." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
  }
  tecfile.setf(std::ios_base::scientific);
  tecfile.precision(14);
  tecfile << "variables=\"x\",\"y\",\"u\",\"v\"" << std::endl;
  tecfile << "zone t=\"div-curl\" n=" << grid.get_num_nodes() << " e=" << grid.get_num_elements() << " et=quadrilateral f=fepoint" << std::endl;

  // Writing nodes and the solution at each node
  std::cout << "Number of nodes: " << grid.nodes.size() << std::endl;
  std::cout << "Dimension of solution: " << u.n_elem << std::endl;
  for (unsigned int n=0; n<grid.nodes.size(); ++n) {
    tecfile << grid.nodes[n].get_x() << " " << grid.nodes[n].get_y() << " " << u(2*n) << " " << u(2*n+1) << "\n";
  }

  // Writing connectivity
  for (unsigned int en=0; en<grid.elements.size(); ++en) {
    con_tmp = grid.elements[en].get_connectivity();
    for (int idx : con_tmp) {
      tecfile << idx+1 << " ";
    }
    tecfile << "\n";
  }
  tecfile.close();

}
#endif
