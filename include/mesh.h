/**
 * \file mesh.h
 * \class mesh
 *
 * This class represents a mesh.  It has methods for reading and
 * writing different types of grids.
 *
 * \author James Grisham
 * \date 05/02/2015
 */


#ifndef MESHHEADER
#define MESHHEADER

#include <vector>
#include <set>
#include <iostream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <initializer_list>
#include <fstream>
#include "node.h"
#include "element.h"
#include "boundary.h"
#include "problem.h"

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

template <typename T>
class mesh {
  
  public:
    mesh();
    mesh(const int iMax, const int jMax, const std::vector<T>& R_inner, const T R_outer);
    mesh(const std::string filename);
    int get_num_nodes() const;
    int get_num_elements() const;
    void set_filename(const std::string filename);
    void read_ugrid2d();
    void read_cgns();
    void write_mesh(const std::string tec_file) const;
    boundary get_boundary(const int index) const;
    void determine_node_element_con();
    int get_nbelements() const;
    std::vector<std::set<int> > get_node_elem_con() const;
    int get_nboundaries() const;
    inline int get_imax() const { return imax; };
    inline int get_jmax() const { return jmax; };
    inline std::vector<T> get_r_inner() const { return r_inner; };
    inline std::vector<node<T> > get_nodes() const { return nodes; };
    inline std::vector<element<T> > get_elements() const { return elements; };

  private:
    int nnodes, nelements, nbelements;
    int imax,jmax;
    int nquads, ntris;
    std::string mesh_file;
    bool file_defined;
    std::vector<std::vector<int> > belem;
    std::vector<int> bmarker;
    std::vector<boundary> boundaries;
    std::vector<std::set<int> > node_elem_con;  // node-element connectivity
    std::vector<T> r_inner;
    std::vector<element<T> > elements;
    std::vector<node<T> > nodes;


};

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

/**
 * Constructor
 */

template <typename T> 
mesh<T>::mesh() {
  mesh_file = "";
  file_defined = false;
}

/**
 * CTOR which generates a structured O-grid.
 *
 * @param[in] 
 */
template <typename T> 
mesh<T>::mesh(const int iMax, const int jMax, const std::vector<T>& R_inner, const T R_outer) {

  // Setting some inputs
  r_inner = R_inner;
  imax = iMax;
  jmax = jMax;
  nnodes = (imax-1)*jmax;
  nelements = (imax-1)*(jmax-1);
  nbelements = 2*(imax-1);

  // Allocating some class members
  nodes.resize(nnodes);
  elements.resize(nelements);
  belem.resize(nbelements);
  boundaries.resize(2);

  // Initializing some variables
  arma::Mat<T> r(imax-1,jmax);
  std::vector<T> theta(imax);

  // Computing necessary steps for r- and theta-directions
  T dtheta = 2.0*M_PI/(T(imax)-1.0);
  std::vector<T> dr(imax-1);
  //std::transform(R_inner.begin(),R_inner.end(),dr.begin(),[&](T Ri) {return (R_outer-Ri)/(T(jmax)-1.0); });
  for (int i=0; i<imax-1; ++i) {
    dr[i] = (R_outer - R_inner[i])/(T(jmax)-1.0);
  }

  // Computing r- and theta-coordinates for mesh
  for (int i=0; i<imax-1; ++i) {
    theta[i] = T(i)*dtheta;
    for (int j=0; j<jmax; ++j) {
      r(i,j) = R_inner[i] + T(j)*dr[i];
    }
  }

  // Creating x-y coordinates for nodes and instantiating node objects
  for (int i=0; i<imax-1; ++i) {
    for (int j=0; j<jmax; ++j) {
      nodes[i*jmax+j] = node<T>(r(i,j)*cos(theta[i]),r(i,j)*sin(theta[i]));
    }
  } 

  // Creating element objects
  std::vector<int> con_tmp(4);
  std::vector<node<T> > node_tmp(4);
  int idx_sw, idx_se, idx_ne, idx_nw;
  for (int i=0; i<imax-2; ++i) {
    for (int j=0; j<jmax-1; ++j) { 

      // Filling in indices 
      idx_sw = i*jmax+j;
      idx_ne = (i+1)*jmax+j+1;
      //idx_se = (i+1)*jmax+j;
      //idx_nw = i*jmax+j+1;
      idx_nw = (i+1)*jmax+j;
      idx_se = i*jmax+j+1;

      // Populating node and connectivity vectors
      con_tmp = std::vector<int>{idx_sw, idx_se, idx_ne, idx_nw};
      node_tmp = std::vector<node<T> >{nodes[idx_sw],nodes[idx_se],nodes[idx_ne],nodes[idx_nw]};

      // Instantiating elements
      elements[i*(jmax-1)+j] = element<T>(con_tmp,node_tmp);

    }
  }

  // This is the final slice in the j-direction which requires information
  // from the first slice in the j-direction.
  for (int j=0; j<jmax-1; ++j) {

    // Filling in indices
    idx_sw = (imax-2)*jmax+j;
    idx_ne = j+1;
    //idx_se = j;
    //idx_nw = (imax-2)*jmax+j+1;
    idx_nw = j;
    idx_se = (imax-2)*jmax+j+1;

    // Populating node and connectivity vectors
    con_tmp = std::vector<int>{idx_sw, idx_se, idx_ne, idx_nw};
    node_tmp = std::vector<node<T> >{nodes[idx_sw],nodes[idx_se],nodes[idx_ne],nodes[idx_nw]};

    // Instantiating elements
    elements[(imax-2)*(jmax-1)+j] = element<T>(con_tmp,node_tmp);
  }

  // Creating boundary elements
  std::vector<std::vector<int> > belem_inner(imax-1);
  std::vector<std::vector<int> > belem_outer(imax-1);
  for (int i=0; i<imax-2; ++i) {
    belem_inner[i] = std::vector<int>({i*jmax,(i+1)*jmax});
    belem_outer[i] = std::vector<int>({(i+1)*jmax-1,(i+2)*jmax-1});
  }
  belem_inner[imax-2] = std::vector<int>({(imax-2)*jmax,0}); 
  belem_outer[imax-2] = std::vector<int>({(imax-2)*jmax+jmax-1,jmax-1});  

  // Creating boundaries
  boundaries[0] = boundary(0,belem_inner);
  boundaries[1] = boundary(1,belem_outer);

}

/**
 * Constructor with file name.
 *
 * @param[in] filename name of mesh file to be imported.
 */
template <typename T> 
mesh<T>::mesh(const std::string filename) : mesh_file{filename} {
  file_defined = true;
#ifdef DEBUG
  std::cout << "mesh object is being created." << std::endl;
#endif
}

/**
 * Method for setting the file name.
 *
 * @param[in] filename name of mesh file to be imported.
 */
template <typename T> 
void mesh<T>::set_filename(const std::string filename) {
  mesh_file = filename;
  file_defined = true;
}

/** 
 * Method for getting number of nodes.
 *
 * @return integer which represent the total number of nodes in the mesh.
 */
template <typename T> 
int mesh<T>::get_num_nodes() const {
  return nnodes;
}

/** 
 * Method for getting number of elements.
 *
 * @return integer which represent the total number of elements in the mesh.
 */
template <typename T> 
int mesh<T>::get_num_elements() const {
  return nelements;
}


/**
 * Method for reading 2D UGRID files.
 */
template <typename T> 
void mesh<T>::read_ugrid2d() {

  // Declaring some variables
  int el;
  double xval, yval, xdummy;
  std::vector<std::vector<int> > elem;
  std::vector<int> quad_tmp(4,0);

  // Making sure the file name has been specified
  if (!file_defined) {
    std::cerr << "\nERROR mesh::read_ugrid() -- mesh_file not defined." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
    exit(1);
  }

  // Opening file
  FILE* fp;
  fp = fopen(mesh_file.c_str(), "r");
  if (fp==NULL) {
    std::cerr << "\nERROR mesh::read_ugrid() -- can't open " << mesh_file << std::endl;
    std::cerr << "Exiting." << std::endl;
    exit(1);
  }

  // Reading header data
  int dummy1, dummy2, dummy3, dummy4;
  fscanf(fp,"%d %d %d %d %d %d %d",&nnodes,&ntris,&nquads,&dummy1,&dummy2,&dummy3,&dummy4);
  nelements = ntris+nquads;
	nodes.resize(nnodes);
	elements.resize(nelements);
  std::cout << "\nNumber of nodes: " << nnodes << std::endl;
  std::cout << "Number of elements: " << nelements << std::endl;

  // Reading points
 std::vector<double> col(3*nnodes,0.0);
  for (unsigned int i=0; i<col.size(); i++) {
		fscanf(fp,"%lf",&xdummy);
		col[i] = xdummy;
  }

  // Separating x- and y-values
	for (int i=0; i<nnodes; i++) {
		xval = col[i*3];
		yval = col[i*3+1];
		nodes[i] = node<T>(xval,yval);
	}

  // Reading connectivity information for triangular elements
  for (int i=0; i<ntris; i++) {
    elem.push_back(quad_tmp);
    for (int j=0; j<3; j++) {
      fscanf(fp,"%d",&el);
      elem[i][j] = el-1;
    }
    elem[i][3] = elem[i][2];  // Using a degenerate quad
		elements[i] = element<T>(elem[i]);
  }

  // Reading connectivity information for quadrilateral elements
  for (int i=ntris; i<nquads+ntris; i++) {
    elem.push_back(quad_tmp);
    for (int j=0; j<4; j++) {
      fscanf(fp,"%d",&el);
      elem[i][j] = el-1;
    }
		elements[i] = element<T>(elem[i]);
  }

  // Assigning nodes for each element
  for (int en=0; en<nquads+ntris; ++en) {
    elements[en].assign_nodes(nodes);
  }

  // Reading dummy
  int idummy;
  for (int i=0; i<nelements; i++) {
    fscanf(fp,"%d",&idummy);
  }

  // Reading number of boundary faces
  fscanf(fp,"%d",&nbelements);
  std::cout << "Number of boundary elements: " << nbelements << std::endl;
  bmarker.resize(nbelements);
  std::vector<int> belem_tmp(2,0);
  for (int i=0; i<nbelements; i++) {
    fscanf(fp,"%d %d %d",&dummy1,&dummy2,&dummy3);
    belem.push_back(belem_tmp);
    belem[i][0] = dummy1-1;
    belem[i][1] = dummy2-1;
    bmarker[i] = dummy3;
  }
  fclose(fp);

  // Creating boundary objects
  int current_marker = bmarker[0];
  std::vector<int> indices;
  std::vector<int> marker;
  std::vector <std::vector<int> > current_belems;
  unsigned int num_bounds = belem.size();

  for (unsigned int i=0; i<num_bounds; i++) {
    if (bmarker[i]!=current_marker) {
      boundaries.push_back(boundary(current_marker,current_belems));
      current_marker = bmarker[i];
      current_belems.clear();
    }
    current_belems.push_back(belem[i]);
  }

	// Printing some debug info
  /*
#ifdef DEBUG
	for (unsigned int i=0; i<nodes.size(); i++) {
		std::cout << nodes[i].get_x() << " " << nodes[i].get_y() << std::endl;
	}
	for (unsigned int i=0; i<elem.size(); i++) {
		for (unsigned int j=0; j<elem[i].size(); j++) {
			std::cout << elem[i][j] << " ";
		}
		std::cout << std::endl;
	}
#endif
*/

  // Adding the last boundary
  boundaries.push_back(boundary(current_marker,current_belems));

  std::cout << "Found " << boundaries.size() << " boundaries." << std::endl;
  std::cout << "Done reading mesh." << std::endl;

}

/**
 * Method for writing the mesh to an ASCII Tecplot file.
 *
 * @param[in] tec_file string which holds the name of the file to which data is written.
 */
template <typename T> 
void mesh<T>::write_mesh(const std::string tec_file) const {

  // This method only writes the mesh and connectivity.  It does not 
  // write the boundary information.

  std::ofstream tecfile;
  std::vector<int> con_tmp;
  std::cout << "\nWriting mesh to Tecplot file named: " << tec_file << std::endl;
  tecfile.open(tec_file.c_str());
  tecfile.setf(std::ios_base::scientific);
  tecfile.precision(14);
  tecfile <<"VARIABLES=\"X\",\"Y\"" << std::endl;
  tecfile <<"ZONE T=\"area\" N=" << nnodes << " E=" << nelements << " ET=QUADRILATERAL F=FEPOINT" << std::endl;
  for (unsigned int i=0; i<nodes.size(); i++) {
    tecfile << nodes[i].get_x() << " " << nodes[i].get_y() << "\n";
  }
  for (unsigned int i=0; i<elements.size(); i++) {	
		con_tmp = elements[i].get_connectivity();
		for (int x : con_tmp) {
	    tecfile << x+1 <<" ";
		}
    if (con_tmp.size()==3) {
      tecfile << con_tmp[2]+1;
    }
		tecfile << "\n";
  }
  tecfile.close();

}

template <typename T> 
boundary mesh<T>::get_boundary(const int index) const {
  return boundaries[index];
}

/**
 * This method is for determining what elements each node is
 * connected to.  Calling this method populates a private member
 * of the mesh class.
 */
template <typename T>
void mesh<T>::determine_node_element_con() {

  std::vector<int> econ;
  node_elem_con.resize(nnodes);
  for (unsigned int en=0; en<elements.size(); ++en) {
    econ = elements[en].get_connectivity();
    for (unsigned int n=0; n<econ.size(); ++n) {
      node_elem_con[econ[n]].insert(en);      
    }
  }
}

template <typename T> 
int mesh<T>::get_nbelements() const {
  return nbelements;
}

template <typename T>
std::vector<std::set<int> > mesh<T>::get_node_elem_con() const {
  return node_elem_con;
}

template <typename T>
int mesh<T>::get_nboundaries() const {
  return boundaries.size();
}

#endif
