#include "mesh.h"
#include "laplace.h"

int main() {

  // Thermodynamic inputs
  double k = 10.0;
  double T_inner = 100.0+273.0;
  double T_outer = 10.0+273.0;

  // Inputs for an o-grid
  int imax = 100;
  int jmax = 60;
  double r_outer = 10.0;
  std::vector<double> r(imax-1);
  for (int i=0; i<imax-1; ++i) {
    r[i] = 5.0;
  }

  // Creating a new problem
  laplace<double>* p = new laplace<double>();

  // Setting thermal conductivity
  p->set_problem_specific_data(k);

  // Creating a mesh and assembling global data structures
  p->discretize(imax,jmax,r,r_outer);

  // Applying BCs
  p->apply_bc_dirichlet(0,0,T_inner);
  p->apply_bc_dirichlet(1,0,T_outer);

  // Solving system
  p->solve();

  // Writing data to file
  p->extract_slice("slice.dat");
  p->write_tecplot("soln.tec");

  delete p;

  return 0;

}
