#include "mesh.h"

int main() {

  // Inputs for an o-grid
  int imax = 5;
  int jmax = 4;
  double r_outer = 6.0;
  std::vector<double> r(imax-1);
  for (int i=0; i<imax-1; ++i) {
    r[i] = 2.0;
  }

  // Creating o-grid
  mesh<double> grid(imax,jmax,r,r_outer);
  
  // Writing to file
  grid.write_mesh("ogrid.tec");

  return 0;
}
