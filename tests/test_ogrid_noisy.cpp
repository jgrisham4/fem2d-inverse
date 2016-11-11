#include <random>
#include "mesh.h"

int main() {

  // Generating uniformly distributed noise 
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-0.25,0.25);

  // Inputs for an o-grid
  int imax = 100;
  int jmax = 50;
  double r_outer = 6.0;
  double noise;
  std::vector<double> r(imax-1);
  for (int i=0; i<imax-1; ++i) {
    noise = distribution(generator);
    r[i] = 2.0 + noise;
  }
  
  // Filtering using a moving average
  r[0] = (r[0] + r[1])/2.0;
  r[imax-2] = (r[imax-2] + r[imax-3])/2.0;
  for (int i=1; i<imax-2; ++i) {
    r[i] = (r[i-1] + r[i] + r[i+1])/3.0;
  }

  // Creating o-grid
  mesh<double> grid(imax,jmax,r,r_outer);
  
  // Writing to file
  grid.write_mesh("ogrid_noisy.tec");

  return 0;
}
