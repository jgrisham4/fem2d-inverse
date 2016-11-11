#include <random>
#include <iostream>
#include <fstream>
#include "laplace.h"
#include "opt_driver.h"

/***************************************************************\
 * This code exists for the purpose of generating initial heat * 
 * flux data along the outer boundary of an o-grid for a       *
 * pseudo-random inner radius geometry.                        *
 *                                                             *
 * UT Arlington CFDLab                                         *
 * Author: James Grisham                                       *
 * Date  : 10/30/2016                                          *
 *                                                             *
\**************************************************************/

int main() {

  // Generating uniformly distributed noise 
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-0.5,0.5);
    //noise = distribution(generator);

  // Inputs for an o-grid
  int imax = 51;
  int jmax = 20;
  double r_outer = 10.0;
  double noise;
  std::vector<double> r(imax-1);
  for (int i=0; i<imax-1; ++i) {
    noise = distribution(generator);
    r[i] = 5.0 + noise;
  }
  
  // Filtering using a moving average
  r[0] = (r[0] + r[1])/2.0;
  r[imax-2] = (r[imax-2] + r[imax-3])/2.0;
  for (int i=1; i<imax-2; ++i) {
    r[i] = (r[i-1] + r[i] + r[i+1])/3.0;
  }

  // Writing initial inner geometry
  double dtheta = 2.0*M_PI/(double(imax)-1.0);
  std::ofstream ig_file("target_geometry");
  ig_file.setf(std::ios_base::scientific);
  ig_file.precision(16);
  for (int i=0; i<r.size(); ++i) {
    ig_file << r[i]*cos(double(i)*dtheta) << " " << r[i]*sin(double(i)*dtheta) << std::endl;
  }
  ig_file.close();

  // Solving for the temperature field
  laplace<double> p;
  p.set_problem_specific_data(10.0);   // k       = 10 W/(m-K)
  p.discretize(imax,jmax,r,r_outer);
  p.apply_bc_dirichlet(0,0,373.0);     // T_inner = 100 deg C
  p.apply_bc_dirichlet(1,0,283.0);     // T_outer =  10 deg C
  p.solve();
  p.write_tecplot("T_target.tec");

  // Computing the normal heat flux
  opt_driver<laplace<double> > opt(&p);
  std::cout << "Computing normal heat flux." << std::endl;
  std::vector<double> qn = opt.normal_heat_flux();
  std::cout << "Done computing normal heat flux." << std::endl;

  // Writing data to file to be used as the target heat flux
  std::ofstream outfile("qn_target.dat");
  outfile.setf(std::ios_base::scientific);
  outfile.precision(16);
  for (int i=0; i<qn.size(); ++i) {
    outfile << qn[i] << "\n";
  }
  outfile.close();

  return 0;

}
