#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <list>
#include <functional>
#include <numeric>
#include <random>
#include "laplace.h"
#include "opt_driver.h"

/***************************************************************\
 * This code solves a steady-state heat transfer inverse       *
 * problem.  The problem involves an infinitely long pipe. The *
 * inner and outer temperatures of the pipe are known along    *
 * with the thermal conductivity and outer radius.  The normal *
 * heat flux on the outer surface is measured with the goal of *
 * computing the geometry on the inside of the pipe required   *
 * to produce the measured heat flux.  This task is            *
 * accomplished using a gradient-based optimization algorithm. *
 * In this case, either steepest descent or conjugate          *
 * direction was used.  The sensitivity of the objective       *
 * function (L2 norm of difference between target and computed *
 * heat flux) is computed using either the semi-analytic       *
 * method or the semi-analytic complex variable method.        *
 *                                                             *
 * UT Arlington CFDLab                                         *
 * Author: James Grisham                                       *
 * Date  : 10/30/2016                                          *
 *                                                             *
\**************************************************************/

int main() {

  // Inputs for optimization
  unsigned int max_iterations = 150;
  double dr = 1.0e-8;
  double tolerance = 1.0e-3;
  bool use_sacvm = false;

  // Creating a new optimization driver
  opt_driver<laplace<double> > opt;

  // Reading target data
  std::ifstream qtargetfile("qn_target.dat");
  if (!qtargetfile.is_open()) {
    std::cerr << "\nCan't open qn_target.dat." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
    exit(-1);
  }
  std::vector<double> qn_target(opt.get_imax()-1);
  for (int i=0; i<opt.get_imax()-1; ++i) {
    qtargetfile >> qn_target[i];
  }
  qtargetfile.close();
  
  // Setting the initial guess to an ellipse
  double a = 6.0;
  double b = 4.5;
  double dtheta = 2.0*M_PI/(double(opt.get_imax())-1.0);
  auto ellipse = [&a,&b] (double th) { return a*b/sqrt(pow(b*cos(th),2) + pow(a*sin(th),2)); };
  auto circle  = [&a,&b] (double th) { return 5.0; };
  std::list<int> indices(opt.get_imax()-1);
  std::vector<double> r_guess(opt.get_imax()-1);
  std::iota(indices.begin(),indices.end(),0);
  //std::transform(indices.begin(),indices.end(),r_guess.begin(),[&](int i){return circle(double(i)*dtheta); });
  std::transform(indices.begin(),indices.end(),r_guess.begin(),[&](int i){return ellipse(double(i)*dtheta); });

  // Writing initial guess to file
  std::ofstream initial_guess_file("initial_guess");
  for (int i=0; i<opt.get_imax()-1; ++i) {
    initial_guess_file << r_guess[i]*cos(double(i)*dtheta) << " " << r_guess[i]*sin(double(i)*dtheta) << std::endl;
  }
  initial_guess_file.close();

  // Creating a laplace object
  std::cout << "Creating a new problem object. " << std::endl;
  laplace<double>* lp = new laplace<double>();
  opt.set_problem(lp);
  opt.set_qn_target(qn_target);

  // Calling optimization function
  std::cout << "Optimizing." << std::endl;
  std::vector<double> ri_opt = opt.optimize_steepest_descent(r_guess,max_iterations,dr,tolerance,use_sacvm);
  //std::vector<double> ri_opt = opt.optimize_conjugate_direction(r_guess,max_iterations,dr,tolerance,use_sacvm);

  delete lp;

  return 0;

}
