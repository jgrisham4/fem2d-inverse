/**
 * This header contains a class which is used to drive the optimization process.
 * The class is templated based on problem type thereby enabling static 
 * polymorphism.  The problem parameter, P, represents an analysis class 
 * that has certain methods defined (e.g., discretize, solve and so on).
 * Two gradient-based optimization algorithms are implemented: steepest descent
 * and BFGS.  BFGS should converge much more quickly than the steepest descent
 * method.  In order to apply this class to another problem, the objective
 * function must be redefined.  Some things were hardwired for the specific
 * case being solved.
 *
 * @author James Grisham and Ashkan Akbariyeh
 * @date 11/17/2016
 */

#ifndef OPTIMIZATIONDRIVERHEADERDEF
#define OPTIMIZATIONDRIVERHEADERDEF

#include <vector>
#include <functional>
#include <numeric>
#include <iomanip>
#include "laplace.h"

#define tau 0.381966
#define TINNER 373.0
#define TOUTER 283.0
#define RO 10.0
#define cond 10.0
#define IMAX 51
#define JMAX 20

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

template <typename P>
class opt_driver {

  public:
    typedef typename P::VT VT;
    opt_driver() { imax=IMAX; jmax=JMAX; r_o=RO; k=cond; T_inner=TINNER; T_outer=TOUTER; };
    opt_driver(P* prob) : p{prob} { imax=IMAX; jmax=JMAX; r_o=RO; k=cond; T_inner=TINNER; T_outer=TOUTER; };
    opt_driver(P* prob, const std::vector<VT>& q_target) : p{prob}, qn_target{q_target} { imax=IMAX; jmax=JMAX; r_o=RO; k=cond; T_inner=TINNER; T_outer=TOUTER; };
    std::vector<VT> normal_heat_flux() const;
    template <typename U> U f(const arma::Col<U>& T, const mesh<U>& g) const;
    template <typename U> U objective_function(const std::vector<U>& design_var);
    VT compute_sensitivities_fdm(const std::vector<VT>& dr_inner);
    VT compute_sensitivities_cvm(const std::vector<VT>& dr_inner);
    VT compute_sensitivities_safdm(const std::vector<VT>& dr_inner);
    VT compute_sensitivities_sacvm(const std::vector<VT>& dr_inner);
    std::vector<VT> optimize_steepest_descent(const std::vector<VT>& r_guess, const unsigned int max_iter, const VT dr, const VT tol, const bool use_sacvm=false);
    std::vector<VT> optimize_conjugate_direction(const std::vector<VT>& r_guess, const unsigned int max_iter, const VT dr, const VT tol, const bool use_sacvm=false);
    std::vector<VT> optimize_bfgs(const std::vector<VT>& r_guess, const unsigned int max_iter, const VT dr, const VT tol, const bool use_sacvm=false);

    // Simple inline methods
    inline int get_imax() const { return imax; };
    inline int get_jmax() const { return jmax; };
    inline void set_problem(P* prob) { p = prob; };
    inline void set_qn_target(const std::vector<VT>& qn_t) { qn_target = qn_t; };

  private:
    P* p;
    std::vector<VT> Xopt;
    std::vector<VT> qn_target;
    std::vector<VT> dfdr;
    int imax;
    int jmax;
    VT k;
    VT r_o;
    VT T_inner;
    VT T_outer;

};

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

/**
 * Member function for computing the normal heat flux at the outer
 * boundary.
 *
 * @return STL vector which contains the normal heat flux at the outer boundary.
 */
template <typename P>
std::vector<typename opt_driver<P>::VT> opt_driver<P>::normal_heat_flux() const {

  // Computing the normal temperature gradient
  arma::Col<VT> T = p->get_u();
  std::vector<VT> qn_computed(imax-1);
  VT gradT,dr,dx,dy;
  mesh<VT> grid = p->get_grid();
  std::vector<node<VT> > nodes = grid.get_nodes();
  for (int i=0; i<imax-1; ++i) {
    dx = nodes[(i+1)*jmax-1].get_x() - nodes[(i+1)*jmax-2].get_x();
    dy = nodes[(i+1)*jmax-1].get_y() - nodes[(i+1)*jmax-2].get_y();
    dr = sqrt(dx*dx + dy*dy);
    gradT = (T((i+1)*jmax-1) - T((i+1)*jmax-2))/dr;
    qn_computed[i] = -k*gradT;
  }

  return qn_computed;

}

/**
 * This is the objective function when temperature is provided.
 * That is, this function computes the norm of the difference
 * between computed and target normal heat fluxes when the 
 * temperature field is provided.
 *
 * @param[in] T armadillo column vector of temperatures.
 * @param[in] g mesh object used to compute finite differences.
 * @return the value of the objective function.
 */
template <typename P>
template <typename U>
U opt_driver<P>::f(const arma::Col<U>& T, const mesh<U>& g) const {

  // Computing the normal temperature gradient
  static int f_ctr = 0;
  ++f_ctr;
  std::vector<U> qn_computed(imax-1);
  U gradT,dr,dx,dy;
  std::vector<node<U> > nodes = g.get_nodes();
  for (int i=0; i<imax-1; ++i) {
    dx = nodes[(i+1)*jmax-1].get_x() - nodes[(i+1)*jmax-2].get_x();
    dy = nodes[(i+1)*jmax-1].get_y() - nodes[(i+1)*jmax-2].get_y();
    dr = sqrt(dx*dx + dy*dy);
    gradT = (T((i+1)*jmax-1) - T((i+1)*jmax-2))/dr;
    qn_computed[i] = -k*gradT;
  }

  // Computing the objective function
  // In this case, the l^2 norm of the dot(q,n)
  // Integral is computed using the trapezoidal rule
  auto ff = [&](int i){ return pow(qn_computed[i] - qn_target[i],2)/pow(qn_target[i],2); };
  U ds,dtheta;
  U error_norm = U(0.0);
  dtheta = 2.0*M_PI/(U(imax)-1.0);
  ds = r_o*dtheta;
  for (int i=0; i<imax-2; ++i) {
    error_norm += ds/2.0*(ff(i+1)+ff(i));
  }

  return sqrt(error_norm);

}

/**
 * This is the definition of the objective function.  It is 
 * problem specific.
 * 
 * @param[in] design_vars vector of design variables.
 * @return value of the objective function.
 */
template <typename P>
template <typename U>
U opt_driver<P>::objective_function(const std::vector<U>& design_vars) {

  static int obj_call_ctr=0;
  obj_call_ctr++;
  
  // Checking to make sure that the inner radius does not exceed the outer radius
  for (auto var : design_vars) {
    if (abs(var) > r_o) {
      std::cout << "Warning: inner radius larger than outer radius." << std::endl;
      return 1.0e4*pow(var - U(r_o),2);
    }
  }

  // Setting thermal conductivity
  laplace<U> ptmp;
  ptmp.set_problem_specific_data(k);

  // Generating grid and assembling system
  ptmp.discretize(imax,jmax,design_vars,r_o);

  // Applying boundary conditions
  ptmp.apply_bc_dirichlet(0,0,T_inner);
  ptmp.apply_bc_dirichlet(1,0,T_outer);

  // Solving the system
  ptmp.solve();
  arma::Col<U> temperatures = ptmp.get_u();

  U fval = f<U>(temperatures,ptmp.get_grid());

  return fval;

}

/**
 * Member function for computing sensitivities using the finite difference
 * method.  Note: first-order accurate finite differences are used.
 * This should be used for testing only.  It does not produce reliable 
 * results.
 *
 * @param[in] dr_inner vector whose only nonzero value is the value of the perturbation for the i-th variable.
 * @return sensitivity of the objective function wrt the i-th variable.
 */
template <typename P>
typename opt_driver<P>::VT opt_driver<P>::compute_sensitivities_fdm(const std::vector<VT>& dr_inner) {

  // Computing r + dr
  std::vector<VT> rpdr(dr_inner.size());
  std::vector<VT> r = p->get_grid().get_r_inner();
  for (int i=0; i<dr_inner.size(); ++i) {
    rpdr[i] = r[i] + dr_inner[i];
  }

  // Getting dr
  VT dr_i = VT(0);
  for (int i=0; i<dr_inner.size(); ++i) {
    dr_i += dr_inner[i];
  }

  // Evaluating objective function: f(r+dr)
  P ptmp;
  ptmp.set_problem_specific_data(k);
  ptmp.discretize(imax,jmax,rpdr,r_o);
  ptmp.apply_bc_dirichlet(0,0,T_inner);
  ptmp.apply_bc_dirichlet(1,0,T_outer);
  ptmp.solve();
  VT frpdr = f<VT>(ptmp.get_u(),ptmp.get_grid());

  // Returning sensitivity
  // df/dr = ((f(r+dr) - f(r))/dr
  return (frpdr - f<VT>(p->get_u(),p->get_grid()))/dr_i;

}

/**
 * Method for computing the sensitivities using the complex variable method.
 *
 * @param[in] dr_inner vector of real values which represents perturbation.
 */
template <typename P>
typename opt_driver<P>::VT opt_driver<P>::compute_sensitivities_cvm(const std::vector<VT>& dr_inner) {

  // Computing r + dr
  std::vector<std::complex<VT> > rpdr(dr_inner.size());
  std::vector<VT> r = p->get_grid().get_r_inner();
  for (int i=0; i<dr_inner.size(); ++i) {
    rpdr[i] = std::complex<VT>(r[i],dr_inner[i]);
  }

  // Getting dr
  VT dr_i = VT(0);
  for (int i=0; i<dr_inner.size(); ++i) {
    dr_i += dr_inner[i];
  }

  // Evaluating objective function: f(r+dr)
  return (objective_function<std::complex<VT> >(rpdr)).imag()/dr_i;

}

/**
 * Member function for computing the sensitivities using the semi-analytic
 * method.  
 *
 * @param[in] dr_inner real perturbation in the inner radius (one non-zero value in the i-th place).
 * @return df/dr_i - sensitivity of the objective wrt the i-th design variable.
 */
template <typename P>
typename opt_driver<P>::VT opt_driver<P>::compute_sensitivities_safdm(const std::vector<VT>& dr_inner) {

  // Finding the step size and finding the perturbed radius
  mesh<VT> gtmp = p->get_grid();
  std::vector<VT> r_original = gtmp.get_r_inner();
  std::transform(dr_inner.begin(),dr_inner.end(),r_original.begin(),r_original.begin(),std::plus<VT>());
  VT dr_i = std::accumulate(dr_inner.begin(),dr_inner.end(),VT(0));  // assuming one nonzero entry
  std::vector<VT> r_perturbed = r_original;

  // Creating a new problem object to manipulate
  P p_pdr,p_mdr;

  // Setting thermal conductivity
  p_pdr.set_problem_specific_data(k);
  p_mdr.set_problem_specific_data(k);

  // Assembling K(x+dx)
  p_pdr.discretize(imax,jmax,r_perturbed,r_o);
  p_pdr.apply_bc_dirichlet(0,0,T_inner);
  p_pdr.apply_bc_dirichlet(1,0,T_outer);
  arma::Mat<VT> K_xpdx = p_pdr.get_K();
  arma::Col<VT> f_xpdx = p_pdr.get_f();

  // Assembling K(x-dx)
  r_original = (p->get_grid()).get_r_inner();
  for (int i=0; i<dr_inner.size(); ++i) {
    r_perturbed[i] = r_original[i] - dr_inner[i];
  }
  p_mdr.discretize(imax,jmax,r_perturbed,r_o);
  p_mdr.apply_bc_dirichlet(0,0,T_inner);
  p_mdr.apply_bc_dirichlet(1,0,T_outer);
  arma::Mat<VT> K_xmdx = p_mdr.get_K();
  arma::Col<VT> f_xmdx = p_mdr.get_f();

  // Computing Delta K = (K(x+dx) - K(x-dx))/2 
  arma::Mat<VT> dK = 0.5*(K_xpdx - K_xmdx);
  arma::Col<VT> df = 0.5*(f_xpdx - f_xmdx);

  // Applying boundary conditions
  int nnodes = p_pdr.get_grid().get_num_nodes();
  arma::Col<VT> du(nnodes);

  // Computing du
  //du = Ki*(df - dK*(p->get_u()));
  arma::solve(du,p->get_K(),df - dK*(p->get_u()));

  // Computing df/dx = (f(u + Delta u) - f(u - Delta u))/(2 Delta u) (Delta u / Delta x)
  // which can be simplified to df/dx = (f(u+du) - f(u-du))/(2 dx)
  arma::Col<VT> updu = p->get_u() + du;
  arma::Col<VT> umdu = p->get_u() - du;
  VT fupdu = f<VT>(updu,p_pdr.get_grid());
  VT fumdu = f<VT>(umdu,p_mdr.get_grid());
  VT dfdr_i = (fupdu - fumdu)/(2.0*dr_i);

  return dfdr_i;

}

/**
 * Member function for computing the sensitivities using the semi-analytic
 * complex variable method.
 *
 * @param[in] dr_inner real perturbation in the inner radius (one non-zero value).
 * @return df/dr_i - sensitivity of the objective wrt the i-th design variable.
 */
template <typename P>
typename opt_driver<P>::VT opt_driver<P>::compute_sensitivities_sacvm(const std::vector<VT>& dr_inner) {

  // Creating a new problem
  laplace<VT,std::complex<VT> > p_perturbed;
  p_perturbed.set_problem_specific_data(k);

  // Creating complex vector of original design variables with complex perturbation added in
  std::vector<std::complex<VT> > rpdr(dr_inner.size());
  std::vector<VT> r_inner = p->get_grid().get_r_inner();
  VT dr_i = VT(0.0);
  for (int i=0; i<dr_inner.size(); ++i) {
    rpdr[i] = std::complex<VT>(r_inner[i],dr_inner[i]);
    dr_i += dr_inner[i];
  }

  // Discretizing
  p_perturbed.discretize_perturbed(imax,jmax,rpdr,r_o);

  // Applying boundary conditions
  // IMPORTANT NOTE: The below application of boundary conditions to the imaginary part
  // of the perturbation in K and f is equivalent to zeroing out the rows for the nodes
  // which are on the boundary.  This is a statement of the fact that the temperature
  // solution is not sensitive on boundary nodes because there is a Dirichlet boundary
  // condition there.  What about Neumann?  
  // More specifically, dK * u = df for boundary nodes, assuming that you solved for u
  // properly in the first place
  p_perturbed.apply_bc_dirichlet(0,0,T_inner);
  p_perturbed.apply_bc_dirichlet(1,0,T_outer);

  // Finding dK
  arma::Mat<VT> dK = p_perturbed.get_K();
  arma::Col<VT> df = p_perturbed.get_f();

  // Finding du
  // K u = f
  // K du + dK u = df
  // du = K^(-1) ( df - dK u )
  //arma::Col<VT> du = (p->get_Kinv())*(df - dK*(p->get_u())) ;  // this is bad...
  arma::Col<VT> du(df.n_elem); 
  arma::solve(du,p->get_K(),df - dK*(p->get_u()));

  // Finding df/dx
  arma::Col<std::complex<VT> > updu(p->get_u(),du);
  return (f<std::complex<VT> >(updu,p_perturbed.get_grid())).imag()/dr_i;
  
}

/**
 * Method for performing the optimization using steepest descent.
 *
 * @param[in] r_guess vector which contains initial guess for design variables.
 * @param[in] max_iter max number of iterations allowed.
 * @param[in] dr step size used to compute sensitivities.
 * @param[in] tol tolerance used to measure convergence of the optimization.
 * @param[in] use_sacvm boolean variable which indicates use of the SACVM method.  It uses the SAM method if false.
 * @return an STL vector of the optimum design variables.
 */
template <typename P>
std::vector<typename opt_driver<P>::VT> opt_driver<P>::optimize_steepest_descent(const std::vector<VT>& r_guess, const unsigned int max_iter, const VT dr, const VT tol, const bool use_sacvm) {

  // Declaring variables
  std::vector<VT> X(r_guess.size());
  std::vector<VT> S(r_guess.size());
  std::vector<VT> dX(r_guess.size());
  std::vector<VT> Fhist;
  VT alpha_opt = 1.0;
  VT x1d_opt, xl, xu, x1, x2, fl, fu, f1, f2;
  VT aa,xmax;
  VT eps = tol/1.0;
  unsigned int K;
  int N;
  VT F;
  X = r_guess;
  F = objective_function<VT>(r_guess);
  Fhist.push_back(F);
  xmax = 10.0;
  
  // Writing initial guess data to file
  p->set_problem_specific_data(k);
  p->discretize(imax,jmax,X,r_o);
  p->apply_bc_dirichlet(0,0,T_inner);
  p->apply_bc_dirichlet(1,0,T_outer);
  p->solve();
  p->write_tecplot("initial.tec");
  
  // Lambda functions for 1D search
  auto project = [&X,&S] (VT alpha) {std::vector<VT> Xn(X.size(),VT{0.0}); for (int i=0; i<X.size(); ++i) Xn[i]=X[i] + alpha*S[i]; return Xn; };   // this returns the X which corresponds to X + alpha*S
  auto one_d_fun = [&] (VT alpha) { std::vector<VT> Xnew = project(alpha); return objective_function<VT>(Xnew); };

  // Iterating
  for (unsigned int i=0; i<max_iter; ++i) {

    // Finding gradient
    if (use_sacvm) {

      // Computing sensitivities using the semi-analytic complex variable method
      std::cout << "Computing sensitivity for variable: " << std::endl;
      for (unsigned int j=0; j<imax-1; ++j) {

        std::cout << std::setw(3);
        std::cout << j << " " << std::flush;
        if ((j+1)%20==0) {
          std::cout << std::endl;
        }

        // Setting up vector for step size
        for (int jj=0; jj<dX.size(); ++jj) {
          if (jj==j) {
            dX[jj] = dr;
          }
          else {
            dX[jj] = VT(0);
          }
        }

        // Calling member function to compute the sensitivity
        S[j] = VT(-1)*compute_sensitivities_sacvm(dX);

      }
      std::cout << std::endl;

    }
    else {

      // Computing sensitivities using the semi-analytic finite difference method
      std::cout << "Computing sensitivity for variable: " << std::endl;
      for (unsigned int j=0; j<imax-1; ++j) {

        std::cout << std::setw(3);
        std::cout << j << " " << std::flush;
        if ((j+1)%20==0) {
          std::cout << std::endl;
        }

        // Setting up vector for step size
        for (int jj=0; jj<dX.size(); ++jj) {
          if (jj==j) {
            dX[jj] = dr;
          }
          else {
            dX[jj] = VT(0);
          }
        }

        // Calling member function to compute the sensitivity
        S[j] = VT(-1)*compute_sensitivities_safdm(dX);

      }
      std::cout << std::endl;

    }
    
    // Adaptively choosing the upper bound on the 1D search
    xu = 2.0*alpha_opt;
    xl = 0.0;
    fu = one_d_fun(xu);
    fl = one_d_fun(xl);

    // Performing 1D search to find minimum along the direction of steepest descent
    std::cout << "Starting 1D search on iteration " << i << std::endl;
    K = 3;
    x1 = (1.0 - tau)*xl + tau*xu;
    x2 = tau*xl + (1.0 - tau)*xu;
    f1 = one_d_fun(x1);
    f2 = one_d_fun(x2);
    N = (int) (ceil(log(eps)/(log(1.0 - tau)) + 3.0));
    while (K<N) {
      ++K;

      if (f1>f2) {
        xl = x1;
        fl = f1;
        x1 = x2;
        f1 = f2;
        x2 = tau*xl + (1.0 - tau)*xu;
        f2 = one_d_fun(x2);
        alpha_opt = x2;
      }
      else {
        xu = x2;
        fu = f2;
        x2 = x1;
        f2 = f1;
        x1 = (1.0 - tau)*xl + tau*xu;
        f1 = one_d_fun(x1);
        alpha_opt = x1;
      }

    }

    // Updating X
    for (unsigned int j=0; j<X.size(); ++j) {
      X[j] += alpha_opt*S[j];
    }
    F = objective_function(X);
    p->set_problem_specific_data(k);
    p->discretize(imax,jmax,X,r_o);
    p->apply_bc_dirichlet(0,0,T_inner);
    p->apply_bc_dirichlet(1,0,T_outer);
    p->solve();
    Fhist.push_back(F);
    printf("Objective function value: %1.6e\n",F);

    // Checking tolerance
    if (fabs(F)<tol) {
      std::cout << "Steepest descent complete." << std::endl;
      break;
    }
  }

  // Writing final result to file
  p->write_tecplot("final.tec");

  return X;

}

/**
 * Method for performing the optimization using the BFGS method.
 *
 * @param[in] r_guess vector which contains initial guess for design variables.
 * @param[in] max_iter max number of iterations allowed.
 * @param[in] dr step size used to compute sensitivities.
 * @param[in] tol tolerance used to measure convergence of the optimization.
 * @param[in] use_sacvm boolean variable which indicates use of the SACVM method.  It uses the SAM method if false.
 * @return an STL vector of the optimum design variables.
 */
template <typename P>
std::vector<typename opt_driver<P>::VT> opt_driver<P>::optimize_bfgs(const std::vector<VT>& r_guess, const unsigned int max_iter, const VT dr, const VT tol, const bool use_sacvm) {

  // Declaring variables
  std::vector<VT> X(r_guess.size());
  std::vector<VT> S(r_guess.size());
  std::vector<VT> Sqm1(r_guess.size());
  arma::Col<VT> Sarma(r_guess.size());
  std::vector<VT> dX(r_guess.size());
  std::vector<VT> Fhist;
  std::vector<std::vector<VT> > Xhist;
  std::vector<VT> gradF(r_guess.size());
  std::vector<std::vector<VT> > gradHist;
  arma::Mat<VT> Hq(r_guess.size(),r_guess.size(),arma::fill::eye);
  arma::Mat<VT> Hqp1(r_guess.size(),r_guess.size(),arma::fill::eye);
  arma::Mat<VT> D(r_guess.size(),r_guess.size());
  arma::Col<VT> pvec,y;
  VT s,t;
  VT alpha_opt = VT(2.0);
  VT x1d_opt, xl, xu, x1, x2, fl, fu, f1, f2;
  VT eps = tol/1.0;
  VT a,b,beta,slope;
  unsigned int K;
  int N;
  VT F;
  X = r_guess;
  F = objective_function<VT>(r_guess);
  Fhist.push_back(F);
  
  // Writing initial guess data to file
  Xhist.push_back(X);
  p->set_problem_specific_data(k);
  p->discretize(imax,jmax,X,r_o);
  p->apply_bc_dirichlet(0,0,T_inner);
  p->apply_bc_dirichlet(1,0,T_outer);
  p->solve();
  p->write_tecplot("initial.tec");
  
  // Lambda functions for 1D search
  auto project = [&X,&S] (VT alpha) {std::vector<VT> Xn; for (int i=0; i<X.size(); ++i) Xn.push_back(X[i] + alpha*S[i]); return Xn;};   // this returns the X which corresponds to X + alpha*S
  auto one_d_fun = [&] (VT alpha) { std::vector<VT> Xnew = project(alpha); return objective_function<VT>(Xnew); };

  // Iterating
  for (unsigned int i=0; i<max_iter; ++i) {

    // Finding gradient using either the SAM or SACVM
    if (use_sacvm) {

      // Computing sensitivities using the semi-analytic complex variable method
      std::cout << "Computing sensitivity for variable: " << std::endl;
      for (unsigned int j=0; j<imax-1; ++j) {

        std::cout << std::setw(3);
        std::cout << j << " " << std::flush;
        if ((j+1)%20==0) {
          std::cout << std::endl;
        }

        // Setting up vector for step size
        for (int jj=0; jj<dX.size(); ++jj) {
          if (jj==j) {
            dX[jj] = dr;
          }
          else {
            dX[jj] = VT(0);
          }
        }

        // Calling member function to compute the sensitivity
        gradF[j] = compute_sensitivities_sacvm(dX);
      }
      std::cout << std::endl;

    }
    else {

      // Computing sensitivities using the semi-analytic finite difference method
      std::cout << "Computing sensitivity for variable: " << std::endl;
      for (unsigned int j=0; j<imax-1; ++j) {

        std::cout << std::setw(3);
        std::cout << j << " " << std::flush;
        if ((j+1)%20==0) {
          std::cout << std::endl;
        }

        // Setting up vector for step size
        for (int jj=0; jj<dX.size(); ++jj) {
          if (jj==j) {
            dX[jj] = dr;
          }
          else {
            dX[jj] = VT(0);
          }
        }

        // Calling member function to compute the sensitivity
        gradF[j] = compute_sensitivities_safdm(dX);

      }
      std::cout << std::endl;

    }

    // Adding gradient to history
    gradHist.push_back(gradF);

    // Computing the proper search direction
    // This is the guts of the BFGS method.  It computes an
    // approximation for the inverse of the Hessian matrix.
    if (i!=0) {
      pvec = arma::Col<VT>(Xhist.back()) - arma::Col<VT>(Xhist[Xhist.size()-2]);
      y = arma::Col<VT>(gradHist.back()) - arma::Col<VT>(gradHist[gradHist.size()-2]);
      s = arma::dot(pvec,y);
      t = arma::dot(y,Hq*y);
      D = (s + t)/(s*s)*(pvec*pvec.t()) - VT(1.0)/s*(Hq*y*pvec.t() + pvec*(Hq*y).t());
      Hqp1 = Hq + D;
      Sarma = -Hqp1*arma::Col<VT>(gradF);
      for (int k=0; k<Sarma.n_elem; ++k) {
        S[k] = Sarma[k];
      }
      Hq = Hqp1;
    }
    else {
      for (int k=0; k<gradF.size(); ++k) {
        S[k] = -gradF[k];
      }
    }


    // Performing 1D search to find minimum along the direction of steepest descent
    std::cout << "Starting 1D search on iteration " << i << std::endl;
    K = 3;
    xu = 2.0*alpha_opt;
    xl = 0.0;
    fl = one_d_fun(xl);
    x1 = (1.0 - tau)*xl + tau*xu;
    x2 = tau*xl + (1.0 - tau)*xu;
    f1 = one_d_fun(x1);
    f2 = one_d_fun(x2);
    N = (int) (ceil(log(eps)/(log(1.0 - tau)) + 3.0));
    while (K<N) {
      ++K;

      if (f1>f2) {
        xl = x1;
        fl = f1;
        x1 = x2;
        f1 = f2;
        x2 = tau*xl + (1.0 - tau)*xu;
        f2 = one_d_fun(x2);
        alpha_opt = x2;
      }
      else {
        xu = x2;
        fu = f2;
        x2 = x1;
        f2 = f1;
        x1 = (1.0 - tau)*xl + tau*xu;
        f1 = one_d_fun(x1);
        alpha_opt = x1;
      }

    }

    // Updating the design variables, X
    for (unsigned int j=0; j<X.size(); ++j) {
      X[j] += alpha_opt*S[j];
    }
    Xhist.push_back(X);
    p->set_problem_specific_data(k);
    p->discretize(imax,jmax,X,r_o);
    p->apply_bc_dirichlet(0,0,T_inner);
    p->apply_bc_dirichlet(1,0,T_outer);
    p->solve();
    F = objective_function<VT>(X);
    printf("Objective function value: %1.6e\n",F);
    Fhist.push_back(F);

    // Checking tolerance
    if (fabs(F)<tol) {
      std::cout << "BFGS complete." << std::endl;
      break;
    }
  }

  // Writing final result to file
  p->write_tecplot("final.tec");

  return X;

}


#endif
