#include <iostream>
#include <vector>
#include <cstdlib>
#include <initializer_list>
#include "gdata.h"

/**
 * \file gquad.h
 *
 * This header contains functions for Gaussian quadrature.  The user
 * must supply a function pointer to the gquad function.  This function
 * pointer must take and return the same type as gquad.  This works for
 * 1D to 3D.  The different dimensions are taken care of using function
 * overloading.  That is, the implementation is defined based upon the
 * number of arguments the function pointer takes.  Optional arguments 
 * can also be passed to the integrand via a parameter pack.
 *
 * \author Ashkan Akbariyeh and James Grisham
 * \date 06/20/2015
 */

// 1-D
template <typename T,typename... Tn>
T gquad(int npts, T (*f)(T,Tn...),Tn... args) {

  // Declaring variables
  T sum = (T) 0;
  std::vector<T> gpts(npts);
  std::vector<T> w(npts);

  // Getting points and weights
  gdata<T>(npts,gpts,w);

  // Performing sum
  for (int i=0; i<npts; ++i) {
    sum += w[i]*(*f)(gpts[i],args...);
  }

  return sum;
}

// 2-D
template <typename T,typename... Tn>
T gquad(int npts, T (*f)(T,T,Tn...),Tn... args) {

  // Declaring variables
  T sum = (T) 0;
  std::vector<T> gpts(npts);
  std::vector<T> w(npts);

  // Getting points and weights
  gdata<T>(npts,gpts,w);

  // Performing sum
  for (int i=0; i<npts; ++i) {
    for (int j=0; j<npts; ++j) {
      sum += w[i]*w[j]*(*f)(gpts[i],gpts[j],args...);
    }
  }

  return sum;
}

// 3-D
template <typename T, typename... Tn>
T gquad(int npts, T (*f)(T,T,T,Tn...),Tn... args) {

  // Declaring variables
  T sum = (T) 0;
  std::vector<T> gpts(npts);
  std::vector<T> w(npts);

  // Getting points and weights
  gdata<T>(npts,gpts,w);

  // Performing sum
  for (int i=0; i<npts; ++i) {
    for (int j=0; j<npts; ++j) {
      for (int k=0; k<npts; ++k) {
        sum += w[i]*w[j]*w[k]*(*f)(gpts[i],gpts[j],gpts[k],args...);
      }
    }
  }

  return sum;
}
