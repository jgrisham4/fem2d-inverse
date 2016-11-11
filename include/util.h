#ifndef UTILHEADERDEF
#define UTILHEADERDEF

#include <cstdlib>
#include <iostream>
#include <vector>

namespace tnsr {

template <typename T>
T dotprod(const std::vector<T>& v1, const std::vector<T>& v2) {
  T sum = (T) 0;
  for (unsigned int i=0; i<v1.size(); ++i) {
    sum += v1[i]*v2[i];
  }
  return sum;
}

template <typename T>
std::vector<T> crossprod(const std::vector<T>& b, const std::vector<T>& c) {
  std::vector<T> v(b.size());
  v[0] = -b[2]*c[1] + b[1]*c[2];
  v[1] = b[2]*c[0] - b[0]*c[2];
  v[2] = -b[1]*c[0] + b[0]*c[1];
  return v;
}

// Levi-cevita symbol (permutation tensor)
template <typename T>
T levi(const int i, const int j, const int k) {
  if (i>2||j>2||k>2) {
    std::cerr << "\nERROR: Indices in levi civita symbol invalid:" << std::endl;
    std::cerr << "(i,j,k) = (" << i << "," << j << "," << k << ")." << std::endl;
    std::cerr << "Exiting.\n" << std::endl;
    exit(-1);
  }
  std::vector<T> v1(3),v2(3),v3(3);
  v1[i] = 1.0;
  v2[j] = 1.0;
  v3[k] = 1.0;
  return dotprod<T>(v1,crossprod<T>(v2,v3));
}

}

#endif 
