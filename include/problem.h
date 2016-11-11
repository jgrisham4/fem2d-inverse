#ifndef PROBLEMHEADERDEF
#define PROBLEMHEADERDEF

#include <string>
#include "mesh.h"

#define D1 1
#define D2 2
#define D3 3
#define GFEM 0
#define LSFEM 1
#define DGFEM 2

template <typename T, const int D, const int DI>
class problem {

  protected:
    void integrate_element(const int npts, const element<T>& elem, arma::Mat<T>& Kelem, arma::Col<T>& Felem);
    void integrate_element(const int npts, const element<T>& elem, T (*f)(T,T), arma::Mat<T>& Kelem, arma::Col<T>& Felem);

  public:
    void discretize(const int imax, const int jmax, const std::vector<T>& r_i, const T r_o);
    void discretize(const std::string mesh_file);
    void discretize(const std::string mesh_file, T (*f)(T,T));
    void solve();
    void write_tecplot(const std::string tec_file) const;
    void apply_bc_dirichlet(const int boundary_index, const int variable_index, T value);
    void apply_bc_dirichlet_normal(const int boundary_index, T value);
    void set_problem_specific_data(const T d);
    void extract_slice(const std::string& slice_file) const;

};

#endif
