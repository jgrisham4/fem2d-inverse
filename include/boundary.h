/**
 * \class boundary
 *
 * This is a simple container class which holds boundary information.
 *
 * \author James Grisham
 * \date 07/23/2015
 */

#ifndef BOUNDARYHEADERDEF
#define BOUNDARYHEADERDEF

#include <vector>

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

class boundary {

  public:
    boundary() {};
    boundary(int b_marker, std::vector< std::vector<int> >& b_elements);
    std::vector< std::vector<int> > get_boundary_connectivity() const;

  private:
    int boundary_marker;
    std::vector< std::vector<int> > boundary_elements;

};

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

/**
  Constructor used to assign private members
  
  @param[in] b_marker the boundary marker.
  @param[in] b_elements a vector of vectors of the global node numbers of the boundary elements.
*/
boundary::boundary(int b_marker, std::vector< std::vector<int> >& b_elements) : boundary_marker(b_marker), boundary_elements(b_elements) {}

/**
  Method for getting the boundary connectivity

  @return vector of vectors of the boundary elements.
*/
std::vector< std::vector<int> > boundary::get_boundary_connectivity() const {
  return boundary_elements;
}

#endif
