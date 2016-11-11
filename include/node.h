/**
 * \file node.h
 * \class node
 *
 * This is a simple node class that holds coordinates for the nodes.
 * It has methods for getting the coordinates.  The coordinates can 
 * only be set through the constructor.  It is templated based on 
 * type.
 *
 * \author James Grisham
 * \date 05/02/2015
 */

#ifndef NODEHEADERDEF
#define NODEHEADERDEF

/***************************************************************\
 * Class definition                                            *
\***************************************************************/

template <typename T>
class node {
  
  public:
		node() {
      x = (T) 0;
      y = (T) 0;
      f = (T) 0;
    };
    node(const T& xpt, const T& ypt) : x{xpt}, y{ypt} {};
    node(const T& xpt, const T& ypt, const T& fval) : x{xpt}, y{ypt}, f{fval} {};
    T get_x() const;
    T get_y() const;
    T get_f() const;

  private:
    T x;  /**< x-coordinate of node. */
    T y;  /**< y-coordinate of node. */
    T f;  /**< function value at node. */

};

/***************************************************************\
 * Class implementation                                        *
\***************************************************************/

/**
  Method for getting the x-coordinate.

  @returns the x-coordinate.
  */
template <typename T> T node<T>::get_x() const {
  return x;
}

/**
  Method for getting the y-coordinate.

  @returns the y-coordinate.
  */
template <typename T> T node<T>::get_y() const {
  return y;
}

/**
  Method for getting the function value at the node.

  @returns the function value.
  */
template <typename T> T node<T>::get_f() const {
  return f;
}

#endif
