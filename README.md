# fem2d-inverse

## Motivation

fem2d-inverse represents an implementation of the Galerkin finite element
method for elliptic problems.  Additionally, an optimization driver is 
included.  The laplace class is used to do the analysis used to form
the objective function.  Matrix operations are handled using the 
Armadillo C++ library.  The user should be aware that Armadillo can
behave differently with different levels of compiler optimization.  
That being said, always use the same optimization flag throughout 
development.

## Features

* Object-orientied implementation of the Galerkin finite element method.
* Statically polymorphic optimization driver class.
* ASCII Tecplot output files.
* Reads UGRID formatted mesh files.
* CMake build files and hand-written makefiles.
