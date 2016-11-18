CXX=g++
CXXFLAGS=-std=c++11 -fmax-errors=1 -g
CPPFLAGS=-DDEBUG -DVERBOSE
CPPFLAGS=-DOPT_VERBOSE
LIBS=-larmadillo
INCLUDE= -I../include
DEPS=$(wildcard ../include/*.h)
SRC_FILES=$(wildcard *.cpp)
TRG_FILES=$(patsubst %.cpp,%,$(SRC_FILES))

all: $(TRG_FILES)

%: %.cpp $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(CPPFLAGS) $(INCLUDE) $< $(LIBS)

.PHONY: clean
clean:
	rm -rf $(TRG_FILES)
