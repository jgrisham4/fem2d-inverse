CXX:=g++
CXXFLAGS:=-std=c++11 -fmax-errors=1 -O 
CPPFLAGS:=-DDEBUG -DVERBOSE
LIBS:=-larmadillo
#INCLUDE:= -I../include -I/usr/include/boost
INCLUDE:= -I../include
DEPS:=$(wildcard ../include/*.h)
SRC_FILES:=$(wildcard *.cpp)
TRG_FILES:=$(patsubst %.cpp,%,$(SRC_FILES))

all: $(TRG_FILES)

%: %.cpp $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(CPPFLAGS) $(INCLUDE) $< $(LIBS)

.PHONY: clean
clean:
	rm -rf $(TRG_FILES) *.tec *.dat
