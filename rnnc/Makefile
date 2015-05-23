MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Object Files
OBJECTFILES= mp.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-m64 -lblas -lcblas -funroll-loops -ffast-math
CXXFLAGS=-m64 -lblas -lcblas -funroll-loops -ffast-math

# Link Libraries and Options
LDLIBSOPTIONS=

build:
# Build Targets
	$(COMPILE.cc) -O3 -std=c++11 -MMD -MP -MF "$@.d" -o mp.o mp.cc
	${LINK.cc} -o ${OBJECTFILES} ${LDLIBSOPTIONS} -lblas -lcblas


# Clean Targets
clean: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/rnnc

