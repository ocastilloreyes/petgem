# Obtain a clean Ubuntu distribution
FROM ubuntu:22.04

# ------------------------------------------------------------------------------- #
# Set the working directory                                                       #
# ------------------------------------------------------------------------------- #
WORKDIR /workspace

# ------------------------------------------------------------------------------- #
# Install first level dependencies                                                #
# ------------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y gcc gfortran g++ make cmake vim 
RUN apt-get update && apt-get install -y flex bison libtool git python3

RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y texinfo 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y doxygen 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y latexmk 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y texlive-latex-base 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y texlive-latex-extra 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y texlive-fonts-recommended  

RUN apt-get update && apt-get install -y python3-breathe
RUN apt-get update && apt-get install -y python3-meshio python3-h5py
RUN apt-get update && apt-get install -y python3-numpy python3-distutils
RUN apt-get update && apt-get install -y python3-pip python3-matplotlib
RUN apt-get update && apt-get install -y python3-sphinx python3-sphinx-rtd-theme
RUN apt-get update && pip3 install cython 

# ------------------------------------------------------------------------------- #
# Install PETSc                                                                   #
# ------------------------------------------------------------------------------- #
RUN mkdir /usr/local/petsc
WORKDIR /usr/local/petsc

# Get PETSc, configure and compile
RUN git clone -b release https://gitlab.com/petsc/petsc.git .

# Obtain new release fixes (since a prior clone or pull)
RUN git pull 

# Configure/compile/check PETSc
RUN ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran --download-mpich --download-fblaslapack \
                --with-scalar-type=complex --download-mumps --download-scalapack --download-ptscotch    \
                --download-cmake --with-debugging=0 --download-hdf5 --download-triangle

# Set PETSc environment variables and PATH
ENV PETSC_ARCH=arch-linux-c-opt \
    PETSC_DIR=/usr/local/petsc
ENV PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:${PATH}"

# Build PETSc
RUN make all check

# Install Python wrapper
RUN pip3 install src/binding/petsc4py

# Add alias for mpirun command
RUN echo 'alias mpirun=\${PETSC_DIR}/\${PETSC_ARCH}/bin/mpirun' >> ~/.bashrc

# ------------------------------------------------------------------------------- #
# Install Extrae and its dependencies                                             #
# ------------------------------------------------------------------------------- #
WORKDIR /workspace
RUN mkdir /usr/local/extrae/
RUN mkdir /usr/local/extrae/libunwind
RUN mkdir /usr/local/extrae/papi
RUN mkdir /usr/local/extrae/libxml2
RUN mkdir /usr/local/extrae/binutils

# Install libunwind
RUN git clone https://github.com/libunwind/libunwind.git 
WORKDIR libunwind
RUN autoreconf -i \
        && ./configure --prefix=/usr/local/extrae/libunwind \
	&& make && make install
	
# Install PAPI (HW counters)
WORKDIR /workspace
RUN git clone https://github.com/icl-utk-edu/papi 
WORKDIR papi/src        
RUN ./configure --prefix=/usr/local/extrae/papi CC=gcc F77=gfortran \
        && make && make install
    
# Install binutils
WORKDIR /workspace
RUN apt-get update && apt-get install -y wget
RUN wget http://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.gz
RUN tar -xf binutils-2.42.tar.gz
WORKDIR binutils-2.42
RUN ./configure --prefix=/usr/local/extrae/binutils --enable-shared CFLAGS="-g -O2 -Wno-error" CXXFLAGS="-g -O2 -Wno-error"
RUN make && make install && cp libiberty/pic/libiberty.a /usr/local/extrae/binutils/lib 
        
# Install libxml2
WORKDIR /workspace
RUN git clone --branch v2.13.0 https://gitlab.gnome.org/GNOME/libxml2.git 
WORKDIR libxml2

RUN apt-get update && apt-get install -y autopoint pkg-config gettext

RUN ./autogen.sh --prefix=/usr/local/extrae/libxml2 --without-python

RUN make && make install 
        
# Install extrae
WORKDIR /workspace
RUN git clone https://github.com/bsc-performance-tools/extrae.git 
WORKDIR extrae
RUN git submodule update --init --recursive
RUN autoreconf -f -i
RUN ./configure --prefix=/usr/local/extrae \
        --with-mpi=/usr/local/petsc/arch-linux-c-opt/ \
        --with-papi=/usr/local/extrae/papi \
        --with-xml-prefix=/usr/local/extrae/libxml2 \
        --with-unwind=/usr/local/extrae/libunwind \
        --with-binutils=/usr/local/extrae/binutils \
        --without-dyninst
RUN make && make install

# ------------------------------------------------------------------------------- #
# Install Unity (Framework for C tests)                                           #
# ------------------------------------------------------------------------------- #
#RUN git clone https://github.com/ThrowTheSwitch/Unity.git /opt/unity

# Define entry point and clean system
WORKDIR /workspace
#RUN rm -rf extrae libunwind libxml2 papi binutils-2.42 binutils-2.42.tar.gz

CMD [ "bash" ]
