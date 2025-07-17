# Obtain a clean Ubuntu distribution
FROM ubuntu:22.04

# ------------------------------------------------------------------------------- #
# Set the working directory                                                       #
# ------------------------------------------------------------------------------- #
WORKDIR /workspace

# ------------------------------------------------------------------------------- #
# Install first level dependencies                                                #
# ------------------------------------------------------------------------------- #
RUN DEBIAN_FRONTEND="noninteractive" apt-get update 
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata texinfo doxygen texlive-latex-base
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y texlive-latex-extra  texlive-fonts-recommended latexmk  

RUN apt-get install -y git gcc lcov gfortran g++ make cmake flex bison vim python3 python3-pip libtool
RUN apt-get install -y automake autoconf pkg-config wget tar

RUN pip3 install meshio h5py numpy disutils matplotlib sphinx sphinx-rtd-theme

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
                --download-cmake --with-debugging=0 --download-hdf5 --download-triangle                 \

ENV PETSC_ARCH=arch-linux-c-opt
ENV PETSC_DIR=/usr/local/petsc
RUN make all check

# Install Python wrapper
RUN pip3 install src/binding/petsc4py

# Add alias for mpirun command
RUN echo 'alias mpirun=/usr/local/petsc/arch-linux-c-opt/bin/./mpirun' >> ~/.bashrc

# Setup path for petsc
ENV PATH="${PETSC_DIR}/${PETSC_ARCH}/bin:${PATH}"

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
RUN wget http://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.gz
RUN tar -xf binutils-2.42.tar.gz
WORKDIR binutils-2.42
RUN ./configure --prefix=/usr/local/extrae/binutils --enable-shared CFLAGS="-g -O2 -Wno-error" CXXFLAGS="-g -O2 -Wno-error"
RUN make && make install && cp libiberty/pic/libiberty.a /usr/local/extrae/binutils/lib 
        
# Install libxml2
WORKDIR /workspace
RUN git clone --branch v2.13.0 https://gitlab.gnome.org/GNOME/libxml2.git 
WORKDIR libxml2
RUN ./autogen.sh --prefix=/usr/local/extrae/libxml2 --without-python
RUN make && make install 
        
# Install extrae
WORKDIR /workspace
RUN git clone https://github.com/bsc-performance-tools/extrae.git 
WORKDIR extrae
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
RUN ls -l

CMD [ "bash" ]
