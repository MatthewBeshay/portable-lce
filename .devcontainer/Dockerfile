FROM ubuntu:24.04 AS clang

ENV DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=20

RUN apt-get update \
    && apt-get install -y \
        wget \
        lsb-release \
        software-properties-common \
        gnupg

RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh ${LLVM_VERSION} \
    && rm llvm.sh

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 \
    --slave /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION}

RUN update-alternatives --install /usr/bin/lld lld /usr/bin/lld-${LLVM_VERSION} 100

RUN apt-get install -y \
        libc++-${LLVM_VERSION}-dev \
        libc++abi-${LLVM_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC="clang" \
    CXX="clang++"

FROM clang AS portablelce

ARG CMAKE_VERSION=3.30.5

RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        ninja-build \
        libsdl2-dev \
        libgl-dev \
        libglm-dev \
        libglu1-mesa-dev \
        libpthread-stubs0-dev \
        libssl-dev \
        gcc-15 g++-15 \
        git \
        curl \
        zip \
        unzip \
        tar \
        pkg-config \
        python3 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 100 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.28+ (older than Ubuntu 24.04's default).
RUN wget -qO cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && bash cmake.sh --skip-license --prefix=/usr/local \
    && rm cmake.sh

# vcpkg in manifest mode, cloned to /opt/vcpkg and pointed to by VCPKG_ROOT.
RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="/opt/vcpkg:${PATH}"
