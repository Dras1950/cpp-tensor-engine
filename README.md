# C++ Tensor Engine

A lightweight, header-only C++ library for efficient tensor operations and neural network primitives.

## Features
- **High Performance:** Optimized for speed with efficient memory management and low-level operations.
- **Header-Only:** Easy to integrate into C++ projects without complex build systems.
- **Tensor Operations:** Provides a comprehensive set of tensor manipulation functions (e.g., addition, multiplication, reshaping).
- **Neural Network Primitives:** Includes basic building blocks for neural networks (e.g., activation functions, loss functions).
- **CPU Optimized:** Designed for optimal performance on modern CPUs, with potential for future GPU acceleration.

## Getting Started

### Prerequisites
- C++17 compatible compiler (e.g., GCC, Clang, MSVC)

### Installation

Simply include the `tensor_engine.hpp` header file in your project.

```bash
git clone https://github.com/Dras1950/cpp-tensor-engine.git
cd cpp-tensor-engine
```

### Usage

```cpp
#include "tensor_engine.hpp"
#include <iostream>

int main() {
    Tensor<float> a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> b({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});

    Tensor<float> c = a + b;
    std::cout << "Tensor C:\n" << c << std::endl;

    Tensor<float> d = a.dot(b);
    std::cout << "Tensor D (dot product):\n" << d << std::endl;

    return 0;
}
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
