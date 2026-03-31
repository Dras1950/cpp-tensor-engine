// src/core/Tensor.hpp
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <algorithm>

template <typename T>
class Tensor {
public:
    // Constructors
    Tensor() : shape_({0}), data_({}) {}

    explicit Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        data_.resize(total_size);
        std::fill(data_.begin(), data_.end(), T{}); // Initialize with default value
    }

    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape), data_(data) {
        size_t total_size = 1;
        for (size_t dim : shape_) {
            total_size *= dim;
        }
        if (data_.size() != total_size) {
            throw std::invalid_argument("Data size does not match tensor shape.");
        }
    }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    const std::vector<T>& data() const { return data_; }

    // Element access (flat index)
    T& operator[](size_t index) {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of bounds.");
        }
        return data_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= data_.size()) {
            throw std::out_of_range("Index out of bounds.");
        }
        return data_[index];
    }

    // Element access (multi-dimensional index)
    T& operator()(const std::vector<size_t>& indices) {
        return data_[get_flat_index(indices)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data_[get_flat_index(indices)];
    }

    // Basic operations
    Tensor<T> operator+(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for addition.");
        }
        Tensor<T> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor<T> operator*(T scalar) const {
        Tensor<T> result(shape_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    // Matrix multiplication (for 2D tensors)
    Tensor<T> matmul(const Tensor<T>& other) const {
        if (shape_.size() != 2 || other.shape_.size() != 2) {
            throw std::invalid_argument("Matmul only supported for 2D tensors.");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }

        size_t rows_a = shape_[0];
        size_t cols_a = shape_[1];
        size_t rows_b = other.shape_[0];
        size_t cols_b = other.shape_[1];

        Tensor<T> result({rows_a, cols_b});

        for (size_t i = 0; i < rows_a; ++i) {
            for (size_t j = 0; j < cols_b; ++j) {
                T sum = T{};
                for (size_t k = 0; k < cols_a; ++k) {
                    sum += (*this)({i, k}) * other({k, j});
                }
                result({i, j}) = sum;
            }
        }
        return result;
    }

    // Print tensor
    void print() const {
        std::cout << "Tensor (Shape: ";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
        }
        std::cout << ")\n";
        // Simple print for 1D/2D tensors
        if (shape_.size() == 1) {
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << data_[i] << " ";
            }
            std::cout << "\n";
        } else if (shape_.size() == 2) {
            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t j = 0; j < shape_[1]; ++j) {
                    std::cout << (*this)({i, j}) << " ";
                }
                std::cout << "\n";
            }
        } else {
            std::cout << "(Complex tensor, printing flat data)\n";
            for (size_t i = 0; i < data_.size(); ++i) {
                std::cout << data_[i] << " ";
            }
            std::cout << "\n";
        }
    }

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;

    size_t get_flat_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices does not match tensor dimensions.");
        }
        size_t flat_index = 0;
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds for dimension " + std::to_string(i));
            }
            flat_index += indices[i] * stride;
            stride *= shape_[i];
        }
        return flat_index;
    }
};

#endif // TENSOR_HPP
