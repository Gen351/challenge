#pragma once

#include<stdexcept>

#include<vector>

namespace MatrixOp {
    template<typename T>
    struct Vector {
        std::vector<T> data;

        Vector() 
            : data(0) {}

        Vector(size_t size) 
            : data(size) {}
        

        T& operator[](size_t index) {
            return data[index];
        }
        const T& operator[](size_t index) const {
            return data[index];
        }

        size_t size() const { return data.size(); }
    };
    
    template<typename T>
    struct Matrix {
        size_t row;
        size_t col;

        std::vector<T> data;

        Matrix() 
            : row(0), col(0), data() {}

        Matrix(size_t r, size_t c, T init = T{}) 
            : row(r), col(c), data(r * c, init) {}

        
        T* operator[](size_t r) {
            if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
            return &data[r * col];
        }
        const T* operator[](size_t r) const {
            if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
            return &data[r * col];
        }

        T& operator()(size_t r, size_t c) {
            if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
            return data[r * col + c];
        }
        const T& operator()(size_t r, size_t c) const {
            if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
            return data[r * col + c];
        }

        size_t rows() const { return row; }
        size_t cols() const { return col; }
    };

    /* Functions using the structs */
    /// @brief x Matrix
    /// @param v - Multidimentional Vector
    /// @param m - Multidimentional Matrix
    /// @return Vector<T> with Dimensions 1 x M.cols()
    /// @warning v.size() must equal m.rows()
    template<typename T>
    Vector<T> multiply(const Vector<T>& v, const Matrix<T>& M) {
        if(v.size() != M.rows()) { throw std::runtime_error("v x M: Dimensions Mismatch!");}
        
        Vector<T> vxM(M.cols());
        for(size_t i = 0; i < M.cols(); i++) {
            T sum = 0;
            // M[j][i]: safe because (v.size() == M.rows())
            for(size_t j = 0; j < v.size(); j++) sum += v[j] * M[j][i];
            vxM[i] = sum;
        }
        return vxM;
    }

    /// @brief  Matrix x Matrix
    /// @param A - matrix1
    /// @param B - matrix2
    /// @return Matrix<T> with Dimensions A.rows() x B.cols()
    /// @attention Dimensions from: https://i0.wp.com/www.mathbootcamps.com/wp-content/uploads/not-commutative.jpg?resize=520%2C463
    /// @warning optimizatio because of the flat matrix implementation
    /*
        By swapping the j and k loops (known as the IKJ pattern), 
            the innermost loop now moves across the rows of Matrix B. 
            Since the memory is flat, 
            **B[k][0], B[k][1], B[k][2]** are all right next to each other. 
            This makes your CNN significantly faster because the CPU can pre-fetch the data.
    */
    template<typename T>
    Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B) {
        if(A.cols() != B.rows()) { throw std::runtime_error("A x B: Dimensions Mismatch!");}

        Matrix<T> AxB(A.rows(), B.cols(), T{});
        for(size_t i = 0; i < A.rows(); i++) {
            for(size_t k = 0; k < A.cols(); k++) {

                T temp = A[i][k];
                for(size_t j = 0; j < B.cols(); j++) {
                    AxB[i][j] += temp * B[k][j];
                }
            }
        }
        return AxB;
    }

    /// @brief 
    /// @param a - vector a of dimention(d): 1 x n
    /// @param b - vector b of dimention(d): 1 x n
    /// @return T scalar dot products of vector a * b
    template<typename T>
    T dot(const Vector<T>& a, const Vector<T>& b) {
        if(a.size() != b.size()) { throw std::runtime_error("a dot b: Dimensions Mismatch!"); }

        T sum = 0;
        for(size_t i = 0; i < a.size(); i++) sum += a[i] * b[i];
        return sum;
    }

    /// @brief 
    /// @param A - matrix1  
    /// @param B - matrix2
    /// @return T scalar dot products of matrix A * B
    /// @warning ni Gemini for CNNs (Technical bs)
    /*  
        The **"CNN Case"**: Frobenius Inner ProductIn a CNN, 
            when a Kernel (Filter) slides over an Input Image, 
            you take a small patch of the image (**A**) and the kernel (**B**). 
            You multiply every corresponding pixel and sum them all up 
            into one single value for the next layer's pixel.
            
            This is technically called the **Frobenius Inner Product**. 
            If this is what you want your dot function to do, 
            it would look like this:
    */
    template<typename T>
    T dot(const Matrix<T>& A, const Matrix<T>& B) {
        if(A.cols() != B.cols() || A.rows() != B.rows()) { throw std::runtime_error("A dot B: Dimensions Mismatch!"); }
        
        T sum = 0;
        for(size_t i = 0; i < A.data.size(); i++)
            sum += A.data[i] * B.data[i];
        return sum;
    }


    /// @example 
    /*
        A = 
            | 1 2 3 |
            | 4 5 6 |
            | 7 8 9 |

        transpose(A) =
            | 1 4 7 |
            | 2 5 8 |
            | 3 6 9 |
    */
    /// @param A - Matrix (n x m) 
    /// @return A^T - Matrix (m x n)
    template<typename T>
    Matrix<T> transpose(const Matrix<T>& A) {
        
    }
}