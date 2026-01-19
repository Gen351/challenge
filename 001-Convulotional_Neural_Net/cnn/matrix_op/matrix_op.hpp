#include<stdexcept>
#include<time.h>

#include<vector>

#include "matrix.hpp"
#include "vector.hpp"

namespace MatrixOp {

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


    template<typename T>
    Matrix<T> add(const Matrix<T>& A, const Matrix<T> B) {
        if(A.rows() != B.rows() || A.cols() != B.cols()) {
            throw std::runtime_error("A + B: Dimensions Mismatch!");
        }
        Matrix<T> res(A.cols(), A.rows());

        for(size_t i = 0; i < A.data.size(); i++) {
            res.data[i] = A.data[i] + B.data[i]; 
        }

        return res;
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

        B = 
            | 1 2 3 4 5 |
            | 6 7 8 9 1 |
            | 2 3 4 5 6 |
        transpose(B) = 
            | 1 6 2 |
            | 2 7 3 |
            | 3 8 4 |
            | 4 9 5 |
            | 5 1 6 |
    */
    /// @param A - Matrix (n x m) 
    /// @return A^T - Matrix (m x n)
    template<typename T>
    Matrix<T> transpose(const Matrix<T>& A) {
        Matrix<T> transposed(A.cols(), A.rows());

        for(size_t i = 0; i < A.cols(); i++) {
            for(size_t j = 0; j < A.rows(); j++) {
                transposed[i][j] = A[j][i];
            }
        }

        return transposed;
    }

    

    /*============================================
    ==============================================
    >> C  N  N    O  P  E  R  A  T  I  O  N  S  <<
    ==============================================
    ============================================*/
    
    /*
    /// @brief 
    /// @param image 
    /// @param kernel 
    /// @return 
    template<typename T>
    Matrix<T> convolve(const Matrix<T>& image, const Matrix<T>& kernel) {
        if(image.rows() < kernel.rows() || image.cols() < kernel.cols()) {
            throw std::runtime_error("Covolve: Image should be: Image >= (3 x 3)");
        }

        Matrix<T> convolved((image.rows() - kernel.rows()) + 1, (image.cols() - kernel.cols()) + 1);

        for(size_t i = 0; i < convolved.rows(); i++) {
            for(size_t j = 0; j < convolved.cols(); j++) {
                // create patch
                T patchSum = 0;
                for(size_t x = 0; x < kernel.rows(); x++) {
                    for(size_t y = 0; y < kernel.cols(); y++) {
                        patchSum += image[i+x][j+y] * kernel[x][y];
                    }
                }

                convolved[i][j] = patchSum;
            }
        }

        return convolved;
    }
    
    /// @brief Converts a Matrix into a "flat" Vector or a 1-D array
    /// @brief ... using std::vector's assignment operator. 
    /// @param A - Matrix (n x m)
    /// @return V - Vector (n * m)
    template<typename T>
    Vector<T> flatten(const Matrix<T>& A) {
        Vector<T> V;
        V.data = A.data;
        return V;
    }
    */

    float initRandFloat() {    
        return ((float)rand()/(float)RAND_MAX)
                - ((float)rand()/(float)RAND_MAX);
    }
    int initRandInt(int max=RAND_MAX) {
        return rand()/max;
    }
}