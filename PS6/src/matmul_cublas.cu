#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

// Cublas headers
#include <cuda_runtime.h>
#include "cublas_v2.h"

// column major indexing of arrays
//#define IDX2C(i,j,ld) (((j)*(ld))+(i))

typedef float real_t;

// Forward Declaration
void matmul_unopt(real_t * const A, real_t * const B, real_t * const C, size_t N);
void write_matrix(const char * const filename, real_t *matrix, size_t matrix_size);
void read_matrix(const char * const filename, real_t *matrix,  size_t matrix_size);
void print_matrix(real_t * const matrix, size_t matrix_size );



int main(int argc, char *argv[])
{
    // Defining and initialize cublas handle.
    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t MATRIX_SIZE = 128;

    if ( argc > 1 ) {
        MATRIX_SIZE = strtoull(argv[1], NULL, 10);
    }

    printf("Calculating on matrices of sizes %zu x %zu\n", MATRIX_SIZE, MATRIX_SIZE);

    // TODO: Remove unused attribute
    real_t *A, *B, *C;
    real_t *d_A, *d_B, *d_C;

    A = (real_t *) calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    B = (real_t *) calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    C = (real_t *) calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));

    cudaMalloc(&d_A, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t));
    cudaMalloc(&d_B, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t));
    cudaMalloc(&d_C, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t));

    

    // ------------------------------------------------------------
    // Construct file path strings
    // ------------------------------------------------------------
    const size_t stringBufSize = 30;

    char filename_A[stringBufSize];
    char filename_B[stringBufSize];
    char filename_C[stringBufSize];

    memset(filename_A, '\0', stringBufSize * sizeof(char));
    memset(filename_B, '\0', stringBufSize * sizeof(char));
    memset(filename_C, '\0', stringBufSize * sizeof(char));

    // IMPORTANT: Do not change these
    sprintf(filename_A, "data/size%zu-A.bin", MATRIX_SIZE);
    sprintf(filename_B, "data/size%zu-B.bin", MATRIX_SIZE);
    sprintf(filename_C, "data/size%zu-C-cublas.bin", MATRIX_SIZE);

    read_matrix(filename_A, A, MATRIX_SIZE);
    read_matrix(filename_B, B, MATRIX_SIZE);

    // Define cuda events
    cudaEvent_t start, end;
    float time;

    // Initalize cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Transfer matrix values to device buffer
    cublasSetMatrix(MATRIX_SIZE,MATRIX_SIZE, sizeof(*A), A, MATRIX_SIZE, d_A, MATRIX_SIZE);
    cublasSetMatrix(MATRIX_SIZE,MATRIX_SIZE, sizeof(*B), B, MATRIX_SIZE, d_B, MATRIX_SIZE);
    //cublasSetMatrix(MATRIX_SIZE,MATRIX_SIZE, sizeof(*C), C, MATRIX_SIZE, d_C, MATRIX_SIZE);



    float alpha = 1.0f;
    float beta = 0.0f;

    // Start recording of event
    cudaEventRecord(start, 0);

    // General matrix-matrix multiplication
    // C = alpha * A @ B + beta * C
    // Note:
    //  Since cuBLAS is column major, we can switch the order of multiplication to B@A,
    //  to get the correct product.

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE,
                &alpha, d_B, MATRIX_SIZE, d_A, MATRIX_SIZE, &beta, d_C, MATRIX_SIZE );

    // Record end
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&time, start, end);
    printf("Elapsed time for size %ld : %f ms \n ", MATRIX_SIZE, time);

    // Retrieve device buffer of result matrix
    cublasGetMatrix(MATRIX_SIZE,MATRIX_SIZE, sizeof(*C), d_C, MATRIX_SIZE, C, MATRIX_SIZE);

    write_matrix(filename_C, C, MATRIX_SIZE);

    free(A);
    free(B);
    free(C);
    
    // Free device buffers
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free handle object.
    cublasDestroy(handle);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}

/**
 * Read a matrix from disk.
 *
 * @param filename The location of the binary file to read from.
 * @param matrix A pointer to where the matrix is allocted.
 * @param matrix_size The size N of a matrix of size N x N.
 */
void read_matrix(const char * const filename, real_t *matrix, size_t matrix_size)
{
    FILE *fh = fopen(filename, "rb");

    if ( ! fh ) {
        fprintf(stderr, "Failed to open file %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }

    unsigned long res = fread((void *) matrix, sizeof(real_t), matrix_size * matrix_size, fh);
    (void) res;

    fclose(fh);
}

/**
 * Writes the matrix out to disk.
 *
 * @param filename The relative path to where the matrix should be written.
 * @param matrix A pointer to the matrix.
 * @param matrix_size The size (N) of the matrix (N x N).
 */
void write_matrix(const char * const filename, real_t *matrix, size_t matrix_size)
{
    FILE *fh = fopen(filename, "wb");

    if ( ! fh ) {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fwrite(matrix, sizeof(real_t), matrix_size * matrix_size, fh);

    fclose(fh);
}

/**
 * Prints the matrix.
 *
 * @param matrix A pointer to the matrix.
 * @param matrix_size The size (N) of the matrix (N x N).
 */
void print_matrix(real_t * const matrix, size_t matrix_size )
{
    for ( size_t j = 0; j < matrix_size; j++ ) {
        for ( size_t i = 0; i < matrix_size; i++ ) {
            printf("%lf\t", matrix[i + j*matrix_size]);
        }
        printf("\n");
    }
}
