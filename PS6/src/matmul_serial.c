#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

typedef float real_t;

// Forward Declaration
void matmul_unopt(real_t * const A, real_t * const B, real_t * const C, size_t N);
void write_matrix(const char * const filename, real_t *matrix, size_t matrix_size);
void read_matrix(const char * const filename, real_t *matrix,  size_t matrix_size);
void print_matrix(real_t * const matrix, size_t matrix_size );


int main(int argc, char *argv[])
{

    size_t MATRIX_SIZE = 128;

    if ( argc > 1 ) {
        MATRIX_SIZE = strtoull(argv[1], NULL, 10);
    }

    printf("Calculating on matrices of sizes %zu x %zu\n", MATRIX_SIZE, MATRIX_SIZE);

    // TODO: Remove unused attribute
    real_t *A, *B, *C;

    A = calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    B = calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    C = calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));

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

    sprintf(filename_A, "data/size%zu-A.bin", MATRIX_SIZE);
    sprintf(filename_B, "data/size%zu-B.bin", MATRIX_SIZE);
    sprintf(filename_C, "data/size%zu-C.bin", MATRIX_SIZE);

    read_matrix(filename_A, A, MATRIX_SIZE);
    read_matrix(filename_B, B, MATRIX_SIZE);

    matmul_unopt(A, B, C, MATRIX_SIZE);

    print_matrix(C, MATRIX_SIZE);
    write_matrix(filename_C, C, MATRIX_SIZE);

    free(A);
    free(B);
    free(C);

    return 0;
}

/**
 * Unoptimized Matrix Multiplication.
 *
 * @param A The first matrix.
 * @param B The second matrix.
 * @param C The resultant matrix.
 * @param N The size of the matrix (N x N).
 */
void matmul_unopt(real_t * const A, real_t * const B, real_t * const C, size_t N)
{

    memset(C, 0, sizeof(real_t) * N * N);

    for ( unsigned int j = 0; j < N; j++ )
    {
        for ( unsigned int i = 0; i < N; i++ )
        {
            for ( unsigned int k = 0; k < N; k++ )
            {
                C[j*N + i] += A[k + j*N] * B[k*N + i];
            }
        }
    }
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
