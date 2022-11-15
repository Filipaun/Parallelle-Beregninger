#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

#include <cuda.h>

typedef float real_t;

// Forward Declaration
void __global__ matmul_unopt(real_t * const A, real_t * const B, real_t * const C, size_t N);
void write_matrix(const char * const filename, real_t *matrix, size_t matrix_size);
void read_matrix(const char * const filename, real_t *matrix,  size_t matrix_size);
void print_matrix(real_t * const matrix, size_t matrix_size );


// Simple selection of grid and block size 
void get_dim(size_t matrix_size, dim3 *grid_dim, dim3 *block_dim)
{
    if (matrix_size <= 32)
    {
        // All matrix can fit inside one block
        grid_dim -> x = 1;
        grid_dim -> y = 1;

        block_dim -> x = matrix_size;
        block_dim -> y = matrix_size;
    }
    else
    {
        // If (N)**2 > 1024, we make a sufficient number of blocks to cover the entire grid_dim with
        // 32x32 thread per block.
        grid_dim -> x = matrix_size/32 + ( matrix_size % 32 != 0);
        grid_dim -> y = matrix_size/32 + ( matrix_size % 32 != 0);

        block_dim -> x = 32;
        block_dim -> y = 32;
    }
}

int main(int argc, char *argv[])
{

    size_t MATRIX_SIZE = 128;

    if ( argc > 1 ) {
        MATRIX_SIZE = strtoull(argv[1], NULL, 10);
    }

    printf("Calculating on matrices of sizes %zu x %zu\n", MATRIX_SIZE, MATRIX_SIZE);

    // TODO: Remove unused attribute
    real_t *A, *B, *C;
    real_t *d_A, *d_B, *d_C;


    dim3 grid_dim,block_dim;

    get_dim(MATRIX_SIZE, &grid_dim, &block_dim);


    A = (real_t *)calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    B = (real_t *)calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));
    C = (real_t *)calloc(MATRIX_SIZE*MATRIX_SIZE, sizeof(real_t));

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

    // IMPORTANT: Do not change this!
    sprintf(filename_A, "data/size%zu-A.bin", MATRIX_SIZE);
    sprintf(filename_B, "data/size%zu-B.bin", MATRIX_SIZE);
    sprintf(filename_C, "data/size%zu-C-cuda.bin", MATRIX_SIZE);

    read_matrix(filename_A, A, MATRIX_SIZE);
    read_matrix(filename_B, B, MATRIX_SIZE);

    // Define cuda events
    cudaEvent_t start, end;
    float time;

    // Initalize cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&end);


    cudaMemcpy(d_A, A, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t), cudaMemcpyHostToDevice);


    // Start recording of event
    cudaEventRecord(start, 0);

    // Matrix multiplication kernel
    matmul_unopt<<<grid_dim,block_dim>>>(d_A, d_B, d_C, MATRIX_SIZE);

    // Record end
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&time, start, end);
    printf("Elapsed time for size %ld : %f ms \n ", MATRIX_SIZE, time);

    // print_matrix(C, MATRIX_SIZE);
    cudaMemcpy(C, d_C, MATRIX_SIZE*MATRIX_SIZE*sizeof(real_t), cudaMemcpyDeviceToHost);
    write_matrix(filename_C, C, MATRIX_SIZE);
    //print_matrix(C, MATRIX_SIZE);

    free(A);
    free(B);
    free(C);

    // Free CUDA device buffers
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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
void __global__ matmul_unopt(real_t * const A, real_t * const B, real_t * const C, size_t N)
{ 
    // Kernel calculates single element C, C[x + y*N]
    // Find global x (row index) and y (column index)
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    
    // Guards for valid x,y 
    if((x < N) && (y < N))
    {
        // Set element to zero
        C[y*N + x] = 0;
        for (int i = 0; i < N; i++)
        {
            // Row y of A dot column x of B
            C[x + y*N] += A[i + y*N] * B[x + i*N];
        }
    }
    else
    {
        // Out of bounds
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
