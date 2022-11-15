#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <math.h>

#define MAX_ELEM M_PI/2
typedef float real_t;

#define SEED 92837

real_t random_real_t()
{
    return (real_t) (((real_t) rand() / (real_t) RAND_MAX) * (real_t) MAX_ELEM);
}

void write_matrix(const char * const filename, real_t *matrix, unsigned int matrix_size)
{
    FILE *fh = fopen(filename, "wb");

    if ( ! fh ) {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fwrite(matrix, sizeof(real_t), matrix_size * matrix_size, fh);

    fclose(fh);
}

int main(int argc, char *argv[])
{
    if ( argc <= 1 ) {
        fprintf(stderr, "Matrix size not supplied!");
    }

    // Seed the random generator
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    srand(SEED);


    unsigned long MATRIX_SIZE;

    MATRIX_SIZE = strtoul(argv[1], NULL, 10);

    real_t *A, *B;

    A = calloc(sizeof(real_t), MATRIX_SIZE * MATRIX_SIZE);
    B = calloc(sizeof(real_t), MATRIX_SIZE * MATRIX_SIZE);

    for ( unsigned int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; i++)
    {
        A[i] = random_real_t();
        B[i] = random_real_t();
    }

    char filename_A[50];
    char filename_B[50];

    sprintf(filename_A, "data/size%zu-A.bin", MATRIX_SIZE);
    sprintf(filename_B, "data/size%zu-B.bin", MATRIX_SIZE);

    write_matrix(filename_A, A, (unsigned int) MATRIX_SIZE);
    write_matrix(filename_B, B, (unsigned int) MATRIX_SIZE);

    return 0;
}
