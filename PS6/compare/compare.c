#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

const uint64_t DOUBLE_EXPONENT_MASK = 0x7FF0000000000000;
const uint32_t SINGLE_EXPONENT_MASK = 0x7F800000;
const uint64_t DOUBLE_MINIMAL_MANTISSA = 0x0000000000000001;
const uint32_t SINGLE_MINIMAL_MANTISSA = 0x00000001;

#define MAX_EPSILONS 8


unsigned int max_epsilons = MAX_EPSILONS;

typedef int bool;
#define true 1
#define false 0

void help_message()
{
    printf("Usage: ./compare SERIAL_FILE PARALLEL_FILE MATRIX_SIZE");
}

typedef float real_t;

bool compareFloat(void *, void *);
bool compareDouble(void *, void *);

bool (*fnPtrs[2])(void *, void *) = {compareFloat, compareDouble};

void read_matrix(char *filename, real_t *matrix, size_t matrix_size);

int main(int argc, char *argv[])
{

    size_t matrix_size;

    if ( argc < 4 ) {
        fprintf(stderr, "Not enough arguments supplied!\n");
        help_message();
        exit(EXIT_FAILURE);
    }


    if ( argc >= 4 && argv[4] ) {
        max_epsilons = (unsigned int) atoi(argv[4]);
    }

    matrix_size = strtoul(argv[3], NULL, 10);

    real_t *serial_matrix, *parallel_matrix;

    serial_matrix = malloc(sizeof(real_t) * matrix_size * matrix_size);
    parallel_matrix = malloc(sizeof(real_t) * matrix_size * matrix_size);

    read_matrix(argv[1], serial_matrix, matrix_size);
    read_matrix(argv[2], parallel_matrix, matrix_size);

    // Use the comparison function that matches either float or double
    bool (*cmpFn)(void *, void *) = fnPtrs[sizeof(real_t) / 4 - 1];

    size_t inequalities = 0;

    real_t max_difference = 0.0;

    for ( size_t i = 0; i < matrix_size * matrix_size; i++ )
    {
        if ( ! cmpFn( (void *) &serial_matrix[i], (void *) &parallel_matrix[i] ) )
        {
            inequalities++;

            real_t diff = (real_t) fabs(serial_matrix[i] - parallel_matrix[i]);
            if ( diff > max_difference ) max_difference = diff;
        }
    }

    //printf("Found %zu inequalities\n", inequalities);

    printf("%zu\n", inequalities);

    return EXIT_SUCCESS;
}



bool compareFloat(void *_a, void *_b)
{
    float a = *(float*)_a;
    float b = *(float*)_b;

    uint32_t BITREP_A;
    uint32_t BITREP_B;

    memcpy(&BITREP_A, &a, sizeof(float));
    memcpy(&BITREP_B, &b, sizeof(float));

    assert(BITREP_A == *(uint32_t *) &a);
    assert(BITREP_B == *(uint32_t *) &b);

    // Find the largest of the two exponent
    uint32_t EXPONENT_A = (BITREP_A & SINGLE_EXPONENT_MASK) >> 23;
    uint32_t EXPONENT_B = (BITREP_B & SINGLE_EXPONENT_MASK) >> 23;
    uint32_t LARGEST_EXPONENT = EXPONENT_A;


    if ( EXPONENT_B > EXPONENT_A ) LARGEST_EXPONENT = EXPONENT_B;

    LARGEST_EXPONENT -= 23;

    // Construct the machine epsilon comparison number
    uint32_t COMPARISON_FLOAT_BITREP = 0;
    COMPARISON_FLOAT_BITREP |= (LARGEST_EXPONENT << 23);

    float comparison_float;
    memcpy(&comparison_float, &COMPARISON_FLOAT_BITREP, sizeof(float));

    comparison_float *= (float) max_epsilons;

    // Compare the difference to the machine epsilon
    if ( fabs( a - b ) >= comparison_float ) return false;
    return true;

}


///< Compares double precision IEEE-754 numbers.
///< Does NOT support subnormal numbers.
bool compareDouble(void *_a, void *_b)
{
    double a = *(double*)_a;
    double b = *(double*)_b;

    uint64_t BITREP_A;
    uint64_t BITREP_B;

    memcpy(&BITREP_A, &a, sizeof(double));
    memcpy(&BITREP_B, &b, sizeof(double));

    assert(BITREP_A == *(uint64_t *) &a);
    assert(BITREP_B == *(uint64_t *) &b);

    // Find the largest of the two exponent
    uint64_t EXPONENT_A = (BITREP_A & DOUBLE_EXPONENT_MASK) >> 52;
    uint64_t EXPONENT_B = (BITREP_B & DOUBLE_EXPONENT_MASK) >> 52;
    uint64_t LARGEST_EXPONENT = EXPONENT_A;

    if ( EXPONENT_B > EXPONENT_A ) LARGEST_EXPONENT = EXPONENT_B;

    LARGEST_EXPONENT -= 52;

    // Construct the machine epsilon comparison number
    uint64_t COMPARISON_FLOAT_BITREP = 0;
    COMPARISON_FLOAT_BITREP |= (LARGEST_EXPONENT << 52);

    double comparison_float;
    memcpy(&comparison_float, &COMPARISON_FLOAT_BITREP, sizeof(double));

    comparison_float *= max_epsilons;

    // Compare the difference to the machine epsilon
    if ( fabs( a - b ) >= comparison_float ) return false;
    return true;

}


void read_matrix(char *filename, real_t *matrix, size_t matrix_size)
{
    FILE *fh = fopen(filename, "rb");

    if ( ! fh ) {
        fprintf(stderr, "Failed to open %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }

    fread(matrix, sizeof(real_t), matrix_size * matrix_size, fh);

    fclose(fh);
}

