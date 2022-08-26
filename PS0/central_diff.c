#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int N = 32;
    printf("Hello world! This is an integer: %d\n", N);

    int* f = calloc(N,sizeof(int));
    int* df = calloc(N-2,sizeof(int));

    printf("Function values :\n");
    for (int i = 0; i < N ; ++i)
    {
        // f(x) = x^2
        *(f+i) = i*i;

        printf("%i, ", *(f+i));
    }

    printf("\nDerivative values:\n");

    for (int i = 0; i < N-2; ++i )
    {
        // df
        *(df+i) = (*(f+(i+2)) - *(f+(i)))/2;

        printf("%i, ", *(df+i));
    }
    printf("\n");


    int* wack_ptr =  (int *)0x42424242DEADBEEF;
    
    //printf("TEST : %i",*wack_ptr);

    // Prints array A from question
    if (0) {

        free(f);
        int* A = (int*) malloc(32*sizeof(int));

        printf("Array created with malloc(): \n");
        for (int i = 0; i < 32; ++i) {
            printf("%i, ",*(A+i));
        }

        printf("\n");
    }


    return 0;
}
