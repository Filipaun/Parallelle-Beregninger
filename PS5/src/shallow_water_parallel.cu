// ---------------------------------------------------------
// TDT4200 Parallel Computing - Graded CUDA
// ---------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

// new cuda headers
#include <cuda.h>
#include <cooperative_groups.h>

#include "../inc/argument_utils.h"

// cooperative group namespace
namespace cg = cooperative_groups;

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency,
    // number of elements
    elements;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *h_mass_0 = NULL,
    *h_mass_1 = NULL,
    *d_mass_0 = NULL,
    *d_mass_1 = NULL,

    *h_mass_velocity_x_0 = NULL,
    *h_mass_velocity_x_1 = NULL,
    *d_mass_velocity_x_0 = NULL,
    *d_mass_velocity_x_1 = NULL,

    *h_mass_velocity_y_0 = NULL,
    *h_mass_velocity_y_1 = NULL,
    *d_mass_velocity_y_0 = NULL,
    *d_mass_velocity_y_1 = NULL,

    *h_mass_velocity = NULL,
    *d_mass_velocity = NULL,

    *h_velocity_x = NULL,
    *d_velocity_x = NULL,
    *h_velocity_y = NULL,
    *d_velocity_y = NULL,

    *h_acceleration_x = NULL,
    *d_acceleration_x = NULL,
    *h_acceleration_y = NULL,
    *d_acceleration_y = NULL,

    dx,
    dt;


#define PN(y,x)         mass_0[(y)*(N+2)+(x)]
#define PN_next(y,x)    mass_1[(y)*(N+2)+(x)]
#define PNU(y,x)        mass_velocity_x_0[(y)*(N+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x_1[(y)*(N+2)+(x)]
#define PNV(y,x)        mass_velocity_y_0[(y)*(N+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y_1[(y)*(N+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(N+2)+(x)]
#define U(y,x)          velocity_x[(y)*(N+2)+(x)]
#define V(y,x)          velocity_y[(y)*(N+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(N+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(N+2)+(x)]


#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



// the two time step kernels for the SERIAL cuda implementation

void __global__ time_step_1_serial (real_t *velocity_x, real_t *velocity_y,
                    real_t *acceleration_x, real_t *acceleration_y,
                    real_t *mass_velocity_x_0, real_t *mass_velocity_y_0,
                    real_t *mass_velocity, real_t *mass_0,
                    real_t dt, real_t dx, int_t N
);
void __global__ time_step_2_serial (real_t *acceleration_x, real_t *acceleration_y,
                    real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                    real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                    real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
                    real_t dt, real_t dx, int_t N
);

// the two time step kernels for the PARALLEL cuda implementation
void __global__ time_step_1_parallel (real_t *velocity_x, real_t *velocity_y,
                    real_t *acceleration_x, real_t *acceleration_y,
                    real_t *mass_velocity_x_0, real_t *mass_velocity_y_0,
                    real_t *mass_velocity, real_t *mass_0,
                    real_t dt, real_t dx, int_t N
);
void __global__ time_step_2_parallel (real_t *acceleration_x, real_t *acceleration_y,
                    real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                    real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                    real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
                    real_t dt, real_t dx, int_t N
);

// single time step kernel for cooperative groups
void __global__ time_step_coop (real_t *velocity_x, real_t *velocity_y,
                    real_t *acceleration_x, real_t *acceleration_y,
                    real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
                    real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
                    real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
                    real_t dt, real_t dx, int_t N
);

// TODO: Rewrite boundary_condition as a device function.
void __device__ boundary_condition_serial ( real_t *domain_variable, int sign, int_t N );
void __device__ boundary_condition_parallel (int x, int y, real_t *domain_variable, int sign, int_t N );


void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );



// Pthreads threaded domain save function
void *domain_save_threaded ( void *iter );

void
swap ( real_t** t1, real_t** t2 )
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}

// generate grid and blocks size for some N
void get_dims( int_t N, dim3* grid_dim, dim3* block_dim )
{
    if ( N+2 <= 32)
    {
        grid_dim -> x = 1;
        grid_dim -> y = 1;

        block_dim -> x = N+2;
        block_dim -> y = N+2;
    }
    else
    {
        grid_dim -> x = (N+2)/32 + ( (N+2) % 32 != 0);
        grid_dim -> y = (N+2)/32 + ( (N+2) % 32 != 0);

        block_dim -> x = 32;
        block_dim -> y = 32;
    }
}


int
main ( int argc, char **argv )
{

    int is_parallel = 2;
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    dim3  
        grid_dim, 
        block_dim;

    if(is_parallel == 1)
    {
        printf("############# \n");
        printf("Running simple parallel \n");
        printf("############# \n");

        get_dims(N,&grid_dim, &block_dim);
        
    }
    else if(is_parallel == 2)
    {
        printf("############# \n");
        printf("Running advanced parallel \n");
        printf("############# \n");

        get_dims(N,&grid_dim, &block_dim);


        int numBlocksPerSm = 0;
        int dev = 0;
        int numThreads = 1024;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, time_step_coop, numThreads, 0);

        printf(" Max numblockpersm: %d   SM: %d,  \n", numBlocksPerSm, deviceProp.multiProcessorCount);
    }
    else
    {
        // ... 
    }

    printf("N: %ld Max iterations: %ld, snap freq.: %ld \n", N, max_iteration, snapshot_frequency);
    printf("Grid.x %d Grid.y %d Block.x %d Block.y %d \n", grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    printf("Total number of blocks: %d \n", grid_dim.x*grid_dim.y );


    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {

        if(is_parallel == 0)
        {
            time_step_1_serial<<<1,1>>>(	d_velocity_x, 		d_velocity_y,
                            d_acceleration_x, 	d_acceleration_y,
                            d_mass_velocity_x_0,   d_mass_velocity_y_0,	
                            d_mass_velocity, 	d_mass_0, 
                            dt, dx, N
	        );
            cudaDeviceSynchronize();
            time_step_2_serial<<<1,1>>>( d_acceleration_x, 	d_acceleration_y,
                            d_mass_velocity_x_0,	d_mass_velocity_x_1,
                            d_mass_velocity_y_0,	d_mass_velocity_y_1,
                            d_mass_velocity, 	d_mass_0, d_mass_1,
                            dt, dx, N
	        );
            cudaDeviceSynchronize();
        }

        else if(is_parallel == 1)
        {
            time_step_1_parallel<<<grid_dim,block_dim>>>(	d_velocity_x, 		d_velocity_y,
                            d_acceleration_x, 	d_acceleration_y,
                            d_mass_velocity_x_0,   d_mass_velocity_y_0,	
                            d_mass_velocity, 	d_mass_0, 
                            dt, dx, N
	        );
            cudaDeviceSynchronize();
            time_step_2_parallel<<<grid_dim,block_dim>>>( d_acceleration_x, 	d_acceleration_y,
                            d_mass_velocity_x_0,	d_mass_velocity_x_1,
                            d_mass_velocity_y_0,	d_mass_velocity_y_1,
                            d_mass_velocity, 	d_mass_0, d_mass_1,
                            dt, dx, N
	        );
            cudaDeviceSynchronize();
        }
        else if(is_parallel == 2)
        {
            void *kernel_args[] = {&d_velocity_x, &d_velocity_y,
                            &d_acceleration_x, 	&d_acceleration_y,
                            &d_mass_velocity_x_0,	&d_mass_velocity_x_1,
                            &d_mass_velocity_y_0,	&d_mass_velocity_y_1,
                            &d_mass_velocity, 	&d_mass_0, &d_mass_1,
                            &dt, &dx, &N};
            cudaError_t c_err = cudaLaunchCooperativeKernel( (void *) time_step_coop, grid_dim, block_dim, kernel_args);
            cudaErrorCheck(c_err);
        }
        else
        {
            // ... 
        }


        // TODO: Launch time_step kernels

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld, (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );


            // TODO: Copy the masses from the device to host prior to domain_save
            cudaMemcpy( h_mass_0, d_mass_0, sizeof(real_t) * elements, cudaMemcpyDeviceToHost );
            domain_save ( iteration );
        }

        // TODO: Swap device buffer pointers between iterations

        swap ( &d_mass_0, &d_mass_1 );
        swap ( &d_mass_velocity_x_0, &d_mass_velocity_x_1 );
        swap ( &d_mass_velocity_y_0, &d_mass_velocity_y_1 );
    }

    domain_finalize();

    exit ( EXIT_SUCCESS );
}


// TODO: Rewrite this function as one or more CUDA kernels
// ---------------------------------------------------------
// To ensure correct results, the participating threads in the thread
// grid must be synchronized after calculating the accelerations (DU, DV).
// If the grid is not synchronized, data dependencies cannot be guaranteed.

void __global__
time_step_1_serial (real_t *velocity_x, real_t *velocity_y,
		real_t *acceleration_x, real_t *acceleration_y,
		real_t *mass_velocity_x_0, real_t *mass_velocity_y_0,
		real_t *mass_velocity, real_t *mass_0,
        real_t dt, real_t dx, int_t N
)
{
    boundary_condition_serial ( mass_0, 1, N );
    boundary_condition_serial ( mass_velocity_x_0, -1, N );
    boundary_condition_serial ( mass_velocity_y_0, -1, N );

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            U(y,x) = PNU(y,x) / PN(y,x);
            V(y,x) = PNV(y,x) / PN(y,x);
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
        }

    for ( int_t y=0; y<=N+1; y++ )
        for ( int_t x=0; x<=N+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }
}

void __global__
time_step_2_serial (real_t *acceleration_x, real_t *acceleration_y,
		real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
		real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
		real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
        real_t dt, real_t dx, int_t N
)
{
    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                            ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                          + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                            ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                          + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=N; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                           ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                         + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
            );
        }
}
void __global__
time_step_1_parallel (real_t *velocity_x, real_t *velocity_y,
		real_t *acceleration_x, real_t *acceleration_y,
		real_t *mass_velocity_x_0, real_t *mass_velocity_y_0,
		real_t *mass_velocity, real_t *mass_0,
        real_t dt, real_t dx, int_t N
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    boundary_condition_parallel ( x, y, mass_0, 1, N );
    boundary_condition_parallel ( x, y, mass_velocity_x_0, -1, N );
    boundary_condition_parallel ( x, y, mass_velocity_y_0, -1, N );

    if(( x > 0 ) && (x < N+1) && (y > 0) && (y < N+1))
    {
        U(y,x) = PNU(y,x) / PN(y,x);
        V(y,x) = PNV(y,x) / PN(y,x);

        PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
    }

    if((x < N+2) && (y < N+2))
    {
        DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );

    }

            
}


void __global__
time_step_2_parallel (real_t *acceleration_x, real_t *acceleration_y,
		real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
		real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
		real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
        real_t dt, real_t dx, int_t N
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(( x > 0 ) && (x < N+1) && (y > 0) && (y < N+1))
    {
        PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                        ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                        + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
        );

        PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                        ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                        + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
        );


        PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                        ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                        + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
        );
    }
}

// single time step kernel for cooperative groups
void __global__ 
time_step_coop (real_t *velocity_x, real_t *velocity_y,
            real_t *acceleration_x, real_t *acceleration_y,
            real_t *mass_velocity_x_0, real_t *mass_velocity_x_1,
            real_t *mass_velocity_y_0, real_t *mass_velocity_y_1,
            real_t *mass_velocity, real_t *mass_0, real_t *mass_1,
            real_t dt, real_t dx, int_t N
)
{   
    // define the grid group o
    cg::grid_group thread_group = cg::this_grid();

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    boundary_condition_parallel ( x, y, mass_0, 1, N );
    boundary_condition_parallel ( x, y, mass_velocity_x_0, -1, N );
    boundary_condition_parallel ( x, y, mass_velocity_y_0, -1, N );

    if(( x > 0 ) && (x < N+1) && (y > 0) && (y < N+1))
    {
        U(y,x) = PNU(y,x) / PN(y,x);
        V(y,x) = PNV(y,x) / PN(y,x);

        PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
    }

    if((x < N+2) && (y < N+2))
    {
        DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );

    }

    thread_group.sync();

    if(( x > 0 ) && (x < N+1) && (y > 0) && (y < N+1))
    {
        PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                        ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                        + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
        );

        PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                        ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                        + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
        );


        PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                        ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                        + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
        );
    }
    thread_group.sync();

}


// TODO: Rewrite boundary_condition as a device function.
void __device__
boundary_condition_serial ( real_t *domain_variable, int sign, int_t N )
{
    #define VAR(y,x) domain_variable[(y)*(N+2)+(x)]
    
    VAR(   0, 0   ) = sign*VAR(   2, 2   );
    VAR( N+1, 0   ) = sign*VAR( N-1, 2   );
    VAR(   0, N+1 ) = sign*VAR(   2, N-1 );
    VAR( N+1, N+1 ) = sign*VAR( N-1, N-1 );

    for ( int_t y=1; y<=N; y++ ) VAR(   y, 0   ) = sign*VAR(   y, 2   );
    for ( int_t y=1; y<=N; y++ ) VAR(   y, N+1 ) = sign*VAR(   y, N-1 );
    for ( int_t x=1; x<=N; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
    for ( int_t x=1; x<=N; x++ ) VAR( N+1, x   ) = sign*VAR( N-1, x   );
    #undef VAR
}

// TODO: Rewrite boundary_condition as a device function.
void __device__
boundary_condition_parallel (int x, int y, real_t *domain_variable, int sign, int_t N )
{
    #define VAR(y,x) domain_variable[(y)*(N+2)+(x)]

    if( x == 0)
    {
        if ( y == N+1)
        {
            VAR( N+1, 0   ) = sign*VAR( N-1, 2   );
        }
        else if ( y < N+1)
        {
            VAR(   y, 0   ) = sign*VAR(   y, 2   );
        }
    }

    else if( x == N+1)
    {
        if ( y == N+1)
        {
            VAR( N+1, N+1 ) = sign*VAR( N-1, N-1 );
        }
        else if ( y < N+1)
        {
            VAR(   y, N+1 ) = sign*VAR(   y, N-1 );
        }
        else
        {
            // Out of bounds
        }
    }
    
    else if( y == 0)
    {   
        if (x == 0)
        {
            VAR(   0, 0   ) = sign*VAR(   2, 2   );
        }
        else if (x <= N+1)
        {
            VAR(   0, x   ) = sign*VAR(   2, x   );
        }
        else
        {
            // Out of bounds
        }
    }

    else if( y == N+1)
    {
        if (y == 0)
        {
            VAR(   0, N+1 ) = sign*VAR(   2, N-1 );
        }
        else if (x <= N+1)
        {
            VAR(   N+1, x   ) = sign*VAR(   N-1, x   );
        }
        else
        {
            // Out of bounds
        }
    }

    #undef VAR
}


void
domain_init ( void )
{
    elements = (N+2)*(N+2);

    // TODO: Allocate device buffers for masses, velocities and accelerations.
    // -----------------------------------------------------
    h_mass_0 = (real_t *)calloc ( elements, sizeof(real_t) );
    h_mass_1 = (real_t *)calloc ( elements, sizeof(real_t) );

    h_mass_velocity_x_0 =(real_t *)calloc ( elements, sizeof(real_t) );
    h_mass_velocity_x_1 =(real_t *)calloc ( elements, sizeof(real_t) );
    h_mass_velocity_y_0 =(real_t *)calloc ( elements, sizeof(real_t) );
    h_mass_velocity_y_1 =(real_t *)calloc ( elements, sizeof(real_t) );

    h_mass_velocity = (real_t *)calloc ( elements, sizeof(real_t) );

    h_velocity_x =  (real_t *)calloc ( elements, sizeof(real_t) );
    h_velocity_y = (real_t *)calloc ( elements, sizeof(real_t) );
    h_acceleration_x = (real_t *)calloc ( elements, sizeof(real_t) );
    h_acceleration_y = (real_t *)calloc ( elements, sizeof(real_t) );

    cudaMalloc ( &d_mass_0, sizeof(real_t) * elements );
    cudaMalloc ( &d_mass_1, sizeof(real_t) * elements );

    cudaMalloc ( &d_mass_velocity_x_0, sizeof(real_t) * elements);
    cudaMalloc ( &d_mass_velocity_x_1, sizeof(real_t) * elements);
    cudaMalloc ( &d_mass_velocity_y_0, sizeof(real_t) * elements);
    cudaMalloc ( &d_mass_velocity_y_1, sizeof(real_t) * elements);

    cudaMalloc ( &d_mass_velocity, sizeof(real_t) * elements);

    cudaMalloc ( &d_velocity_x, sizeof(real_t) * elements);
    cudaMalloc ( &d_velocity_y, sizeof(real_t) * elements);
    cudaMalloc ( &d_acceleration_x, sizeof(real_t) * elements);
    cudaMalloc ( &d_acceleration_y, sizeof(real_t) * elements);

    for ( int_t y=1; y<=N; y++ )
    {
        for ( int_t x=1; x<=N; x++ )
        {
	    h_mass_0[y*(N+2) + x] = 1e-3;
	    h_mass_velocity_x_0[y*(N+2) + x] = 0.0;
	    h_mass_velocity_y_0[y*(N+2) + x] = 0.0;

            real_t cx = x-N/2;
            real_t cy = y-N/2;
            if ( sqrt ( cx*cx + cy*cy ) < N/20.0 )
            {
                h_mass_0[y*(N+2) + x] -= 5e-4*exp (
                    - 4*pow( cx, 2.0 ) / (real_t)(N)
                    - 4*pow( cy, 2.0 ) / (real_t)(N)
                );
            }

            h_mass_0[y*(N+2) + x] *= density;
        }
    }

    dx = domain_size / (real_t) N;
    dt = 5e-2;

    cudaMemcpy( d_mass_0, h_mass_0, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_mass_1, h_mass_1, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    
    cudaMemcpy( d_mass_velocity_x_0, h_mass_velocity_x_0, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_mass_velocity_x_1, h_mass_velocity_x_1, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_mass_velocity_y_0, h_mass_velocity_y_0, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_mass_velocity_y_1, h_mass_velocity_y_1, sizeof(real_t) * elements, cudaMemcpyHostToDevice );

    cudaMemcpy( d_mass_velocity, h_mass_velocity, sizeof(real_t) * elements, cudaMemcpyHostToDevice );

    cudaMemcpy( d_velocity_x, h_velocity_x, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_velocity_y, h_velocity_y, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_acceleration_x, h_acceleration_x, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
    cudaMemcpy( d_acceleration_y, h_acceleration_y, sizeof(real_t) * elements, cudaMemcpyHostToDevice );
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( !out )
    {
        fprintf( stderr, "Failed to open file %s\n", filename );
        exit(1);
    }
    //fwrite ( mass[0], (N+2)*(N+2), sizeof(real_t), out );
    for ( int_t y = 1; y <= N; y++ )
    {
        fwrite ( &h_mass_0[y*(N+2)+1], N, sizeof(real_t), out );
    }
    fclose ( out );
}

void
domain_finalize ( void )
{
    free ( h_mass_0 );
    free ( h_mass_1 );
    free ( h_mass_velocity_x_0 );
    free ( h_mass_velocity_x_1 );
    free ( h_mass_velocity_y_0 );
    free ( h_mass_velocity_y_1 );
    free ( h_mass_velocity );
    free ( h_velocity_x );
    free ( h_velocity_y );
    free ( h_acceleration_x );
    free ( h_acceleration_y );

    cudaFree ( d_mass_0 );
    cudaFree ( d_mass_1 );
    cudaFree ( d_mass_velocity_x_0 );
    cudaFree ( d_mass_velocity_x_1 );
    cudaFree ( d_mass_velocity_y_0 );
    cudaFree ( d_mass_velocity_y_1 );
    cudaFree ( d_mass_velocity );
    cudaFree ( d_velocity_x );
    cudaFree ( d_velocity_y );
    cudaFree ( d_acceleration_x );
    cudaFree ( d_acceleration_y );
}
