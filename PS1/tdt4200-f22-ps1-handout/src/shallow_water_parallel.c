#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
    gravity = 9.81,
    density = 997.0;

real_t
    *mass[2] = { NULL, NULL },
    *mass_velocity_x[2] = { NULL, NULL },
    *velocity_x = NULL,
    *acceleration_x = NULL,
    dx,
    dt;

// Global value for number of subdivisions on sub grids.
int_t sub_grid_N;
int_t standard_sub_grid_N;

#define PN(x)        mass[0][(x)]
#define PN_next(x)   mass[1][(x)]
#define PNU(x)       mass_velocity_x[0][(x)]
#define PNU_next(x)  mass_velocity_x[1][(x)]
#define U(x)         velocity_x[(x)]
#define DU(x)        acceleration_x[(x)]

void time_step ( void );
void boundary_condition( real_t *domain_variable, int sign, int my_rank, int comm_sz );
void domain_init (int my_rank, int comm_sz );
void domain_save ( int_t iteration, int my_rank );
void domain_finalize ( void );

// new function for sending border values
void send_border ( int my_rank, int my_size);

void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int
main ( int argc, char **argv )
{
    // TODO 1 Initialize MPI
    // Done

    int size;
    int rank;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // TODO 2 Parse arguments in the rank 0 processes
    // and broadcast to other processes
    // Done

    if (rank == 0)
    {
        OPTIONS * options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;

    }

    // Better to do in one BCast?
    MPI_Bcast(&N, 1, MPI_LONG, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_LONG, 0 , MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_LONG, 0 , MPI_COMM_WORLD);

    printf("Hello, from process %d of %d, N = %ld \n", rank, size, N);

    // TODO 3 Allocate space for each process' sub-grid
    // and initialize data for the sub-grid
    domain_init(rank,size);

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 7 Communicate border values
        if ( size != 0) {
            send_border(rank, size);
        }        

        // TODO 5 Boundary conditions
        boundary_condition(mass[0], 1, rank, size);
        boundary_condition(mass_velocity_x[0], -1, rank, size);

        // TODO 4 Time step calculations
        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            // TODO 6 MPI I/O
            domain_save ( iteration, rank);
        }

        swap( &mass[0], &mass[1] );
        swap( &mass_velocity_x[0], &mass_velocity_x[1] );
    }

    printf("DONE \n");
    domain_finalize();

    // TODO 1 Finalize MPI
    // Done

    MPI_Finalize();


    exit ( EXIT_SUCCESS );
}

void 
send_border ( int my_rank, int comm_sz)
{

    // Better than 4 MPI_send?
    real_t east_out[4] = {PN(sub_grid_N),PNU(sub_grid_N),U(sub_grid_N),DU(sub_grid_N)};
    real_t east_in[4];

    real_t west_out[4] = {PN(1),PNU(1),U(1),DU(1)};
    real_t west_in[4];

    // Send borders east
    if ( my_rank % 2 == 0) 
    {
        MPI_Send ( east_out , 4 , MPI_DOUBLE , ( my_rank + 1) % comm_sz , 1, MPI_COMM_WORLD ) ;
        MPI_Recv ( east_in , 4 , MPI_DOUBLE , ( my_rank +comm_sz - 1) % comm_sz , 1, MPI_COMM_WORLD , MPI_STATUS_IGNORE ) ;
    } else 
    {
        MPI_Recv ( east_in , 4 , MPI_DOUBLE , ( my_rank +comm_sz -1) % comm_sz , 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE ) ;
        MPI_Send ( east_out , 4 , MPI_DOUBLE , ( my_rank + 1) % comm_sz , 1, MPI_COMM_WORLD ) ;
    }

    // Send borders west
    if ( my_rank % 2 == 0) 
    {
        MPI_Send ( west_out , 4 , MPI_DOUBLE , ( my_rank +comm_sz - 1) % comm_sz , 2, MPI_COMM_WORLD ) ;
        MPI_Recv ( west_in , 4 , MPI_DOUBLE , ( my_rank + 1) % comm_sz , 2, MPI_COMM_WORLD , MPI_STATUS_IGNORE ) ;
    } else 
    {
        MPI_Recv ( west_in , 4 , MPI_DOUBLE , ( my_rank + 1) % comm_sz , 2, MPI_COMM_WORLD , MPI_STATUS_IGNORE ) ;
        MPI_Send ( west_out , 4 , MPI_DOUBLE , ( my_rank +comm_sz - 1) % comm_sz , 2, MPI_COMM_WORLD ) ;
    }


    PN(0) = east_in[0];
    PNU(0) = east_in[1];
    U(0) = east_in[2];
    DU(0) = east_in[3];

    PN(sub_grid_N+1) = west_in[0];
    PNU(sub_grid_N+1) = west_in[1];
    U(sub_grid_N+1) = west_in[2];
    DU(sub_grid_N+1) = west_in[3];

}



void
time_step ( void )
{
    // TODO 4 Time step calculations

    for ( int_t x=0; x<=sub_grid_N+1; x++ )
    {
        DU(x) = PN(x) * U(x) * U(x)
                + 0.5 * gravity * PN(x) * PN(x) / density;
    }

    for ( int_t x=1; x<=sub_grid_N; x++ )
    {
        PNU_next(x) = 0.5*( PNU(x+1) + PNU(x-1) ) - dt*(
                      ( DU(x+1) - DU(x-1) ) / (2*dx)
        );
    }

    for ( int_t x=1; x<=sub_grid_N; x++ )
    {
        PN_next(x) = 0.5*( PN(x+1) + PN(x-1) ) - dt*(
                       ( PNU(x+1) - PNU(x-1) ) / (2*dx)
        );
    }

    for ( int_t x=1; x<=sub_grid_N; x++ )
    {
        U(x) = PNU_next(x) / PN_next(x);
    }
}


void
boundary_condition ( real_t *domain_variable, int sign, int my_rank, int comm_sz )
{
    // TODO 5 Boundary conditions
    // Done 
    #define VAR(x) domain_variable[(x)]
    if (my_rank == 0)
    {
        VAR(   0 ) = sign*VAR( 2   );
    }
    if ( my_rank == comm_sz-1)
    {
        VAR( sub_grid_N+1 ) = sign*VAR( sub_grid_N-1 );
    }
    #undef VAR
}


void
domain_init (int my_rank, int comm_sz)
{
    // TODO 3 Allocate space for each process' sub-grid
    // and initialize data for the sub-grid
    int ugly_grid = N%comm_sz;
    int temp_sub_grid_N = N/comm_sz;

    // Sub grids have at most 1 more than N/comm_sz, but the last process will have a smaller grid.
    if (ugly_grid == 0)
    {
        sub_grid_N = temp_sub_grid_N;
        standard_sub_grid_N = temp_sub_grid_N;
    }
    else if (my_rank < comm_sz -1)
    {
        sub_grid_N = temp_sub_grid_N + 1;
        standard_sub_grid_N = temp_sub_grid_N + 1;
    }
    else 
    {
        sub_grid_N = N/comm_sz - (comm_sz-1)+ugly_grid;
        standard_sub_grid_N = temp_sub_grid_N + 1;
    }
    
    printf("Size of sub grix : %li \n", sub_grid_N);

    mass[0] = calloc ( (sub_grid_N+2), sizeof(real_t) );
    mass[1] = calloc ( (sub_grid_N+2),  sizeof(real_t) );

    mass_velocity_x[0] = calloc ( (sub_grid_N+2), sizeof(real_t) );
    mass_velocity_x[1] = calloc ( (sub_grid_N+2),  sizeof(real_t) );

    velocity_x = calloc ( (sub_grid_N+2), sizeof(real_t) );
    acceleration_x = calloc ( (sub_grid_N+2), sizeof(real_t) );
 
    printf("BRO IT IS FINE %i \n", my_rank);
    
    // Data initialization
    for ( int_t x=1; x<=sub_grid_N; x++ )
    {
        PN(x) = 1e-3;
        PNU(x) = 0.0;

        real_t c = (x+my_rank*standard_sub_grid_N)-N/2;
        if ( sqrt ( c*c ) < N/20.0 )
        {
            PN(x) -= 5e-4*exp (
                    - 4*pow( c, 2.0 ) / (real_t)(N)
            );
        }

        PN(x) *= density;
    }

    dx = domain_size / (real_t) N;
    dt = 0.1*dx;

}


void
domain_save ( int_t iteration, int my_rank )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    // TODO 6 MPI I/O

    MPI_File out;
    MPI_File_open ( MPI_COMM_WORLD, 
                    filename, 
                    MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL,
                    &out);
    
    MPI_Offset offset = my_rank*sizeof(double)*standard_sub_grid_N;
    MPI_File_write_at_all ( out, offset, &mass[0][1], sub_grid_N, MPI_DOUBLE, MPI_STATUS_IGNORE );

    MPI_File_close ( &out );
}


void
domain_finalize ( void )
{
    free ( mass[0] );
    free ( mass[1] );
    free ( mass_velocity_x[0] );
    free ( mass_velocity_x[1] );
    free ( velocity_x );
    free ( acceleration_x );
}

