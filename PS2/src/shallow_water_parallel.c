#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "../inc/argument_utils.h"

#include <mpi.h>

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// the cartesian communicator
MPI_Comm
    cart;
MPI_Datatype
    grid,
    subgrid;

int
    rank,
    comm_size,
    local_rows,
    local_cols;

#define MPI_RANK_ROOT  ( rank == 0 )

struct timeval
    t_start,
    t_stop;
double
    t_total;

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size= 10.0,
    gravity = 9.81,
    density = 997.0;
 
real_t
    *mass[2] = { NULL, NULL},
    *mass_velocity_x[2] = { NULL, NULL },
    *mass_velocity_y[2] = { NULL, NULL },
    *mass_velocity = NULL,
    *velocity_x = NULL,
    *velocity_y = NULL,
    *acceleration_x = NULL,
    *acceleration_y = NULL,
    dx,
    dt;

// New variables and enumerations

int 
    neighbour_ranks[4],
    coords[2],
    dims[2] = {0,0},
    local_rows_standard,
    local_cols_standard;

MPI_Datatype 
    column_vector,
    row_vector;

enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};


// ------

#define PN(y,x)         mass[0][(y)*(local_cols+2)+(x)]
#define PN_next(y,x)    mass[1][(y)*(local_cols+2)+(x)]
#define PNU(y,x)        mass_velocity_x[0][(y)*(local_cols+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x[1][(y)*(local_cols+2)+(x)]
#define PNV(y,x)        mass_velocity_y[0][(y)*(local_cols+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y[1][(y)*(local_cols+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(local_cols+2)+(x)]
#define U(y,x)          velocity_x[(y)*(local_cols+2)+(x)]
#define V(y,x)          velocity_y[(y)*(local_cols+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(local_cols+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(local_cols+2)+(x)]

void time_step ( void );
void boundary_condition ( real_t *domain_variable, int sign );
void create_types ( void );
void domain_init ();
void domain_save ( int_t iteration );
void domain_finalize ( void );


// new functions
void border_exchange( void );

void sendrecv_down(real_t* domain_variable);
void sendrecv_up(real_t* domain_variable);
void sendrecv_left(real_t* domain_variable);
void sendrecv_right(real_t* domain_variable);

// ----

void
swap ( real_t** t1, real_t** t2 )
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}


int
main ( int argc, char **argv )
{
    MPI_Init ( &argc, &argv );
    // TODO 1 Create a communicator with cartesian topology
    MPI_Comm_size ( MPI_COMM_WORLD, &comm_size );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );

    if ( MPI_RANK_ROOT )
    {
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
    }

    MPI_Bcast ( &N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD );

    MPI_Dims_create(comm_size,2,dims);
    
    if (rank == 0)
    {
        for(int i = 0; i < 2; i++)
        {
            printf("%d : %d \n", i, dims[i]);
        }
    }

    // Non-periodic topology.
    int periods[2] = {0,0};
    // Reorder the processes based on the cartesian topology
    int reorder = 1;

    // Creates the cartesian communcatior with the 
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,reorder, &cart);

    // gets the new rank from the communicator
    MPI_Comm_rank(cart,&rank);
    MPI_Cart_coords(cart,rank,2,coords);
    
    // Gets the rank of the neighbouring processes in the topology
    MPI_Cart_shift( cart, 1, 1, &neighbour_ranks[LEFT], &neighbour_ranks[RIGHT]);
    MPI_Cart_shift( cart, 0, 1, &neighbour_ranks[DOWN], &neighbour_ranks[UP]);


    // TODO 2 Find the number of columns and rows of each subgrid
    //        and find the local x and y offsets for each process' subgrid
    // Uncomment domain_save() and create_types() after TODO2 is complete
    domain_init();

    create_types();

    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration<=max_iteration; iteration++ )
    {
        // TODO 5 Implement border exchange

        border_exchange();

        // TODO 4 Change application of boundary condition to match cartesian topology

        boundary_condition ( mass[0], 1 );
        boundary_condition ( mass_velocity_x[0], -1 );
        boundary_condition ( mass_velocity_y[0], -1 );

        // TODO 3 Update the area of iteration in the time step
        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            if ( MPI_RANK_ROOT )
            {
                printf (
                    "Iteration %ld of %ld, (%.2lf%% complete)\n",
                    iteration,
                    max_iteration,
                    100.0 * (real_t) iteration / (real_t) max_iteration
                );
            }
            //test_values();
            domain_save ( iteration );
        }
        
        swap ( &mass[0], &mass[1] );
        swap ( &mass_velocity_x[0], &mass_velocity_x[1] );
        swap ( &mass_velocity_y[0], &mass_velocity_y[1] );
         
    }
       
   
    domain_finalize();

    gettimeofday ( &t_stop, NULL );
    t_total = WALLTIME(t_stop) - WALLTIME(t_start);

    if ( MPI_RANK_ROOT )
        printf ( "%.2lf seconds total runtime\n", t_total );

    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}
void 
border_exchange( void )
{
    // Send and recieve procedures for the differnet variables needed at the boundary.
    sendrecv_down(mass[0]);
    sendrecv_up(mass[0]);
    sendrecv_left(mass[0]);
    sendrecv_right(mass[0]);

    sendrecv_down(mass_velocity_x[0]);
    sendrecv_up(mass_velocity_x[0]);
    sendrecv_left(mass_velocity_x[0]);
    sendrecv_right(mass_velocity_x[0]);

    sendrecv_down(mass_velocity_y[0]);    
    sendrecv_up(mass_velocity_y[0]);  
    sendrecv_left(mass_velocity_y[0]);
    sendrecv_right(mass_velocity_y[0]);

}


void sendrecv_down(real_t* domain_variable)
{   
    // Sends and recieve variables down/ from up
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    MPI_Sendrecv(   &VAR(1, 1), 1, row_vector, neighbour_ranks[DOWN], 0, 
                    &VAR(local_rows+1, 1), 1, row_vector, neighbour_ranks[UP], 0, cart, MPI_STATUS_IGNORE);

    #undef VAR
}

void sendrecv_up(real_t* domain_variable)
{   
    // Sends and recieve variables up/ from down
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    MPI_Sendrecv(   &VAR(local_rows, 1), 1, row_vector, neighbour_ranks[UP], 0, 
                    &VAR(0, 1), 1, row_vector, neighbour_ranks[DOWN], 0, cart, MPI_STATUS_IGNORE);

    #undef VAR
}

void sendrecv_left(real_t* domain_variable)
{      
    // Sends and recieve variables left/ from right
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    MPI_Sendrecv(   &VAR(1, 1), 1, column_vector, neighbour_ranks[LEFT], 0, 
                    &VAR(1, local_cols+1), 1, column_vector, neighbour_ranks[RIGHT], 0, cart, MPI_STATUS_IGNORE);

    #undef VAR
}

void sendrecv_right(real_t* domain_variable)
{   
    // Sends and recieve variables right/ from left
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    MPI_Sendrecv(   &VAR(1, local_cols), 1, column_vector, neighbour_ranks[RIGHT], 0, 
                    &VAR(1, 0), 1, column_vector, neighbour_ranks[LEFT], 0, cart, MPI_STATUS_IGNORE);

    #undef VAR
}



void
time_step ( void )
{
    // TODO 3 Update the area of iteration in the time step
    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            U(y,x) = PNU(y,x) / PN(y,x);
            V(y,x) = PNV(y,x) / PN(y,x);
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
        }

    for ( int_t y=0; y<=local_rows+1; y++ )
        for ( int_t x=0; x<=local_cols+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                            ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                          + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                            ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                          + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
            );
        }

    for ( int_t y=1; y<=local_rows; y++ )
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                           ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                         + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
            );
        }
}


void
boundary_condition ( real_t *domain_variable, int sign )
{
    // TODO 4 Change application of boundary condition to match cartesian topology
    #define VAR(y,x) domain_variable[(y)*(local_cols+2)+(x)]

    // in n x m process grid as in the slides from the lecture

    // lower edge
    if (coords[0] == dims[0]-1)
    {
        for ( int_t x=1; x<=local_cols; x++ ) VAR( local_rows+1, x   ) = sign*VAR( local_rows-1, x   );
    }

    // upper edge
    if (coords[0] == 0 )
    {
        for ( int_t x=1; x<=local_cols; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
    }

    // left edge
    if (coords[1] == 0)
    {
        for ( int_t y=1; y<=local_rows; y++ ) VAR(   y, 0   ) = sign*VAR(   y, 2   );

        // lower left corner
        // 0 x m
        if (coords[0] == dims[0]-1)
        {
            VAR(   0, 0   ) = sign*VAR(   2, 2   );
        }

        // upper left corner
        // 0 x 0 
        if (coords[0] == 0)
        {
            VAR(   local_rows+1, 0   ) = sign*VAR(   local_rows-1, 2   );
        }
    }

    // right edge
    if (coords[1] == dims[1]-1)
    {
        for ( int_t y=1; y<=local_rows; y++ ) VAR(   y, local_cols+1 ) = sign*VAR(   y, local_cols-1 );

        // lower right corner
        // n x m
        if (coords[0] == dims[0]-1)
        {
            VAR(   0, local_cols+1   ) = sign*VAR(   2, local_cols-1   );
        }

        // upper right corner
        // n x 0
        if (coords[0] == 0)
        {
            VAR(   local_rows+1, local_cols+1   ) = sign*VAR(   local_rows-1, local_cols-1   );
        }
    }

    #undef VAR
}


void
create_types ( void )
{
    int cart_rank, cart_offset[2];
    MPI_Comm_rank ( cart, &cart_rank );
    MPI_Cart_coords ( cart, cart_rank, 2, cart_offset);

    MPI_Type_create_subarray ( 2, (int[2]) { local_rows+2, local_cols+2 }, (int[2]) { local_rows, local_cols }, (int[2]) {1,1}, MPI_ORDER_C, MPI_DOUBLE, &subgrid );
    MPI_Type_create_subarray ( 2, (int[2]) {N, N} , (int[2]) { local_rows, local_cols }, (int[2]) { cart_offset[0] * local_rows, cart_offset[1] * local_cols}, MPI_ORDER_C, MPI_DOUBLE, &grid );
    MPI_Type_commit ( &subgrid );
    MPI_Type_commit ( &grid ) ;



    // added types
    // row vector, equivalent to contious MPI_double.
    
    MPI_Type_vector( 1, local_cols, 0, MPI_DOUBLE, &row_vector );
    MPI_Type_commit( &row_vector);

    // column vector passes columns of the arrays mass, mass_velocity etc.
    MPI_Type_vector( local_rows, 1, local_cols+2, MPI_DOUBLE, &column_vector);
    MPI_Type_commit( &column_vector);
}


void
domain_init ()
{
    // TODO 2 Find the number of columns and rows of each subgrid
    // Hint: you can get useful information from the cartesian communicator
    local_rows  = N;
    local_cols  = N;

    int spare_rows = N%dims[0];
    int spare_cols = N%dims[1];
    int temp_local_rows = N/dims[0];
    int temp_local_cols = N/dims[1];

    // Local dimension i have at most 1 element more than N/dims[i], but the last process will have less elements.
    if (spare_rows == 0)
    {
        local_rows  = temp_local_rows;
        local_rows_standard = temp_local_rows;
    }

    // Only needed in case of non even N to processes.
    else if (coords[0] < dims[0]-1)
    {
        local_rows  = temp_local_rows + 1;
        local_rows_standard= temp_local_rows + 1;
    }
    else 
    {
        local_rows = N/dims[0] - (dims[0]-1)+spare_rows;
        local_rows_standard = temp_local_cols + 1;
    }

    // columns
    if (spare_cols == 0)
    {
        local_cols  = temp_local_cols;
        local_cols_standard = temp_local_cols;
    }

    // Only needed in case of non even N to processes.
    else if (coords[1] < dims[1]-1)
    {
        local_cols  = temp_local_cols + 1;
        local_cols_standard= temp_local_cols + 1;
    }
    else 
    {
        local_cols = N/dims[1] - (dims[1]-1)+spare_cols;
        local_cols_standard = temp_local_cols + 1;
    }

    printf("Rank: %i. \n Coords, y: %i, x: %i.\n Rows : %i, Cols: %i \n ------ \n",rank,coords[0],coords[1],local_rows,local_cols);
    int_t local_size = (local_rows + 2) * (local_cols + 2);

    mass[0] = calloc ( local_size, sizeof(real_t) );
    mass[1] = calloc ( local_size, sizeof(real_t) );

    mass_velocity_x[0] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_x[1] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_y[0] = calloc ( local_size, sizeof(real_t) );
    mass_velocity_y[1] = calloc ( local_size, sizeof(real_t) );

    mass_velocity = calloc ( local_size, sizeof(real_t) );

    velocity_x = calloc ( local_size, sizeof(real_t) );
    velocity_y = calloc ( local_size, sizeof(real_t) );

    acceleration_x = calloc ( local_size, sizeof(real_t) );
    acceleration_y = calloc ( local_size, sizeof(real_t) );

    // TODO 2 Find the local x and y offsets for each process' subgrid
    // Hint: you can get useful information from the cartesian communicator
    int_t local_x_offset = (coords[1])*local_cols_standard;
    int_t local_y_offset = (coords[0])*local_rows_standard;

    for ( int_t y=1; y<=local_rows; y++ )
    {
        for ( int_t x=1; x<=local_cols; x++ )
        {
            PN(y,x) = 1e-3;
            PNU(y,x) = 0.0;
            PNV(y,x) = 0.0;

            real_t cx = (local_x_offset + x) - N/2;
            real_t cy = (local_y_offset + y) - N/2;

            if ( sqrt ( cx*cx + cy*cy ) < N/20.0 )
            {
                PN(y,x) -= 5e-4*exp (
                    - 4*pow( cx, 2.0 ) / (real_t)(N)
                    - 4*pow( cy, 2.0 ) / (real_t)(N)
                );
            }
            PN(y,x) *= density;
        }
    }

    dx = domain_size / (real_t) N;
    dt = 5e-2;
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    MPI_File out;
    MPI_File_open ( cart, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out );

    MPI_File_set_view ( out, 0, MPI_DOUBLE, grid, "native", MPI_INFO_NULL );
    MPI_File_write_all ( out, mass[0], 1, subgrid, MPI_STATUS_IGNORE );

    MPI_File_close ( &out );
}


void
domain_finalize ( void )
{
    free ( mass[0] );
    free ( mass[1] );
    free ( mass_velocity_x[0] );
    free ( mass_velocity_x[1] );
    free ( mass_velocity_y[0] );
    free ( mass_velocity_y[1] );
    free ( mass_velocity );
    free ( velocity_x );
    free ( velocity_y );
    free ( acceleration_x );
    free ( acceleration_y );
}
