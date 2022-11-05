#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>

#include "../inc/argument_utils.h"


typedef int64_t int_t;
typedef double real_t;

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

struct timeval
    t_start,
    t_stop;
double
    t_total;

int_t
    N,
    max_iteration,
    snapshot_frequency;

const real_t
    domain_size = 10.0,
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

// number of threads
int thread_count = 128;
int flag = 0;
int* flag_list;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

int barrier_option = 3;

#define PN(y,x)         mass[0][(y)*(N+2)+(x)]
#define PN_next(y,x)    mass[1][(y)*(N+2)+(x)]
#define PNU(y,x)        mass_velocity_x[0][(y)*(N+2)+(x)]
#define PNU_next(y,x)   mass_velocity_x[1][(y)*(N+2)+(x)]
#define PNV(y,x)        mass_velocity_y[0][(y)*(N+2)+(x)]
#define PNV_next(y,x)   mass_velocity_y[1][(y)*(N+2)+(x)]
#define PNUV(y,x)       mass_velocity[(y)*(N+2)+(x)]
#define U(y,x)          velocity_x[(y)*(N+2)+(x)]
#define V(y,x)          velocity_y[(y)*(N+2)+(x)]
#define DU(y,x)         acceleration_x[(y)*(N+2)+(x)]
#define DV(y,x)         acceleration_y[(y)*(N+2)+(x)]

void time_step ( void );
void boundary_condition ( real_t *domain_variable, int sign );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );

// functions for pthread
void *pthread_fat_step_func( void * rank);
void pthread_boundary_condition ( real_t *domain_variable, int sign, long rank, int_t * domain );
void pthread_time_step( long rank, int_t* domain);
void get_domain( long rank, int_t* domain);



void
swap ( real_t** t1, real_t** t2 )
{
    real_t* tmp;
	tmp = *t1;
	*t1 = *t2;
	*t2 = tmp;
}

int
clamp (int my_var, int lower, int upper)
{
    if (my_var < lower) 
    {
        return lower;
    }
    else if (my_var > upper) 
    {
        return upper;
    }
    else 
    {
        return my_var;
    }
}


int
main ( int argc, char **argv )
{

    long thread_id;
    pthread_t* thread_handles;

    thread_handles = malloc( thread_count * sizeof(pthread_t));
    flag_list = calloc(thread_count, sizeof(int));
    

    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;
    int err_create;
    int err_join;

    domain_init();

    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        flag = 0;
        for ( thread_id = 0; thread_id < thread_count; thread_id ++)
        {
            err_create  = pthread_create(&thread_handles[thread_id], NULL, pthread_fat_step_func, (void*) thread_id);
            if ( err_create != 0)
            {
                printf("errorcode : %d for id: %ld \n ", err_create, thread_id);
            }
            
        }

        for (thread_id = 0; thread_id < thread_count; thread_id++)
        {
            err_join = pthread_join(thread_handles[thread_id], NULL);
            if ( err_join != 0)
            {
                printf("errorjoin : %d for id: %ld \n ", err_join, thread_id);
            }
        }

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %lld of %lld, (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );
            domain_save ( iteration );
        }

        swap ( &mass[0], &mass[1] );
        swap ( &mass_velocity_x[0], &mass_velocity_x[1] );
        swap ( &mass_velocity_y[0], &mass_velocity_y[1] );
    }

    gettimeofday ( &t_stop, NULL );
    t_total = WALLTIME(t_stop) - WALLTIME(t_start);
    printf ( "%.2lf seconds total runtime\n", t_total );

    free(flag_list);
    free(thread_handles);
    domain_finalize();

    exit ( EXIT_SUCCESS );
}

// new function for timestep in thread

void*
pthread_fat_step_func( void * rank)
{
    int_t domain[2];
    long my_rank = (long) rank;
    get_domain(my_rank, domain);

    pthread_boundary_condition(mass[0], 1, rank, domain);
    pthread_boundary_condition(mass_velocity_x[0], -1, rank, domain);
    pthread_boundary_condition(mass_velocity_y[0], -1, rank, domain);

    pthread_time_step(rank, domain);
}

void 
pthread_time_step ( long rank, int_t* domain )
{
    //printf("Hello time step %ld, a: %ld, b %ld \n", my_rank, domain[0], domain[1]);

    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            U(y,x) = PNU(y,x) / PN(y,x);
            V(y,x) = PNV(y,x) / PN(y,x);
        }

    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNUV(y,x) = PN(y,x) * U(y,x) * V(y,x);
        }

    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=0; x<=N+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }
    if (domain[0] == 1)
    {
        int_t y = 0;
        for ( int_t x=0; x<=N+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }
    }
    if (domain[1] == N)
    {
        int_t y = N+1;
        for ( int_t x=0; x<=N+1; x++ )
        {
            DU(y,x) = PN(y,x) * U(y,x) * U(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
            DV(y,x) = PN(y,x) * V(y,x) * V(y,x)
                    + 0.5 * gravity * ( PN(y,x) * PN(y,x) / density );
        }
    }


    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNU_next(y,x) = 0.5*( PNU(y,x+1) + PNU(y,x-1) ) - dt*(
                            ( DU(y,x+1) - DU(y,x-1) ) / (2*dx)
                          + ( PNUV(y,x+1) - PNUV(y,x-1) ) / (2*dx)
            );
        }
    

    switch (barrier_option)
    {
        case 1:
            pthread_mutex_lock(&mutex);
            flag++;
            pthread_mutex_unlock(&mutex);

            while(flag < thread_count);
            break;
        
        case 2:
            flag_list[rank] = 1;
            while( flag_list[(rank-1)%thread_count] != 1 && flag_list[(rank+1)%thread_count] != 1);
            break;

        case 3:
            flag_list[rank] = 1;
            while( flag_list[clamp(rank-1,0,thread_count)] != 1 && flag_list[clamp(rank+1,0,thread_count)] != 1);
            break;
        
        default:
            break;

    }


    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PNV_next(y,x) = 0.5*( PNV(y+1,x) + PNV(y-1,x) ) - dt*(
                            ( DV(y+1,x) - DV(y-1,x) ) / (2*dx)
                          + ( PNUV(y+1,x) - PNUV(y-1,x) ) / (2*dx)
            );
        }

    for ( int_t y=domain[0]; y<=domain[1]; y++ )
        for ( int_t x=1; x<=N; x++ )
        {
            PN_next(y,x) = 0.25*( PN(y,x+1) + PN(y,x-1) + PN(y+1,x) + PN(y-1,x) ) - dt*(
                           ( PNU(y,x+1) - PNU(y,x-1) ) / (2*dx)
                         + ( PNV(y+1,x) - PNV(y-1,x) ) / (2*dx)
            );
        }
}

void
pthread_boundary_condition ( real_t *domain_variable, int sign, long rank, int_t * domain )
{
    #define VAR(y,x) domain_variable[(y)*(N+2)+(x)]
    // if thread has bottom y slice
    if (domain[0] == 1)
    {
        // lower corners
        VAR(   0, 0   ) = sign*VAR(   2, 2   );
        VAR(   0, N+1 ) = sign*VAR(   2, N-1 );

        // y = 0
        for ( int_t x=1; x<=N; x++ ) VAR(   0, x   ) = sign*VAR(   2, x   );
    }
    // if thread has top y slice
    if ( domain[1] == N)
    {   
        // upper corners
        VAR( N+1, 0   ) = sign*VAR( N-1, 2   );
        VAR( N+1, N+1 ) = sign*VAR( N-1, N-1 );

        // y = N+1
        for ( int_t x=1; x<=N; x++ ) VAR( N+1, x   ) = sign*VAR( N-1, x   );
    }

    // left and right boundary values
    for ( int_t y=domain[0]; y<=domain[1]; y++ ) VAR(   y, 0   ) = sign*VAR(   y, 2   );
    for ( int_t y=domain[0]; y<=domain[1]; y++ ) VAR(   y, N+1 ) = sign*VAR(   y, N-1 );
    
    
    #undef VAR
}



void
get_domain( long rank, int_t* domain )
{
    // get domain of thread
    if( rank == thread_count)
    {
        domain[0] = 1 + rank*(N/thread_count);
        domain[1] = N;
    }
    else
    {
        domain[0] = 1 + rank*(N/thread_count);
        domain[1] = (rank+1)*(N/thread_count);
    }
    
}


void
time_step ( void )
{
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


void
boundary_condition ( real_t *domain_variable, int sign )
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


void
domain_init ( void )
{
    mass[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );

    mass_velocity_x[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_x[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_y[0] = calloc ( (N+2)*(N+2), sizeof(real_t) );
    mass_velocity_y[1] = calloc ( (N+2)*(N+2), sizeof(real_t) );

    mass_velocity = calloc ( (N+2)*(N+2), sizeof(real_t) );

    velocity_x = calloc ( (N+2)*(N+2), sizeof(real_t) );
    velocity_y = calloc ( (N+2)*(N+2), sizeof(real_t) );
    acceleration_x = calloc ( (N+2)*(N+2), sizeof(real_t) );
    acceleration_y = calloc ( (N+2)*(N+2), sizeof(real_t) );

    for ( int_t y=1; y<=N; y++ )
    {
        for ( int_t x=1; x<=N; x++ )
        {
            PN(y,x) = 1e-3;
            PNU(y,x) = 0.0;
            PNV(y,x) = 0.0;

            real_t cx = x-N/2;
            real_t cy = y-N/2;
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
    sprintf ( filename, "data/%.5lld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( !out )
    {
        fprintf( stderr, "Failed to open file %s\n", filename );
        exit(1);
    }
    for ( int_t y = 1; y <= N; y++ )
    {
        fwrite ( &mass[0][y*(N+2)+1], N, sizeof(real_t), out );
    }
    fclose ( out );
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
