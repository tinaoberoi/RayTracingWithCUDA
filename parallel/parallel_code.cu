#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <cuda.h>
#include <stdbool.h>

#define MAX_BLOCKS_PER_DIM 65535
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef struct vec
{
    double x, y, z;
} vec;

typedef struct point
{
    int i, j;
} point;

// Function to allocate 2D array of size n and initialize with value as val
double *allocate_array(int n, double val)
{
    // Allocate array of size n*n
    double *arr = (double *)malloc(n * n * sizeof(double));

    // Assign value to array
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i * n + j] = val;
        }
    }
    return arr;
}

// function to free the memory of an array pointer
void free_mem(double *arr)
{
    free(arr);
}

// function to calculate dot product of 2 vectors
__device__ void dot_product(vec *a, vec *b, double *dot_prod)
{
    *dot_prod = ((a->x * b->x) + (a->y * b->y) + (a->z * b->z));
}

// function to calculate product of a vector and scalar
__device__ void vec_scalar_prod(double alpha, vec *v, vec *valpha)
{
    valpha->x = alpha * v->x;
    valpha->y = alpha * v->y;
    valpha->z = alpha * v->z;
}

// function to calculate magnitude of a vector
__device__ void vec_modulo(vec *a, double *magnitude)
{
    *magnitude = sqrt(pow(a->x, 2) + pow(a->y, 2) + pow(a->z, 2));
}

// Function to calculate unit normal vector
__device__ void vec_unit_normal(vec *a, vec *b, vec *c)
{
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
    double magnitude;
    vec_modulo(c, &magnitude);
    vec_scalar_prod((double)1.0 / (double)magnitude, c, c);
}

// A 63−bit LCG
// Returns a double p r e c i s i o n value from a uniform d i s t r i b u t i o n
// between 0.0 and 1.0 using a c a l l e r −owned state variable .
__device__ void LCG_random_double ( uint64_t *seed , double* rand_num)
{
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = ( a * (* seed ) + c ) % m;
    *rand_num = ( double ) (*seed ) / ( double ) m;
}

//”Fast Forwards” an LCG PRNG stream
// seed : s t a r t i n g seed
// n : number of i t e r a t i o n s ( samples ) to forward
// Returns : forwarded seed value
__device__ void fast_forward_LCG ( uint64_t seed , uint64_t n, uint64_t* new_seed)
{
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while (n >0)
    {
        if (n & 1)
        {
            a_new *= a ;
            c_new = c_new * a + c ;
        }
        c *= ( a + 1) ;
        a *= a ;
        n >>= 1;
    }
    *new_seed = ( a_new *seed + c_new ) % m;
}

// function to generate random number between 0 and 1
__device__ void generate_rand_0_and_1(uint64_t *seed, double* rand_num){
    LCG_random_double(seed, rand_num);
}

// function to generate random number between -1 and 1
__device__ void generate_rand_neg_1_and_1(uint64_t *seed, double* rand_num){
    double r;
    generate_rand_0_and_1(seed, &r);
    *rand_num = (2*r - 1);
}

// function to generate random ray
__device__ void generate_random_ray(vec *V, uint64_t* seed)
{
    double phi, cos_theta, sin_theta;
    generate_rand_0_and_1(seed, &phi);
    phi = phi * 2 * M_PI;
    generate_rand_neg_1_and_1(seed, &cos_theta);
    sin_theta = sqrt(1 - pow(cos_theta, 2));

    V->x = sin_theta * cos(phi);
    V->y = sin_theta * sin(phi);
    V->z = cos_theta;
}

__device__ void find_coord(double x, double z, double w_max, int grid_points, point *coord)
{
    coord->i = grid_points - (grid_points * ((x + w_max) / (2 * w_max)));
    coord->j = grid_points * ((z + w_max) / (2 * w_max));
}

__device__ void intersection_condition(vec *v, vec *c, double r, double *condition)
{
    double dp_vc, dp_cc;
    dot_product(v, c, &dp_vc);
    dot_product(c, c, &dp_cc);
    *condition = pow(dp_vc, 2) + pow(r, 2) - dp_cc;
}

__global__ void device_ray_trace(double *grid, int grid_points, long int num_rays)
{
    vec L, C, V, W, I, N, S;
    point p;
    L.x = 4;
    L.y = 4;
    L.z = -1;
    C.x = 0;
    C.y = 12;
    C.z = 0;

    double r = 6, w_y = 10, w_max = 10;
    double t, b, intersect_condition, dp_vc, dp_sn;
    uint64_t lcg_seed;

    int tid0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid0; i < num_rays; i += blockDim.x * gridDim.x)
    {
        fast_forward_LCG(42, 200*(i), &lcg_seed);
        generate_random_ray(&V, &lcg_seed);
        vec_scalar_prod((w_y / V.y), &V, &W);
        intersection_condition(&V, &C, r, &intersect_condition);
        while (!(fabs(W.x) < w_max && fabs(W.z) < w_max && (intersect_condition > 0)))
        {
            generate_random_ray(&V, &lcg_seed);
            vec_scalar_prod((w_y / V.y), &V, &W);
            intersection_condition(&V, &C, r, &intersect_condition);
            lcg_seed++;
        }
        dot_product(&V, &C, &dp_vc);
        intersection_condition(&V, &C, r, &intersect_condition);
        t = dp_vc - sqrt(intersect_condition);
        vec_scalar_prod(t, &V, &I);
        vec_unit_normal(&I, &C, &N);
        vec_unit_normal(&L, &I, &S);
        dot_product(&S, &N, &dp_sn);
        b = fmaxf(0.0, dp_sn);
        find_coord(W.x, W.z, w_max, (grid_points), &p);
        atomicAdd(&grid[p.i * (grid_points) + p.j], b);
    }
}

// function to write data to file
void write_to_file(char *filename, double *grid, int n)
{
    printf("write to file\n");
    FILE *fptr = fopen(filename, "w");
    if (fptr == NULL)
    {
        printf("Error printing grid!\n");
        exit(1);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(fptr, "%f, ", grid[i * n + j]);
        }
        fprintf(fptr, "\n");
    }
}

void parallel_ray_trace_algo(int grid_points, long int num_rays, int nthreads_per_block)
{
    // function to populate brightness grid
    int nblocks;
    double *grid;
    
    nblocks = MIN(num_rays / nthreads_per_block + 1, MAX_BLOCKS_PER_DIM);

    // allocate host memory
    double *grid_h = allocate_array(grid_points, 0.0);

    // allocate device memory
    cudaMalloc((void **)&grid, grid_points * grid_points * sizeof(double));
    cudaMemset(grid, 0.0, sizeof(double) * grid_points * grid_points);

    device_ray_trace<<<nblocks, nthreads_per_block>>>(grid, grid_points, num_rays);
    cudaDeviceSynchronize();

    // copy data to host memory 
    cudaMemcpy(grid_h, grid, grid_points * grid_points * sizeof(double), cudaMemcpyDeviceToHost);

    char filename[] = "parallel_code.txt";
    write_to_file(filename, grid_h, grid_points);

    // free Memory
    free_mem(grid_h);
    cudaFree(grid);
}

int main(int argc, char *argv[])
{
    // takes grid points, num of arrays and threads per block as user input
    int grid_points = atoi(argv[1]);
    long int num_rays = atof(argv[2]);
    int threads_per_block = atoi(argv[3]);

    parallel_ray_trace_algo(grid_points, num_rays, threads_per_block);
    cudaDeviceSynchronize();

    return 0;
}