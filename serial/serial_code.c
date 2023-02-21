#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cstring>
#include <omp.h>

typedef struct vec {
    double x, y, z;
} vec;

typedef struct point {
    int i, j;
} point;

// Function to allocate 2D array of size n and initialize with value as val
double* allocate_array(int n, double val){
    // Allocate array of size n*n
    double *arr = (double*)malloc(n*n*sizeof(double));

    // Assign value to array
    for(int i =0; i<n; i++){
        for(int j = 0; j<n; j++){
            arr[i*n + j] = val;
        }
    }
    return arr;
}

// Function to free the memory of an array pointer
void free_mem(double* arr){
    free(arr);
}

// Function to calculate dot product of 2 vectors
double dot_product(vec *a, vec *b){
    double res = ((a->x * b->x) + (a->y * b->y) + (a->z * b->z));
    return res;
}

// Function to calculate product of a vector and scalar
void vec_scalar_prod(double alpha, vec* v, vec* valpha){
    valpha->x = alpha * v->x;
    valpha->y = alpha * v->y;
    valpha->z = alpha * v->z;
}

// Function to calculate magnitude of a vector
void vec_modulo(vec* a, double* magnitude){
    *magnitude = sqrt(pow(a->x, 2) + pow(a->y, 2) + pow(a->z, 2));
}

// Function to calculate unit normal vector
void vec_unit_normal(vec *a, vec *b, vec *c){
    c->x = a->x - b->x ;
    c->y = a->y - b->y; 
    c->z = a->z - b->z;
    double magnitude;
    vec_modulo(c, &magnitude);
    vec_scalar_prod((double)1.0/(double)magnitude, c, c);
}

// Function to generate random number between 0 and 1
double generate_rand_0_and_1()
{
    double r = ((double) rand() / (double) RAND_MAX);
    return r;
}

// Function to generate random number between -1 and 1
double generate_rand_neg_1_and_1()
{
    double r = (2.0 * ((double)rand() / (double) RAND_MAX) - 1.0);
    return r;
}

// Function to generate random ray
void generate_random_ray(vec* V)
{
    double phi, cos_theta, sin_theta;
    phi = generate_rand_0_and_1()*2*M_PI;
    cos_theta = generate_rand_neg_1_and_1();
    sin_theta = sqrt(1 - pow(cos_theta, 2));
    
    V->x = sin_theta*cos(phi);
    V->y = sin_theta*sin(phi);
    V->z = cos_theta;
}

void find_coord(double x , double z, double w_max, int grid_points, point* coord)
{
    coord->i = grid_points - (grid_points * ((x + w_max)/(2 * w_max)));
    coord->j = grid_points * ((z + w_max)/(2 * w_max));
}

double intersection_condition(vec *v, vec *c, double r){
    return (pow(dot_product(v, c), 2) + pow(r, 2) - dot_product(c, c));
}

// Function to write data to file
void write_to_file(FILE *fptr, char *filename, double* grid, int n)
{
    fptr = fopen(filename, "w");
    if (fptr == NULL)
    {
        printf("Error printing grid!\n");
        exit(1);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fprintf(fptr, "%f, ", grid[i*n + j]);
        }
        fprintf(fptr, "\n");
    }
}

void serial_ray_trace_algo ( int grid_points, long int num_rays)
{
    // Function to populate brightness grid
    srand(time(NULL));
    
    vec* L = (vec *)malloc(sizeof(vec));
    L->x = 4; L->y = 4; L->z = -1;
    vec* C = (vec *)malloc(sizeof(vec));
    C->x = 0; C->y = 12; C->z = 0;
    double r = 6, w_y = 10, w_max = 10;

    vec *V = (vec *)malloc(sizeof(vec));
    vec *W = (vec *)malloc(sizeof(vec));
    vec *I = (vec *)malloc(sizeof(vec));
    vec *N = (vec *)malloc(sizeof(vec));
    vec *S = (vec *)malloc(sizeof(vec));
    point* p = (point*)malloc(sizeof(point));
    double* grid = allocate_array(grid_points, 0);

    double t, b;

    for(long int i = 0; i<num_rays; i++)
    {
        generate_random_ray(V);
        vec_scalar_prod((w_y/V->y), V, W);
        while (!(fabs(W->x) < w_max && fabs(W->z) < w_max && ( intersection_condition(V, C, r) > 0)))
        {
            generate_random_ray(V);
            vec_scalar_prod((w_y/V->y), V, W);
        }
        t = dot_product(V, C) -  sqrt(intersection_condition(V, C, r));
        vec_scalar_prod(t, V, I);
        vec_unit_normal(I, C, N);
        vec_unit_normal(L, I, S);
        b = fmaxf(0.0, dot_product(S, N));
        find_coord(W->x, W->z, w_max, grid_points, p);
        grid[p->i * grid_points + p->j] += b;
    }

    // Store the grid points in file
    FILE *fptr;
    char filename[]  = "serial_code.txt";
    write_to_file(fptr, filename, grid, grid_points);
    
    // Free Memory
    free_mem(grid);
    free(W);
    free(V);
    free(I);
    free(N);
    free(S);
    free(p);
    free(L);
    free(C);
}

int main(int argc, char *argv[])
{
    // Takes grid points and num of arrays as user input
    int grid_points = atof(argv[1]);
    long int num_arrays = atof(argv[2]);
    double host_start, host_end;

    host_start = omp_get_wtime(); 
    serial_ray_trace_algo(grid_points, num_arrays);
    host_end = omp_get_wtime();
    printf("time elapsed: %f\n", host_end-host_start);

    return 0;
}