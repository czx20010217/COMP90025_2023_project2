#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include <unistd.h> 

constexpr double EPSILON = 0.01;
const MPI_Comm comm = MPI_COMM_WORLD;
const int root = 0;
const double MASS = 1000.0;
// constant
const double dt = 1;
const double G = 6.67430e-11;
const int MAX_LOOP = 1000;
const double SIZE = 50.0;
const double THETA = 0.3;

// It would be cleaner to put seed and Gaussian_point into this class,
// but this allows them to be called like regular C functions.
// Feel free to make the whole code more C++-like.
class unit_normal {
    std::mt19937 gen;
    std::normal_distribution<double> d{0,1};
    public :
    void seed (long int s) { gen.seed (s); }
    double sample () { return d(gen); }
};

void
seed (unit_normal *un, long int s) {
    un->seed (s);
}

double *
Gaussian_point (double *out, unit_normal *un, int D, double *mean, double sd) {
    for (int i = 0; i < D; i++)
        out[i] = un->sample () * sd + mean[i];
    return out;
}

double getMean(double **points, int N, int D){
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sqrt_sum = 0.0;
            for (int k=0; k<D; k++){
                double d = points[i][k] - points[j][k];
                sqrt_sum += d * d;
            }
            sum += sqrt(sqrt_sum);
        }
    }

    return sum / N / N;
    
}

double compute_variance(double **points, int N, int D){
    double total[D];
    double average[D];
    double squaredDiffs[D];
    double sum = 0.0;

    memset(total, 0, D*sizeof(*total));
    
    for (int i=0; i<N; i++){
        for (int j=0; j<D; j++){
            total[j] += points[i][j];
        }
    }

    for (int i=0; i<D; i++){
        average[i] = total[i] / N;
        squaredDiffs[i] = 0.0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            double diff = points[i][j] - average[j];
            squaredDiffs[j] += diff * diff;
        }
    }

    for (int i=0; i<D; i++){
        sum += squaredDiffs[i];
    }

    return sum / N;
    
}

int findCumulativeProbabilityIndex(double *gmm_probs, int c, int index, int N){
    double cumulativeSum = 0.0;
    for (int i = 0; i < c; i++){
        cumulativeSum += gmm_probs[i] * N;
        if (cumulativeSum > index){
            return i;
        }
    }
    return c-1;
}

void compute_velocity(double **points, double **velocity, double N, double D){
    return;
}

struct QuadTreeNode {
    std::vector<double> center;
    double size;
    double mass;
    double* point;
    std::vector<QuadTreeNode*> children;

    QuadTreeNode(const std::vector<double>& center, double size) : size(size) {
        this->center = center;
        mass = 0.0;
        point = nullptr;
        // printf("default tree children size: %d\n", children.size());
    }

    ~QuadTreeNode() {
        for (QuadTreeNode* child : children) {
            // printf("delete here\n");
            delete child;
        }
    }
};

void insert(QuadTreeNode* node, double* point, int D, int reinsert) {
    if (node->mass == 0) {
        node->mass = MASS;
        node->point = point;
        return;
    }else if (!reinsert){
        node->mass += MASS;
    }

    if (node->point != nullptr) {
        // If the node already contains a point, split it and redistribute
        double* old_point = node->point;
        node->point = nullptr;
        for (int i = 0; i < (1 << D); i++) { // generate 2**D child
            std::vector<double> newCenter(node->center);
            for (size_t j = 0; j < D; j++) {
                newCenter[j] += (i & (1 << j) ? 0.25 : -0.25) * node->size; // use a binary representation for children to compute center
            }
            node->children.push_back(new QuadTreeNode(newCenter, node->size * 0.5)); // insert children into the tree
        }
        insert(node, old_point, D, 1); // reinsert the point to children
    }

    std::vector<int> pointPosition(D, 0.0);
    for (size_t i = 0; i < D; i++) {
        pointPosition[i] = (point[i] >= node->center[i]) ? 1 : 0;
    }

    int index = 0;
    for (size_t i = 0; i < pointPosition.size(); i++) {
        index |= (pointPosition[i] << i);
    }
    insert(node->children[index], point, D, 0);
}

void generate_points(double ** local_points, double *gmm_probs, double **gmm_means, double *gmm_stddevs, int c, int D, int sub_N, unit_normal un){
    int gmm_index;
    for (int i = 0; i < sub_N; i++) {
        gmm_index = findCumulativeProbabilityIndex(gmm_probs, c, i, sub_N);
        
        Gaussian_point (local_points[i], &un, D, gmm_means[gmm_index], gmm_stddevs[gmm_index]);
    }
}

void calculate_local_force(QuadTreeNode* node, double* point, int D, double theta, double G, double* force) {
    double distance = 0.0;
    for (int i = 0; i < D; i++) {
        double d = node->center[i] - point[i];
        distance += d * d;
    }
    distance = std::sqrt(distance);
    if (distance == 0) return;

    if (node->point != nullptr && node->point != point) { // leaf node
        double force_magnitude = G * MASS * MASS / (distance * distance + EPSILON*EPSILON);
        for (int i = 0; i < D; i++) {
            force[i] += force_magnitude * (node->center[i] - point[i]) / distance;
        }
    } else if (node->size / distance < theta) { // distance large enough
        double force_magnitude = G * node->mass * MASS / (distance * distance + EPSILON*EPSILON);
        for (int i = 0; i < D; i++) {
            force[i] += force_magnitude * (node->center[i] - point[i]) / distance;
        }
    } else { // distance too small, use children to compute force
        for (QuadTreeNode* child : node->children) {
            if (child != nullptr) {
                calculate_local_force(child, point, D, theta, G, force);
            }
        }
    }
}

// replace each node's center with mass center
void calculate_mass_center(QuadTreeNode* node, int D) {
    if (node->mass == 0) {
        return;
    }

    if (node->point != nullptr){
        for (int i=0; i<D; i++){
            node->center[i] = node->point[i] ;
        }
        return;
    }


    double mass_center[D];
    memset(mass_center, 0, D*sizeof(*mass_center));

    for (QuadTreeNode* child : node->children) {
        if (child != nullptr) {
            calculate_mass_center(child, D);
        }
        for (int i=0; i<D; i++){
            mass_center[i] += child->mass * child->center[i];
        }
    }
    for (int i=0; i<D; i++){
        node->center[i] = mass_center[i] / node->mass ;
    }
}

void update_position(double **points, double** cluster_points, double** velocity, double* force_data, 
                    double **force, int N, int D, int local_size, int start_index, std::vector<double> root_center){

    // Build the tree
    QuadTreeNode *tree_root = new QuadTreeNode(root_center, SIZE);
    for (int i = 0; i < local_size; i++) {
        insert(tree_root, cluster_points[i], D, 0);
    }
    calculate_mass_center(tree_root, D); // compute mass center

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        calculate_local_force(tree_root, points[i], D, THETA, G, force[i]);
    }
    MPI_Allreduce(MPI_IN_PLACE, force_data, N*D, MPI_DOUBLE, MPI_SUM, comm);

    #pragma omp parallel for
    for (int i = 0; i < local_size; i++){
        for (int j = 0; j < D; j++) {
            velocity[i][j] += force[start_index+i][j] / MASS * dt;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < local_size; i++){
        for (int j = 0; j < D; j++) {
            cluster_points[i][j] += velocity[i][j] * dt;
        }
    }

    delete tree_root;
}

void generate_centroids(double** centroids, double** points, double** local_points, int first_centroid_index, int N, int sub_N, int rank, int D, int K){
    for (int i = 0; i < D; i++) {
        centroids[0][i] = points[first_centroid_index][i];
    }

    // Initialize the remaining centroids using k-means++
    for (int k = 1; k < K; k++) {
        double weights[N];
        double total_weights = 0.0;
        double local_weights[sub_N];
        double local_total_weight = 0.0;

        // Find the data point farthest from the existing centroids
        for (int i = 0; i < sub_N; i++) {
            double min_dist_sqr = INFINITY;
            
            // find shortest distance to any centroid
            for (int j = 0; j < k; j++) {
                double dist_sqr = 0.0;

                for (int d = 0; d < D; d++) {
                    double diff = local_points[i][d] - centroids[j][d];
                    dist_sqr += diff * diff;
                }

                if (dist_sqr < min_dist_sqr) {
                    min_dist_sqr = dist_sqr;
                }
            }

            local_weights[i] = min_dist_sqr;
            local_total_weight += local_weights[i];
        }

        MPI_Gather(local_weights, sub_N, MPI_DOUBLE, weights, sub_N, MPI_DOUBLE, root, comm);
        MPI_Reduce(&local_total_weight, &total_weights, 1, MPI_DOUBLE, MPI_SUM, root, comm);

        if (rank == root) {
            double tmp = 0.0;
            for (int i = 0; i < N; i++){
                tmp += weights[i];
            }
        }

        int next_centroid_index;

        if (rank == root) {
            // Set the next centroid to the selected data point
            double random_value = static_cast<double>(rand()) / RAND_MAX * total_weights;
            next_centroid_index = -1;
            for (int i = 0; i < N; i++) {
                random_value -= weights[i];
                if (random_value <= 0.0) {
                    next_centroid_index = i;
                    break;
                }
            }
        }
        MPI_Bcast(&next_centroid_index, 1, MPI_INT, root, comm);
        //printf("Now on node: %d, total size: %d, %d centroid index: %d\n", rank, size, k, next_centroid_index);
        for (int i = 0; i < D; i++) {
            centroids[k][i] = points[next_centroid_index][i];
            //printf("Now on node: %d, total size: %d, %d centroid %d value: %lf\n", rank, size, k, i, points[next_centroid_index][i]);
        }
        
        
    }
}

void arrange_cluster(double** centroids, double** local_points, 
                    int N, int sub_N, int K, int D, int* local_cluster_index){

    int rank;
    MPI_Comm_rank(comm, &rank);

    // split to clusters
    for (int i = 0; i < sub_N; i++) {
        double min_dist_sqr = INFINITY;
        int min_cluster_index = 0;
        for (int k = 0; k < K; k++){
            double dist_sqr = 0.0;
            for (int d = 0; d < D; d++){
                double diff = local_points[i][d] - centroids[k][d];
                dist_sqr += diff * diff;
            }

            if (dist_sqr < min_dist_sqr) {
                min_dist_sqr = dist_sqr;
                min_cluster_index = k;
            }
        }
        local_cluster_index[i] = min_cluster_index;
    }
}

int
main (int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(comm, &rank);
    int size;
    MPI_Comm_size(comm, &size);

    printf("Now on node: %d, total size: %d\n", rank, size);

    // reading file will be performed on all node to reduce overhead
    int N, D, c;        // number of points, dimensions and GMM components
    int sub_N;
    FILE *fp;
    if (argc < 2) {
        fprintf (stderr, "usage: %s [input_file]\n", argv[0]);
        exit (1);
    }

    // Read number of points and dimension
    fp = fopen (argv[1], "r");
    fscanf (fp, "%d%d", &N, &D);
    // It would be cleaner to put this all in a class,
    // but this keeps the skeleton C-friendly.
    fscanf (fp, "%d", &c);

    double *gmm_mean_data = (double*)malloc(c*D * sizeof(*gmm_mean_data));
    double **gmm_means  = (double**)malloc(c * sizeof(*gmm_means));
    double *gmm_stddevs = (double*)malloc(c * sizeof(*gmm_stddevs));
    double *gmm_probs   = (double*)malloc(c * sizeof(*gmm_probs));

    for (int i = 0; i < c; i++) {
        gmm_means[i] = &(gmm_mean_data[i * D]);
        for (int d = 0; d < D; d++)
            fscanf (fp, "%lf", &gmm_means[i][d]);
        fscanf (fp, "%lg%lg", &gmm_stddevs[i], &gmm_probs[i]);
    }

    // initialize generator
    unit_normal un;
    seed (&un, rank);
    int count = 0;

    // points generated on each node
    sub_N = N / size;

    double *local_points_data = (double*)malloc(sub_N*D * sizeof(*local_points_data));
    double **local_points  = (double**)malloc(sub_N * sizeof(*local_points));
    for (int i = 0; i < sub_N; i++) {
        local_points[i] = &(local_points_data[i * D]);
    }

    generate_points(local_points, gmm_probs, gmm_means, gmm_stddevs, c, D, sub_N, un);

    // gather points
    double *points_data = (double*)malloc(N*D * sizeof(*points_data));
    double **points  = (double**)malloc(N * sizeof(*points));
    for (int i = 0; i < N; i++) {
        points[i] = &(points_data[i * D]);
    }

    MPI_Allgather(local_points_data, sub_N * D, MPI_DOUBLE, points_data, sub_N * D, MPI_DOUBLE, comm);
    // if (rank == root){  for (int i = 0; i < N; i++){for (int j = 0; j < D; j++) {printf ("%lf ", points[i][j]);}printf ("\n");} } 
    MPI_Barrier(comm);

    // k-means++
    const int K = size; // Number of clusters
    double *centroids_date = (double*)malloc(K*D * sizeof(*centroids_date));
    double **centroids  = (double**)malloc(K * sizeof(*centroids));
    for (int i = 0; i < K; i++) {
        centroids[i] = &(centroids_date[i * D]);
    }
    int first_centroid_index;

    if (rank == root) {
        first_centroid_index = rand() % N;
    }


    MPI_Bcast(&first_centroid_index, 1, MPI_INT, root, comm);
    MPI_Barrier(comm);
    for (int i = 0; i < D; i++) {
        centroids[0][i] = points[first_centroid_index][i];
    }

    generate_centroids(centroids, points, local_points, first_centroid_index, N, sub_N, rank, D, K);


    // arrange to clusters
    int local_cluster_index[sub_N];
    int cluster_index[N];

    arrange_cluster(centroids, local_points, N, sub_N, K, D, local_cluster_index);

    // scatter clusters

    int local_size = 0;
    MPI_Allgather(local_cluster_index, sub_N, MPI_INT, cluster_index, sub_N, MPI_INT, comm);
    // if (rank == root){  for (int i = 0; i < N; i++){printf ("%d ", cluster_index[i]);}printf ("\n");} 

    for (int i = 0; i < N; i++){
        if (cluster_index[i] == rank){
            local_size += 1;
        }
    }

    double *cluster_points_data = (double*)malloc(local_size*D * sizeof(*cluster_points_data));
    double **cluster_points  = (double**)malloc(local_size * sizeof(*cluster_points));
    for (int i = 0; i < local_size; i++) {
        cluster_points[i] = &(cluster_points_data[i * D]);
    }

    int index = 0;
    for (int i = 0; i < N; i++){
        if (cluster_index[i] == rank){
            for (int j = 0; j < D; j++) {
                cluster_points[index][j] = points[i][j];
            }
            index++;
        }
    }
    printf("Now on node: %d, total size: %d cluster size: %d %d\n", rank, size, local_size, index);

    int sizes[size];
    int data_sizes[size];
    MPI_Allgather(&local_size, 1, MPI_INT, sizes, 1, MPI_INT, comm);
    int total_size = 0;
    for (int i = 0; i < size; i++) {
        total_size += sizes[i];
        data_sizes[i] = sizes[i]*D;
    }
    assert(total_size == N);
    //if (rank == root){  for (int i = 0; i < size; i++){printf ("%d ", data_sizes[i]);}printf ("\n");} 

    int displacements[size];
    displacements[0] = 0;
    for (int i = 1; i < size; i++) {
        displacements[i] = displacements[i - 1] + sizes[i - 1]*D;
    }
    //if (rank == root){  for (int i = 0; i < size; i++){printf ("%d ", displacements[i]);}printf ("\n");} 

    int point_displacements[size];
    point_displacements[0] = 0;
    for (int i = 1; i < size; i++) {
        point_displacements[i] = point_displacements[i - 1] + sizes[i - 1];
    }

    MPI_Allgatherv(
        cluster_points_data,
        local_size*D,
        MPI_DOUBLE,
        points_data,
        data_sizes,
        displacements,
        MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    MPI_Barrier(comm);

    // start of stimulation
    std::vector<double> rootCenter(D, 0.0);
    for (int i = 0; i < D; i++) {
        rootCenter[i] = centroids[rank][i];
    }
    int loops = MAX_LOOP;

    // velocity
    double *velocity_data = (double*)malloc(N*D * sizeof(*velocity_data));
    double **velocity  = (double**)malloc(N * sizeof(*velocity)); 
    for (int i = 0; i < N; i++) {
        velocity[i] = &(velocity_data[i * D]);
    }
    memset(velocity_data, 0, N*D*sizeof(*velocity_data));

    // global force
    double *force_data = (double*)malloc(N*D * sizeof(*force_data));
    memset(force_data, 0, N*D*sizeof(*force_data));
    double **force  = (double**)malloc(N * sizeof(*force)); 
    for (int i = 0; i < N; i++) {
        force[i] = &(force_data[i * D]);
        for (int j=0; j<D;j++){
            assert(force[i][j] == 0);
        }
    }
    
    
    double startVariance = compute_variance(points, N, D);
    
    int start_index = point_displacements[rank];
    for (int i = 0; i < local_size; i++) {
        for (int j=0; j<D;j++){
            assert(cluster_points[i][j] == points[start_index+i][j]);
        }
    }
    
    int anySolutionFound;
    int isSolutionFound = 0;
    while(loops){
        loops--;
        memset(force_data, 0, N*D*sizeof(*force_data));
        update_position(points, cluster_points, velocity, force_data, force, N, D, local_size, start_index, rootCenter);

        MPI_Allgatherv(
            cluster_points_data,
            local_size*D,
            MPI_DOUBLE,
            points_data,
            data_sizes,
            displacements,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        
        // for (int i = 0; i < N; i++){for (int j = 0; j < D; j++) {printf ("%lf ", points[i][j]);}printf ("\n");}
        count += 1;

        if (rank == root){
            double variance = compute_variance(points, N, D);
            // printf("current variance: %lf\n", variance);
            double factor = startVariance / variance;
            if (factor > 3.0){
                printf("current variance: %lf\n", variance);
                printf("end\n");
                isSolutionFound = 1;
            } else if (factor < 0.5){
                printf("failed, start variance: %lf, current variance: %lf, factor: %lf\n", startVariance, variance, factor);
                printf("current variance: %lf\n", variance);
                printf("failed\n");
                isSolutionFound = 1;
            }
            if (count == 50){
                printf("current variance: %lf\n", variance);
                count =  0;
            }
        }

        MPI_Allreduce(&isSolutionFound, &anySolutionFound, 1, MPI_INT, MPI_MAX, comm);
        if (anySolutionFound || isSolutionFound) {
            if (rank == 0) {
                std::cout << "Root process found the solution." << std::endl;
            }
            std::cout << "Process " << rank << " terminates." << std::endl;
            MPI_Finalize();
            return 0;
        }
    }
    MPI_Finalize();
    return 0;
}
