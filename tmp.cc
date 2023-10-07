#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

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

double compute_variance(double **points, int N, int D){
    double total[D];
    double average[D];
    double squaredDiffs[D] = {};
    double sum = 0.0;
    
    for (int i=0; i<N; i++){
        for (int j=0; j<D; j++){
            total[j] += points[i][j];
        }
    }

    for (int i=0; i<D; i++){
        average[i] = total[i] / N;
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            double diff = points[i][j] - average[j];
            squaredDiffs[j] += diff * diff;
        }
    }

    for (int i=0; i<D; i++){
        sum += squaredDiffs[i];
    }

    return sum / N;
    
}



int
main (int argc, char *argv[])
{
    int N, D, c;        // number of points, dimensions and GMM components

    if (argc < 2) {
        fprintf (stderr, "usage: %s [input_file]\n", argv[0]);
        exit (1);
    }

    // Read number of points and dimension
    FILE *fp = fopen (argv[1], "r");
    fscanf (fp, "%d%d", &N, &D);

    N /= 1;     // Spread among ranks in MPI communicator

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

    unit_normal un;

    // Example use
    seed (&un, 6);
    double *vec = (double*)malloc (D * sizeof(*vec));
    Gaussian_point (vec, &un, D, gmm_means[c-1], gmm_stddevs[c-1]);
    for (int i = 0; i < D; i++) {
        printf ("%lf ", vec[i]);
    }
    printf ("\n");
    int count = 0;
    // real code
    try {
        printf ("real code starts\n");
        double *points_data = (double*)malloc(N*D * sizeof(*points_data));
        double **points  = (double**)malloc(N * sizeof(*points)); 
        for (int i = 0; i < N; i++) {
            points[i] = &(points_data[i * D]);
            Gaussian_point (points[i], &un, D, gmm_means[c-1], gmm_stddevs[c-1]);
            for (int j = 0; j < D; j++) {
                printf ("%lf ", points[i][j]);
            }
            printf ("\n");
        }

        // k-means++
        #define K 3 // Number of clusters
        double centroids[K][D];

        int first_centroid_index = rand() % N;
        for (int i = 0; i < D; i++) {
            centroids[0][i] = points[first_centroid_index][i];
        }

        // Initialize the remaining centroids using k-means++
        for (int k = 1; k < K; k++) {
            double weights[N];
            double total_weight = 0.0;

            // Find the data point farthest from the existing centroids
            for (int i = 0; i < N; i++) {
                double min_dist_sqr = INFINITY;
                
                // find shortest distance to any centroid
                for (int j = 0; j < k; j++) {
                    double dist_sqr = 0.0;

                    for (int d = 0; d < D; d++) {
                        double diff = points[i][d] - centroids[j][d];
                        dist_sqr += diff * diff;
                    }

                    if (dist_sqr < min_dist_sqr) {
                        min_dist_sqr = dist_sqr;
                    }
                }

                weights[i] = min_dist_sqr;
                total_weight += weights[i];
            }

            // Set the next centroid to the selected data point
            double random_value = static_cast<double>(rand()) / RAND_MAX * total_weight;
            int next_centroid_index = -1;
            for (int i = 0; i < N; i++) {
                random_value -= weights[i];
                if (random_value <= 0.0) {
                    next_centroid_index = i;
                    break;
                }
            }
            for (int i = 0; i < D; i++) {
                centroids[k][i] = points[next_centroid_index][i];
            }
        }

        const double dt = 0.01;
        int loops = 50000;
        // Gravitational constant
        const double G = 6.67430e-11;
        const double MASS = 1000.0;

        // velocity
        double *velocity_data = (double*)malloc(N*D * sizeof(*velocity_data));
        double **velocity  = (double**)malloc(N * sizeof(*velocity)); 
        for (int i = 0; i < N; i++) {
            velocity[i] = &(velocity_data[i * D]);
            for (int j = 0; j < D; j++) {
                velocity[i][j] = 0;
            }
        }

        
        // stimulate gravity

        printf("stimulation starts, current variance: %lf\n", compute_variance(points, N, D));
        while(loops){
            loops--;
            for (int i = 0; i < N; i++) {
                double f[D] = {};
                for (int j = 0; j < N; j++) {
                    if (i == j) continue;
                    double distance_sqrt = 0.0;
                    double d[D];
                    for (int k = 0; k < D; k++) {
                        d[k] = points[j][k] - points[i][k];
                        distance_sqrt += d[k]*d[k];
                    }
                    double distance = sqrt(distance_sqrt);
                    if (distance <= 0.01) continue;

                    double force_magnitude = (G * MASS * MASS) / (distance*distance);
                    
                    // printf ("%lf \n", G);

                    for (int k = 0; k < D; k++) {
                        f[k] += force_magnitude * (d[k] / distance);
                    }
                }

                for (int k = 0; k < D; k++) {
                    velocity[i][k] += f[k] / MASS * dt;
                }
            }

            for (int i = 0; i < N; i++){
                for (int j = 0; j < D; j++) {
                    
                    points[i][j] += velocity[i][j] * dt;
                }
            }
            // for (int i = 0; i < N; i++){for (int j = 0; j < D; j++) {printf ("%lf ", points[i][j]);}printf ("\n");}
            count++;
            if (count == 100){
                double variance = compute_variance(points, N, D);
                printf("current variance: %lf\n", variance);
                count =  0;
            }
            

        }
    }catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
