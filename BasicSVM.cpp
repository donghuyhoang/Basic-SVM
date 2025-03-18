#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 100
#define LEARNING_RATE 0.001
#define EPOCHS 1000

typedef struct {
    double x;
    double y;
    int label;
} Point;

double random_double() {
    return (double)rand() / RAND_MAX;
}

int main() {
    srand(time(NULL));

    Point points[NUM_POINTS];
    for (int i = 0; i < NUM_POINTS; i++) {
        points[i].x = random_double();
        points[i].y = random_double();
        points[i].label = (points[i].x + points[i].y > 1) ? 1 : -1;
    }

    FILE *data_file = fopen("data.txt", "w");
    if (data_file == NULL) {
        perror("Error opening data.txt");
        return 1;
    }

    for (int i = 0; i < NUM_POINTS; i++) {
        fprintf(data_file, "%lf,%lf,%d\n", points[i].x, points[i].y, points[i].label);
    }

    fclose(data_file);

    // Khoi tao sieu phang ngau nhien
    double w1 = random_double();
    double w2 = random_double();
    double b = random_double();

    // Gradient Descent
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < NUM_POINTS; i++) {
            double result = w1 * points[i].x + w2 * points[i].y + b;
            if ((points[i].label * result) < 1) { // Misclassified hoac nam trong margin
                w1 = w1 + LEARNING_RATE * (points[i].label * points[i].x - 2 * 0.01 * w1);
                w2 = w2 + LEARNING_RATE * (points[i].label * points[i].y - 2 * 0.01 * w2);
                b = b + LEARNING_RATE * points[i].label;
            }
        }
    }

    FILE *hyperplane_file = fopen("hyperplane.txt", "w");
    if (hyperplane_file == NULL) {
        perror("Error opening hyperplane.txt");
        return 1;
    }

    fprintf(hyperplane_file, "%lf,%lf,%lf\n", w1, w2, b);
    fclose(hyperplane_file);

    // Tim support vectors
    FILE *support_vectors_file = fopen("support_vectors.txt", "w");
    if (support_vectors_file == NULL) {
        perror("Error opening support_vectors.txt");
        return 1;
    }
    int support_vector_count = 0;
    double margin = 1 / sqrt(w1 * w1 + w2 * w2); // Tinh margin
    for (int i = 0; i < NUM_POINTS; i++) {
        double result = w1 * points[i].x + w2 * points[i].y + b;

        if (fabs(result) <= margin) {
            fprintf(support_vectors_file, "%lf,%lf\n", points[i].x, points[i].y);
            support_vector_count++;
        }
    }
    fclose(support_vectors_file);

    if (support_vector_count == 0) {
        remove("support_vectors.txt");
    }

    // Ghi margin vao file
    FILE *margin_file = fopen("margin.txt", "w");
    if (margin_file == NULL) {
        perror("Error opening margin.txt");
        return 1;
    }
    fprintf(margin_file, "%lf\n", margin);
    fclose(margin_file);

    return 0;
}

