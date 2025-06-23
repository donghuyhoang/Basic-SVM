#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_POINTS 100
#define LEARNING_RATE 0.005
#define EPOCHS 30000
#define LAMBDA 0.005
// use Thuat toan support vector machine
typedef struct {
    double x;
    double y;
    int label;
} Point;

double random_double() {
    return (double)rand() / RAND_MAX;
}

void generate_data_case1(Point points[]) {
    // Du lieu phan chia theo duong x + y = 1
    printf("Data case: x + y > 1\n");
    for (int i = 0; i < NUM_POINTS; i++) {
        do {
            points[i].x = random_double();
            points[i].y = random_double();
            points[i].label = (points[i].x + points[i].y > 1) ? 1 : -1;
        } while (fabs((points[i].x + points[i].y - 1) / sqrt(2)) < 0.13);
    }
}

void generate_data_case2(Point points[]) {
    // Du lieu phan chia theo duong x - y = 0
    printf("Data case: x - y > 0\n");
    for (int i = 0; i < NUM_POINTS; i++) {
        do {
            points[i].x = random_double();
            points[i].y = random_double();
            points[i].label = (points[i].x - points[i].y > 0) ? 1 : -1;
        } while (fabs((points[i].x - points[i].y) / sqrt(2)) < 0.13);
    }
}

double check_hyperplane_accuracy(Point points[], int num_points, double w1, double w2, double b) {
    if (num_points == 0) return 0.0; // Tranh chia cho 0
    int correct = 0;
    for (int i = 0; i < num_points; i++) {
        double res = w1 * points[i].x + w2 * points[i].y + b;
        int pred = (res >= 0) ? 1 : -1;
        if (pred == points[i].label) correct++;
    }
    return (double)correct / num_points;
}

void noise(Point points[], int n){
	for (int i = 0; i < n; i++){
		(points[i].label == 1 ? points[i].label = -1 : points[i].label = 1);
	}
}
void noise1(Point points[],int n){
	int cnt = 0;
	int t = NUM_POINTS;
	int i = 0;
	while(cnt <= n && t--){
		if(points[i].label == 1){
			++cnt;
			points[i].label = -1;
		}
		i++;
	}
}
void noise2(Point points[],int n){
	int cnt = 0;
	int t = NUM_POINTS;
	int i = 0;
	while(cnt < n && t--){
		if(points[i].label == -1){
			++cnt;
			points[i].label = 1;
		}
		i++;
	}
}
void shuffle_indices(int *array, int n) { // Thay doi thu tu tap du lieu
    if (n > 1) {
        for (int i = n - 1; i > 0; i--) {
        		// 
            // Su dung ham rand_r de cai thien tinh ngau nhien (neu can)
            int j = rand() % (i + 1); // Tao mot chi so ngau nhien j sao cho 0 <= j <= i

            // Hoan doi array[i] voi array[j] chi khi j khac i de giam so lan hoan doi khong can thiet
            if (i != j) {
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
    }
}

int main() {
    srand(time(NULL));
    Point points[NUM_POINTS];
    int use_case = rand() % 2;
    if (use_case == 0) {
        generate_data_case1(points);
    } else {
        generate_data_case2(points);
    }
    //noise(points,9);
    noise1(points,5);
    noise2(points,1);
    FILE *data_file = fopen("data.txt", "w");
    if (!data_file) {
        perror("Cannot open data.txt");
        return 1;
    }
    for (int i = 0; i < NUM_POINTS; i++) {
        fprintf(data_file, "%lf,%lf,%d\n", points[i].x, points[i].y, points[i].label);
    }
    fclose(data_file);

    double w1 = (rand() % 2000 - 1000) / 10000.0;
    double w2 = (rand() % 2000 - 1000) / 10000.0;
    double b = 0;

    // Tao mang chi so de xao tron
    int indices[NUM_POINTS];
    for (int k = 0; k < NUM_POINTS; k++) {
        indices[k] = k;
    }

    printf("Starting training with LR=%.3f, Epochs=%d, Lambda=%.2f\n", LEARNING_RATE, EPOCHS, LAMBDA);

    // Huan luyen bang Gradient Descent
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        // Xao tron cac chi so o dau moi epoch
        shuffle_indices(indices, NUM_POINTS);
        for (int k = 0; k < NUM_POINTS; k++) {
            int i = indices[k]; // Lay chi so thuc cua diem du lieu
            // margin = 1 / ||W||
            double result = w1 * points[i].x + w2 * points[i].y + b;
            if (points[i].label * result < 1) {
                w1 -=(LEARNING_RATE * (-points[i].label * points[i].x + 2 * LAMBDA * w1));
                w2 -=(LEARNING_RATE * (-points[i].label * points[i].y + 2 * LAMBDA * w2));
                b  -= -LEARNING_RATE * points[i].label;
            } else {
                w1 -= LEARNING_RATE * 2 * LAMBDA * w1;
                w2 -= LEARNING_RATE * 2 * LAMBDA * w2;
            }
        }
        if (epoch % (EPOCHS / 20) == 0 || epoch == 1 || epoch == EPOCHS) {
            double acc = check_hyperplane_accuracy(points, NUM_POINTS, w1, w2, b);
            printf("Epoch %5d/%d, Accuracy: %.4lf, w1: %.4f, w2: %.4f, b: %.4f\n",
                   epoch, EPOCHS, acc, w1, w2, b);
        }
    }
    int support_vectors = 0;
    for (int i = 0; i < NUM_POINTS; i++){
    	double result = w1 * points[i].x + w2 * points[i].y + b;
    	if(points[i].label * result <= 1 + 1e-9) ++support_vectors;
		}
    FILE *hyper_file = fopen("hyperplane.txt", "w");
    if (!hyper_file) {
        perror("Cannot open hyperplane.txt");
        return 1;
    }
    fprintf(hyper_file, "%lf,%lf,%lf\n", w1, w2, b);
    fclose(hyper_file);

    double norm_w = sqrt(w1 * w1 + w2 * w2);
    double margin = 0.0;
    if (norm_w > 1e-9) { // Tranh chia cho 0 neu w1 va w2 rat nho
         margin = 2.0 / norm_w;
    }
    FILE *margin_file = fopen("margin.txt", "w");
    if (!margin_file) {
        perror("Cannot open margin.txt");
        return 1;
    }
    fprintf(margin_file, "%lf\n", margin);
    fclose(margin_file);

    printf("\nFinal trained model:\n");
    printf("w1 = %.6f, w2 = %.6f, b = %.6f\n", w1, w2, b);
    printf("||w|| = %.6f\n", norm_w);
    printf("Margin (2/||w||) = %.6f\n", margin);
    printf("Final Accuracy: %.4lf\n", check_hyperplane_accuracy(points, NUM_POINTS, w1, w2, b));
    printf("Support Vectors: %d",support_vectors);
    return 0;
}
