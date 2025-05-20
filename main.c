#include "FXP.H"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define MAX_EPOCHS 10000000
#define LEARNING_RATE 0.2

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;

char fxptoa_buff0[16];
char fxptoa_buff1[16];
char fxptoa_buff2[16];
char fxptoa_buff3[16];
char fxptoa_buff4[16];

fxp sigmoid(fxp x) {
    return fxp_div(FXP_ONE, (FXP_ONE + FXP_FROM_FLOAT(expf(FXP_TO_FLOAT(-x)))));
}

fxp sigmoid_derivative(fxp x) {
    return fxp_mul(x, (FXP_ONE - x));
}

typedef struct {
    fxp hidden_weights[INPUT_NEURONS][HIDDEN_NEURONS];
    fxp output_weights[HIDDEN_NEURONS][OUTPUT_NEURONS];
    
    // Biases
    fxp hidden_bias[HIDDEN_NEURONS];
    fxp output_bias[OUTPUT_NEURONS];
    
    // Activations
    fxp hidden_layer[HIDDEN_NEURONS];
    fxp output_layer[OUTPUT_NEURONS];

    // Learning Rate
    fxp lr;
} NeuralNetwork;

// Initialize network with random weights
void init_network(NeuralNetwork *nn) {
    // Seed the random number generator
    srand(time(NULL));
    
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            nn->hidden_weights[i][j] = FXP_FROM_FLOAT(((float)rand() / RAND_MAX) * 2.0 - 1.0);
        }
    }
    
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        nn->hidden_bias[j] = FXP_FROM_FLOAT(((float)rand() / RAND_MAX) * 2.0 - 1.0);
    }
    
    // Initialize output layer weights and biases
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        for (int k = 0; k < OUTPUT_NEURONS; k++) {
            nn->output_weights[j][k] = FXP_FROM_FLOAT(((float)rand() / RAND_MAX) * 2.0 - 1.0);
        }
    }
    
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        nn->output_bias[k] = FXP_FROM_FLOAT(((float)rand() / RAND_MAX) * 2.0 - 1.0);
    }

    nn->lr = FXP_FROM_FLOAT(LEARNING_RATE);
}

// Forward pass through the network
void forward_pass(NeuralNetwork *nn, fxp inputs[INPUT_NEURONS]) {
    // Calculate hidden layer activations
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        fxp activation = nn->hidden_bias[j];
        for (int i = 0; i < INPUT_NEURONS; i++) {
            activation += fxp_mul(inputs[i], nn->hidden_weights[i][j]);
        }

        nn->hidden_layer[j] = sigmoid(activation);
    }
    
    // Calculate output layer activations
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        fxp activation = nn->output_bias[k];
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            activation += fxp_mul(nn->hidden_layer[j], nn->output_weights[j][k]);
        }

        nn->output_layer[k] = sigmoid(activation);
    }
}

// Backpropagation to update weights
void backpropagate(NeuralNetwork *nn, fxp inputs[INPUT_NEURONS], fxp target) {
    // Output layer error
    fxp output_errors[OUTPUT_NEURONS];
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        output_errors[k] = fxp_mul((target - nn->output_layer[k]), sigmoid_derivative(nn->output_layer[k]));
    }
    
    // Hidden layer error
    fxp hidden_errors[HIDDEN_NEURONS];
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        hidden_errors[j] = FXP_ZERO;
        for (int k = 0; k < OUTPUT_NEURONS; k++) {
            hidden_errors[j] += fxp_mul(output_errors[k], nn->output_weights[j][k]);
        }

        hidden_errors[j] = fxp_mul(hidden_errors[j], sigmoid_derivative(nn->hidden_layer[j]));
    }
    
    // Update output weights
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        for (int k = 0; k < OUTPUT_NEURONS; k++) {
            nn->output_weights[j][k] += fxp_mul(fxp_mul(nn->lr, output_errors[k]), nn->hidden_layer[j]);
        }
    }
    
    // Update output bias
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        nn->output_bias[k] += fxp_mul(nn->lr, output_errors[k]);
    }
    
    // Update hidden weights
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            nn->hidden_weights[i][j] += fxp_mul(fxp_mul(nn->lr, hidden_errors[j]), inputs[i]);
        }
    }
    
    // Update hidden bias
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        nn->hidden_bias[j] += fxp_mul(nn->lr, hidden_errors[j]);
    }
}

// Train the network
void train(NeuralNetwork *nn) {
    // XOR training data
    fxp training_inputs[4][INPUT_NEURONS] = {
        {FXP_ZERO, FXP_ZERO},
        {FXP_ZERO, FXP_ONE},
        {FXP_ONE, FXP_ZERO},
        {FXP_ONE, FXP_ONE}
    };
    
    fxp training_outputs[4] = {FXP_ZERO, FXP_ONE, FXP_ONE, FXP_ZERO};
    
    printf("Starting training for %d epochs...\n", MAX_EPOCHS);
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        fxp total_error = FXP_ZERO;
        
        // Log every 100 epochs
        int log_epoch = (epoch % 1000 == 0) || epoch == MAX_EPOCHS - 1;
        
        if (log_epoch) {
            printf("\n--- Epoch %d ---\n", epoch);
        }
        
        for (int sample = 0; sample < 4; sample++) {
            // Forward pass
            forward_pass(nn, training_inputs[sample]);
            
            // Calculate error
            fxp error = training_outputs[sample] - nn->output_layer[0];
            total_error += fxp_abs(error);
            
            // Log sample details
            if (log_epoch) {
                fxptoa(fxptoa_buff0, training_inputs[sample][0]);
                fxptoa(fxptoa_buff1, training_inputs[sample][1]);
                fxptoa(fxptoa_buff2, training_outputs[sample]);
                fxptoa(fxptoa_buff3, nn->output_layer[0]);
                fxptoa(fxptoa_buff4, error);
                printf("Sample [%s, %s] -> Expected: %s, Predicted: %s, Error: %s\n", 
                       fxptoa_buff0, fxptoa_buff1, fxptoa_buff2, fxptoa_buff3, fxptoa_buff4);
            }
            
            // Backpropagation
            backpropagate(nn, training_inputs[sample], training_outputs[sample]);
        }
        
        // Log epoch summary
        if (log_epoch) {
            fxptoa(fxptoa_buff0, total_error);
            printf("Total error: %s\n", fxptoa_buff0);
            
            // Log a few weights to see how they change
            fxptoa(fxptoa_buff0, nn->hidden_weights[0][0]);
            fxptoa(fxptoa_buff1, nn->output_weights[0][0]);
            printf("Sample weights - Hidden[0][0]: %s, Output[0][0]: %s\n", 
                   fxptoa_buff0, fxptoa_buff1);
        }
    }
}

// Test the trained network
void test(NeuralNetwork *nn) {
    fxp test_inputs[4][INPUT_NEURONS] = {
        {FXP_ZERO, FXP_ZERO},
        {FXP_ZERO, FXP_ONE},
        {FXP_ONE, FXP_ZERO},
        {FXP_ONE, FXP_ONE}
    };
    
    fxp expected_outputs[4] = {FXP_ZERO, FXP_ONE, FXP_ONE, FXP_ZERO};
    
    printf("\n--- Final Results ---\n");
    
    for (int i = 0; i < 4; i++) {
        forward_pass(nn, test_inputs[i]);
        fxptoa(fxptoa_buff0, test_inputs[i][0]);
        fxptoa(fxptoa_buff1, test_inputs[i][1]);
        fxptoa(fxptoa_buff2, nn->output_layer[0]);
        fxptoa(fxptoa_buff3, expected_outputs[i]);
        printf("Input: [%s, %s] -> Output: %s (Expected: %s)\n", 
               fxptoa_buff0, fxptoa_buff1, fxptoa_buff2, fxptoa_buff3);
    }
}

// Log the final network structure
void log_network(NeuralNetwork *nn) {
    printf("\n--- Final Network Weights and Biases ---\n");
    
    printf("Hidden Layer Weights:\n");
    for (int i = 0; i < INPUT_NEURONS; i++) {
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            fxptoa(fxptoa_buff0, nn->hidden_weights[i][j]);
            printf("w_h[%d][%d] = %s\n", i, j, fxptoa_buff0);
        }
    }
    
    printf("\nHidden Layer Biases:\n");
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        fxptoa(fxptoa_buff0, nn->hidden_bias[j]);
        printf("b_h[%d] = %s\n", j, fxptoa_buff0);
    }
    
    printf("\nOutput Layer Weights:\n");
    for (int j = 0; j < HIDDEN_NEURONS; j++) {
        for (int k = 0; k < OUTPUT_NEURONS; k++) {
            fxptoa(fxptoa_buff0, nn->output_weights[j][k]);
            printf("w_o[%d][%d] = %s\n", j, k, fxptoa_buff0);
        }
    }
    
    printf("\nOutput Layer Biases:\n");
    for (int k = 0; k < OUTPUT_NEURONS; k++) {
        fxptoa(fxptoa_buff0, nn->output_bias[k]);
        printf("b_o[%d] = %s\n", k, fxptoa_buff0);
    }
}

int main() {
    fxptoa(fxptoa_buff0, FXP_FROM_FLOAT(0.5678f));
    printf("'%s'\n", fxptoa_buff0);
    exit(3);
    // Create and initialize the neural network
    NeuralNetwork nn;
    init_network(&nn);
    
    // Train the network
    train(&nn);
    
    // Log the final network structure
    log_network(&nn);
    
    // Test the network
    test(&nn);
    
    return 0;
}