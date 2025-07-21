#include <iostream>
#include <vector>
#include <cmath>
#include "mlp.h"
#include "exec_time.h"

using namespace std;

vector<double> normalize(const vector<double>& input, int maxValue) {
    vector<double> normalized = input;
    for (auto& val : normalized) {
       val /= double(maxValue);
    }
    return normalized;
}


vector<double> denormalize(const vector<double>& output, int maxValue) {
    vector<double> denormalized = output;
    for (auto& val : denormalized) {
        val *= double(maxValue);
    }
    return denormalized;
}


void quadratic_example() 
{
    int     maxValue = 36;
    double  targetAccuracy = 0.1;
    double  learning_rate = 0.005;

    MLP mlp(
    {
        1, 
        24, 
        24, 
        24,
        24,
        1
    }, 
    {
        MLPActivators::leaky_relu,
        MLPActivators::leaky_relu,
        MLPActivators::leaky_relu,
        MLPActivators::leaky_relu,
        MLPActivators::identity
    },
    {
        MLPActivators::leaky_relu_derivative,
        MLPActivators::leaky_relu_derivative,
        MLPActivators::leaky_relu_derivative,
        MLPActivators::leaky_relu_derivative,
        MLPActivators::identity_derivative
    },
    1.5);

    vector<vector<double>> inputs;
    vector<vector<double>> targets;

    for (int x = 0; x <= maxValue; x++) {
        inputs.push_back({double(x)});
        targets.push_back({double(x*x)});
    }

    AppExecutionTimeCounter::StartMeasurement();

    int    epoch = 0;
    double total_error;
    do 
    {
        total_error = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = denormalize(mlp.train(normalize(inputs[i], maxValue), normalize(targets[i], maxValue), learning_rate), maxValue);
            double error = output[0] - targets[i][0];
            total_error += error * error;
        }

        total_error /= inputs.size();

        if (epoch % 10000 == 0) 
        {
            cout << "Эпоха: " << epoch << ", Ошибка: " << total_error << endl;
        }

        epoch ++;
    }
    while(total_error > targetAccuracy);

    double trainingTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время ренировки  (мек.) %1.3lf\n", trainingTimeSeconds);

    cout << "Результаты после обучения:" << endl;
    cout << "x   Сеть   Мат.  Разность" << endl;

    AppExecutionTimeCounter::StartMeasurement();
    for (int x = 0; x <= maxValue; x ++) {
        auto output = denormalize(mlp.predict(normalize({double(x)}, maxValue)), maxValue);
        printf("%3d %5.0lf %5d\t%2.0f\n", x, round(output[0]), x*x, double(x*x) - round(output[0]));
    }
    
    double predictTimeSeconds = AppExecutionTimeCounter::EndMeasurement();
    printf("Время вычислений (мсек.) %1.3lf\n", predictTimeSeconds * 1000.0);

    mlp.save_weights("quadratic_weights.bin");
   
}

int main() {
   
    quadratic_example();

    return 0;
}
