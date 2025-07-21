#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include "mlp_activators.h"

class MLP {
private:
    std::vector<std::vector<double>> bias;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> data;
    std::vector<std::function<double(double)>> activations;
    std::vector<std::function<double(double)>> activationDerivatives;


    double alpha = 1.0;
  
   
    void calculate();
    double random_double(double min, double max);

public:
    
    MLP(const std::vector<size_t>& neurons,
        const std::vector<std::function<double(double)>> &activations,
        const std::vector<std::function<double(double)>> &activationDerivatives,
        double maxBiasValue,
        double alpha = 1.0);

   
    std::vector<double> train(const std::vector<double>& input,
                             const std::vector<double>& target,
                             double learning_rate = 0.01);

    std::vector<double> predict(const std::vector<double>& input);


    void save_weights(const std::string& filename) const;
    void load_weights(const std::string& filename);
};

#endif // MLP_H
