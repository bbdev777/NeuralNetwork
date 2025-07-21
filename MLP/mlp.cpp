#include "mlp.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

using namespace std;

double MLP::random_double(double min, double max)
{
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(min, max);
    return dist(gen);
}

MLP::MLP(const vector<size_t> &neurons, 
        const std::vector<std::function<double(double)>> &activations, 
        const std::vector<std::function<double(double)>> &activationDerivatives,
        double maxBiasValue,
        double alpha)
    : activations(activations), activationDerivatives(activationDerivatives), alpha(alpha)
{

    if (neurons.size() < 2)
    {
        throw invalid_argument("Network must have at least 2 layers (input and output)");
    }

    if (activations.size() != neurons.size() - 1)
    {
        throw invalid_argument("Number of activations must be equal to number of layers - 1");
    }

    bias.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        bias[i].resize(neurons[i]);
        for (auto &val : bias[i])
        {
            val = random_double(-maxBiasValue, maxBiasValue);
        }
    }

    weights.resize(neurons.size() - 1);
    for (size_t i = 0; i < neurons.size() - 1; ++i)
    {
        double scale = sqrt(2.0 / neurons[i]);
        weights[i].resize(neurons[i]);
        for (size_t j = 0; j < neurons[i]; ++j)
        {
            weights[i][j].resize(neurons[i + 1]);
            for (size_t k = 0; k < neurons[i + 1]; ++k)
            {
                weights[i][j][k] = 1.0;//random_double(-scale, scale);
            }
        }
    }

    data.resize(neurons.size());
    for (size_t i = 0; i < neurons.size(); ++i)
    {
        data[i].resize(neurons[i]);
    }
}

void MLP::calculate()
{
    for (size_t layer = 1; layer < data.size(); layer++)
    {
        for (size_t neuron = 0, neuronCount = data[layer].size(); neuron < neuronCount; neuron++)
        {
            double sum = bias[layer][neuron];
            for (size_t prev_neuron = 0, neuronCount1 = data[layer - 1].size(); prev_neuron < neuronCount1; prev_neuron++)
            {
                sum += weights[layer - 1][prev_neuron][neuron] * data[layer - 1][prev_neuron];
            }
            data[layer][neuron] = sum;
        }

        auto activation = activations[layer - 1];

        transform(data[layer].begin(), data[layer].end(), data[layer].begin(),
                  [&activation](double x)
                  { return activation(x); });
    }
}

vector<double> MLP::train(const vector<double> &input, const vector<double> &target, double learning_rate)
{
    if (input.size() != data[0].size())
    {
        throw invalid_argument("Input size doesn't match network input layer size");
    }
    if (target.size() != data.back().size())
    {
        throw invalid_argument("Target size doesn't match network output layer size");
    }

    data[0] = input;
    calculate();

    vector<vector<double>> gradients(data.size());
    for (size_t i = 0; i < data.size(); ++i)
    {
        gradients[i].resize(data[i].size(), 0.0);
    }

    for (size_t i = 0; i < data.back().size(); ++i)
    {
        gradients.back()[i] = data.back()[i] - target[i];
    }

    for (size_t layer = data.size() - 1; layer > 0; --layer)
    {
        auto activationDerivative = activationDerivatives[layer - 1];

        for (size_t neuron = 0; neuron < data[layer].size(); ++neuron)
        {

            double derivative = activationDerivative(data[layer][neuron]);
  
            gradients[layer][neuron] *= derivative;

            for (size_t prev_neuron = 0; prev_neuron < data[layer - 1].size(); ++prev_neuron)
            {
                gradients[layer - 1][prev_neuron] += gradients[layer][neuron] *
                                                     weights[layer - 1][prev_neuron][neuron];
            }
        }
    }

    double max_grad = 1.0;
    for (auto &layer_grads : gradients)
    {
        for (auto &grad : layer_grads)
        {
            grad = max(-max_grad, min(max_grad, grad));
        }
    }

    for (size_t layer = 0; layer < weights.size(); ++layer)
    {
        for (size_t prev_neuron = 0; prev_neuron < weights[layer].size(); ++prev_neuron)
        {
            for (size_t neuron = 0; neuron < weights[layer][prev_neuron].size(); ++neuron)
            {
                weights[layer][prev_neuron][neuron] -= learning_rate *
                                                       gradients[layer + 1][neuron] *
                                                       data[layer][prev_neuron];
            }
        }
    }

    for (size_t layer = 1; layer < bias.size(); ++layer)
    {
        for (size_t neuron = 0; neuron < bias[layer].size(); ++neuron)
        {
            bias[layer][neuron] -= learning_rate * gradients[layer][neuron];
        }
    }

    return data.back();
}

vector<double> MLP::predict(const vector<double> &input)
{
    if (input.size() != data[0].size())
    {
        throw invalid_argument("Input size doesn't match network input layer size");
    }
    data[0] = input;
    calculate();
    return data.back();
}

void MLP::save_weights(const string &filename) const
{
    ofstream file(filename, ios::binary);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open file for saving weights");
    }

    size_t num_layers = bias.size();
    file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

    for (const auto &layer : bias)
    {
        size_t layer_size = layer.size();
        file.write(reinterpret_cast<const char *>(&layer_size), sizeof(layer_size));
    }

    for (const auto &layer : weights)
    {
        for (const auto &neuron_weights : layer)
        {
            file.write(reinterpret_cast<const char *>(neuron_weights.data()),
                       neuron_weights.size() * sizeof(double));
        }
    }

    for (const auto &layer : bias)
    {
        file.write(reinterpret_cast<const char *>(layer.data()),
                   layer.size() * sizeof(double));
    }
}

void MLP::load_weights(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        throw runtime_error("Cannot open file for loading weights");
    }

    size_t num_layers;
    file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));
    if (num_layers != bias.size())
    {
        throw runtime_error("Number of layers does not match");
    }

    vector<size_t> layer_sizes(num_layers);
    for (size_t i = 0; i < num_layers; ++i)
    {
        file.read(reinterpret_cast<char *>(&layer_sizes[i]), sizeof(size_t));
        if (layer_sizes[i] != bias[i].size())
        {
            throw runtime_error("Layer size does not match");
        }
    }

    vector<string> saved_activations;
    for (size_t i = 0; i < num_layers - 1; ++i)
    {
        size_t act_size;
        file.read(reinterpret_cast<char *>(&act_size), sizeof(act_size));
        string act(act_size, ' ');
        file.read(&act[0], act_size);
        saved_activations.push_back(act);
    }

    for (auto &layer : weights)
    {
        for (auto &neuron_weights : layer)
        {
            file.read(reinterpret_cast<char *>(neuron_weights.data()),
                      neuron_weights.size() * sizeof(double));
        }
    }

    for (auto &layer : bias)
    {
        file.read(reinterpret_cast<char *>(layer.data()),
                  layer.size() * sizeof(double));
    }
}
