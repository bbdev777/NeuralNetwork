#include <math.h>
#include <vector>

namespace MLPActivators
{
    using namespace std;

    const double alpha = 1.0;
    const double leaky_relu_alpha = 0.01;
    const double selu_scale = 1.0507;
    const double selu_alpha = 1.67326;

    static double relu(double x) { return max(0.0, x); }
    static double leaky_relu(double x) { return x > 0 ? x : leaky_relu_alpha * x; }
    static double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double tanh_activation(double x) { return tanh(x); }
    static double swish(double x) { return x * sigmoid(x); }
    static double elu(double x) { return x >= 0 ? x : alpha * (exp(x) - 1); }
    static double silu(double x) { return x * sigmoid(x); }
    static double gelu(double x)
    {
        return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
    }

    static double selu(double x)
    {
        return selu_scale * (x > 0 ? x : selu_alpha * (exp(x) - 1));
    }

    static double softplus(double x) { return log(1.0 + exp(x)); }
    static double softsign(double x) { return x / (1.0 + abs(x)); }
    static double binary_step(double x) { return x < 0 ? 0 : 1; }
    static double identity(double x) { return x; }

    static vector<double> softmax(const vector<double> &z)
    {
        vector<double> res(z.size());
        double max_z = *max_element(z.begin(), z.end());
        double sum_exp = 0.0;

        for (size_t i = 0; i < z.size(); ++i)
        {
            res[i] = exp(z[i] - max_z);
            sum_exp += res[i];
        }

        for (auto &val : res)
            val /= sum_exp;
        return res;
    }

    static double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }
    static double leaky_relu_derivative(double x) { return x > 0 ? 1.0 : leaky_relu_alpha; }
    static double sigmoid_derivative(double x)
    {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    static double tanh_derivative(double x)
    {
        return 1.0 - tanh_activation(x) * tanh_activation(x);
    }

    static double swish_derivative(double x)
    {
        double s = sigmoid(x);
        return s + x * s * (1 - s);
    }

    static double elu_derivative(double x) { return x >= 0 ? 1.0 : alpha * exp(x); }
    static double silu_derivative(double x)
    {
        double s = sigmoid(x);
        return s + x * s * (1 - s);
    }

    static double gelu_derivative(double x)
    {
        double cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * pow(x, 3))));
        return cdf + x * (1.0 / sqrt(2 * M_PI)) * exp(-0.5 * x * x) * (1 + 0.134145 * x * x);
    }

    static double selu_derivative(double x)
    {
        return selu_scale * (x > 0 ? 1.0 : selu_alpha * exp(x));
    }

    static double softplus_derivative(double x) { return 1.0 / (1.0 + exp(-x)); }
    static double softsign_derivative(double x)
    {
        double denom = 1.0 + abs(x);
        return 1.0 / (denom * denom);
    }
    
    static double binary_step_derivative(double x) { return 0.0; }
    static double identity_derivative(double x) { return 1.0; }

    static vector<double>  softmax_derivative(const vector<double> &z)
    {
        vector<double> sm = softmax(z);
        vector<double> derivative(z.size() * z.size(), 0.0);
        for (size_t i = 0; i < z.size(); ++i)
        {
            for (size_t j = 0; j < z.size(); ++j)
            {
                derivative[i * z.size() + j] = sm[i] * ((i == j) ? 1.0 - sm[j] : -sm[j]);
            }
        }
        return derivative;
    }
};

