#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iomanip>   // 用于 setprecision
#include <string>
#include <fstream>
#define input_u 1e2
#define output_u 1e2
#define input_dimension 1e3
#define output_dimension 1
//#define USE_PRETRAINED  // 定义此宏以使用预训练模型（需在 createPreTrained 中填入权重）

enum class Activation { ReLU, LeakyReLU, Linear };

double activate(double x, Activation type) {
    switch (type) {
    case Activation::ReLU:       return x > 0 ? x : 0.0;
    case Activation::LeakyReLU:  return x > 0 ? x : 0.01 * x;
    case Activation::Linear:     return x;
    default: return x;
    }
}

double activate_derivative(double x, Activation type) {
    switch (type) {
    case Activation::ReLU:       return x > 0 ? 1.0 : 0.0;
    case Activation::LeakyReLU:  return x > 0 ? 1.0 : 0.01;
    case Activation::Linear:     return 1.0;
    default: return 1.0;
    }
}

// 数据生成器（根据题目要求生成符合条件的输入数据）
std::vector<double> data_generator()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10);
    std::uniform_real_distribution<double> dist2(0.0, input_dimension);
	int n = std::round(dist2(gen));
    std::vector<double> input(n);
    for (int i=0;i<n;i++)
    {
        input[i] = std::round(dist(gen));
    }
    return input;
}

void normalize_input(std::vector<double>& input)
{
    // 归一化输入（根据题目数据范围进行缩放）
    for (auto& val : input)
    {
        val /= input_u;
    }
}
void normalize_target(std::vector<double>& target)
{
    // 归一化目标输出（根据题目数据范围进行缩放）
    for (auto& val : target)
    {
        val /= output_u;
    }
}
void denormalize_output(std::vector<double>& output)
{
    // 反归一化输出（将网络输出还原到原始范围）
    for (auto& val : output)
    {
        val *= output_u;
    }
}
// 神经网络类
class NeuralNetwork {
public:
    // 构造函数1：随机初始化（原构造）
    NeuralNetwork(const std::vector<size_t>& layer_sizes,
        const std::vector<Activation>& activations)
        : layer_sizes_(layer_sizes), activations_(activations) {
        if (layer_sizes.size() < 2)
            throw std::invalid_argument("At least input and output layers required");
        if (layer_sizes.size() != activations.size())
            throw std::invalid_argument("Each layer must have an activation function");

        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            size_t rows = layer_sizes[i];
            size_t cols = layer_sizes[i - 1];
            double stddev = std::sqrt(2.0 / cols);  // He初始化标准差

            std::normal_distribution<double> dist(0.0, stddev);
            std::vector<std::vector<double>> w(rows, std::vector<double>(cols));
            for (auto& row : w)
                for (auto& val : row)
                    val = dist(gen);
            weights_.push_back(std::move(w));

            // 偏置初始化为0（或很小的正数，如0.01）
            std::vector<double> b(rows, 0.0);
            biases_.push_back(std::move(b));
        }
    }

    // 构造函数2：使用预训练的权重和偏置
    NeuralNetwork(const std::vector<size_t>& layer_sizes,
        const std::vector<Activation>& activations,
        const std::vector<std::vector<std::vector<double>>>& weights,
        const std::vector<std::vector<double>>& biases)
        : layer_sizes_(layer_sizes), activations_(activations),
        weights_(weights), biases_(biases) {
        // 维度验证
        if (layer_sizes.size() < 2)
            throw std::invalid_argument("At least input and output layers required");
        if (layer_sizes.size() != activations.size())
            throw std::invalid_argument("Each layer must have an activation function");
        if (weights.size() != layer_sizes.size() - 1)
            throw std::invalid_argument("Number of weight matrices mismatch");
        if (biases.size() != layer_sizes.size() - 1)
            throw std::invalid_argument("Number of bias vectors mismatch");

        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i].size() != layer_sizes[i + 1])
                throw std::invalid_argument("Weight matrix row count mismatch for layer " + std::to_string(i));
            for (const auto& row : weights[i]) {
                if (row.size() != layer_sizes[i])
                    throw std::invalid_argument("Weight matrix column count mismatch for layer " + std::to_string(i));
            }
            if (biases[i].size() != layer_sizes[i + 1])
                throw std::invalid_argument("Bias vector size mismatch for layer " + std::to_string(i));
        }
    }

    // 静态工厂方法：返回一个预训练的网络（用户需要在此填入训练好的权重）
    static NeuralNetwork createPreTrained(std::vector<size_t> layers, std::vector<Activation> activations)
    {

        // TODO: 将下面两个数组替换为从 printWeights() 输出的内容
        // 预训练权重
        std::vector<std::vector<std::vector<double>>> pre_weights = {
    { // layer 0 (10x2)
        {-0.378139546714594, -0.423619607066733},
        {-0.457030816334783, -0.300863386638082},
        {0.263918166206942, 0.176519610203784},
        {0.724902310858908, 0.73986875536507},
        {-0.569935608869162, -0.61365308086858},
        {0.628937367067075, -0.65865163654934},
        {-0.676222682114548, -0.689090402970485},
        {-0.631643350639555, 0.660282585409484},
        {0.562993717367589, 0.513091435719127},
        {-0.148979893554832, -0.468974441993705}
    },
    { // layer 1 (2x10)
        {-0.074994260447833, -0.791622555816683, -0.344942554030911, 0.731914815842965, -0.00168419026358969, 0.00127666666868898, -0.767226438261317, -0.000568474339071189, 0.0897168888713446, -0.373528826336658},
        {0.65451245927931, -0.0448302225715979, -0.39202737321295, 0.322369044414213, 0.782598942652887, -0.863096671727844, 0.273503001127417, -0.880066404619507, 0.751808890661131, -0.249760065199086}
    }
        };

        std::vector<std::vector<double>> pre_biases = {
            { // layer 0 biases
                -0.223118716317892, 0.722231986285083, -0.43973175879122, -0.0624471337560772, 0.593506582084621, 0.371836652873577, 0.0622020090775132, 0.3358520215639, 0.651216100436663, 0.606318589121562
            },
            { // layer 1 biases
                0.785795836576222, -0.207374319179951
            }
        };

        return NeuralNetwork(layers, activations, pre_weights, pre_biases);
    }

    // 前向传播：给定输入，计算各层输出，并保存中间值用于反向传播
    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != layer_sizes_[0])
            throw std::invalid_argument("Input size mismatch");

        layer_inputs_.clear();
        layer_outputs_.clear();
        layer_inputs_.push_back(input);  // 第一层输入即原始输入

        std::vector<double> current = input;
        for (size_t i = 0; i < weights_.size(); ++i) {
            const auto& W = weights_[i];
            const auto& b = biases_[i];
            Activation act = activations_[i + 1];  // 当前层激活（注意索引：输入层无激活）
            std::vector<double> next(W.size(), 0.0);

            // 计算加权和：next_j = sum_k W[j][k] * current[k] + b[j]
            for (size_t j = 0; j < W.size(); ++j) {
                double sum = b[j];
                for (size_t k = 0; k < current.size(); ++k) {
                    sum += W[j][k] * current[k];
                }
                // 应用激活函数
                next[j] = activate(sum, act);
            }

            layer_inputs_.push_back(current);   // 当前层的输入（即上一层的输出）
            layer_outputs_.push_back(next);     // 当前层的输出
            current = std::move(next);
        }
        return current;  // 最终输出
    }

    // 反向传播：根据目标输出更新权重和偏置（使用学习率 lr）
    void backward(const std::vector<double>& target, double lr) {
        if (target.size() != layer_sizes_.back())
            throw std::invalid_argument("Target size mismatch");

        size_t L = weights_.size();  // 隐藏层+输出层的数量

        // 存储每一层输出的误差 (delta)
        std::vector<std::vector<double>> deltas(L);

        // ----- 1. 计算输出层误差 -----
        deltas[L - 1].resize(layer_outputs_.back().size());
        for (size_t i = 0; i < layer_outputs_.back().size(); ++i) {
            double output = layer_outputs_.back()[i];
            double derivative = activate_derivative(output, activations_.back());
            deltas[L - 1][i] = (output - target[i]) * derivative;
        }

        // ----- 2. 反向传播，计算前面各层的误差 -----
        for (int l = L - 2; l >= 0; --l) {
            const auto& W_next = weights_[l + 1];           // 下一层的权重
            const auto& next_delta = deltas[l + 1];
            const auto& output_this = layer_outputs_[l];  // 当前层的输出
            deltas[l].resize(output_this.size(), 0.0);
            for (size_t k = 0; k < output_this.size(); ++k) {
                double error = 0.0;
                for (size_t j = 0; j < W_next.size(); ++j) {
                    error += next_delta[j] * W_next[j][k];
                }
                double derivative = activate_derivative(output_this[k], activations_[l + 1]);
                deltas[l][k] = error * derivative;
            }
        }

        // ----- 3. 更新所有层的权重和偏置 -----
        for (size_t l = 0; l < L; ++l) {
            const auto& input = layer_inputs_[l + 1];   // 当前层的输入
            const auto& delta = deltas[l];
            auto& W = weights_[l];
            auto& b = biases_[l];
            for (size_t j = 0; j < W.size(); ++j) {
                for (size_t k = 0; k < input.size(); ++k) {
                    W[j][k] -= lr * delta[j] * input[k];
                }
                b[j] -= lr * delta[j];
            }
        }
    }

    // 训练函数：使用用户提供的 oracle 生成训练数据，进行在线学习
    void train(const std::function<std::vector<double>(const std::vector<double>&)>& oracle,
        size_t input_dim, size_t output_dim,
        size_t iterations, double learning_rate, int print_interval = 1000) {
        if (input_dim != layer_sizes_[0] || output_dim != layer_sizes_.back())
            throw std::invalid_argument("Network dimensions do not match oracle");

        for (size_t iter = 0; iter < iterations; ++iter) {
            // 随机生成输入
            std::vector<double> input;

            //在这里自己思考怎么搓随机输入，别忘了input_dim是数据的最大范围。所以我觉得可以将多出来的直接编码成0
            //而且别忘了屌丝题目对数据有特殊要求（树、图），所以不能直接乱随机。学学怎么生成符合要求的数据吧。
            input = data_generator();

            // 调用 oracle 获得目标输出
            std::vector<double> target = oracle(input);

            //归一化
            normalize_input(input);
            normalize_target(target);

			input.resize(input_dim, 0.0); // 如果输入不足，补零

            // 前向传播
            std::vector<double> output = forward(input);

            // 反向传播更新
            backward(target, learning_rate);

            // 打印损失（可选）
            if (print_interval > 0 && iter % print_interval == 0) {
                double loss = 0.0;
                for (size_t i = 0; i < output.size(); ++i) {
                    double diff = output[i] - target[i];
                    loss += diff * diff;
                }
                loss /= output.size();
                std::cout << "\nIteration " << iter << ", MSE: " << loss <<"\n\n";
            }
        }
    }

    // 获取网络输出（直接调用 forward）
    std::vector<double> predict(const std::vector<double>& input) {
        return forward(input);
    }

    // 打印当前权重和偏置，格式可直接复制到 createPreTrained() 中
    void printWeights() const {
        // 打开文件
        std::ofstream outFile("weights.txt");
        if (!outFile.is_open()) {
            std::cerr << "Error: Cannot open file weights.txt for writing.\n";
            return; // 或抛出异常
        }

        outFile << std::setprecision(15); // 高精度输出

        // 打印权重
        outFile << "std::vector<std::vector<std::vector<double>>> pre_weights = {\n";
        for (size_t l = 0; l < weights_.size(); ++l) {
            outFile << "    { // layer " << l << " (" << weights_[l].size() << "x" << (l == 0 ? layer_sizes_[0] : layer_sizes_[l]) << ")\n";
            for (const auto& row : weights_[l]) {
                outFile << "        {";
                for (size_t i = 0; i < row.size(); ++i) {
                    if (i > 0) outFile << ", ";
                    outFile << row[i];
                }
                outFile << "}";
                if (&row != &weights_[l].back()) outFile << ",";
                outFile << "\n";
            }
            outFile << "    }";
            if (l < weights_.size() - 1) outFile << ",";
            outFile << "\n";
        }
        outFile << "};\n\n";

        // 打印偏置
        outFile << "std::vector<std::vector<double>> pre_biases = {\n";
        for (size_t l = 0; l < biases_.size(); ++l) {
            outFile << "    { // layer " << l << " biases\n        ";
            for (size_t i = 0; i < biases_[l].size(); ++i) {
                if (i > 0) outFile << ", ";
                outFile << biases_[l][i];
            }
            outFile << "\n    }";
            if (l < biases_.size() - 1) outFile << ",";
            outFile << "\n";
        }
        outFile << "};\n";

        // 文件会在 outFile 析构时自动关闭，此处可省略 close()
        // outFile.close();
    }

private:
    std::vector<size_t> layer_sizes_;                     // 每层神经元数
    std::vector<Activation> activations_;                 // 每层激活函数（索引与 layer_sizes_ 对应）
    std::vector<std::vector<std::vector<double>>> weights_; // 权重列表，weights[l][j][k]
    std::vector<std::vector<double>> biases_;               // 偏置列表，biases[l][j]

    // 缓存前向传播中的中间值（用于反向传播）
    std::vector<std::vector<double>> layer_inputs_;   // 每层的输入（第一层为原始输入）
    std::vector<std::vector<double>> layer_outputs_;  // 每层的输出（最后一层为网络输出）
};
#include <queue>
std::vector<double> my_algorithm(const std::vector<double>& input) {
    // 处理空输入的情况，返回只包含 0 的向量
    if (input.empty()) {
        return { 0.0 };
    }

	double ans = 0.0;
	for (auto val : input)
    {
        ans += val;
    }
    // 返回只包含最终总代价的向量
    return { ans };
}

int main() {
    // 定义网络结构
    std::vector<size_t> layers = { size_t(input_dimension), 2, output_dimension };
    std::vector<Activation> activations = { Activation::Linear, Activation::LeakyReLU, Activation::Linear };

#ifdef USE_PRETRAINED
    // 使用预训练模型（需提前在 createPreTrained 中填入权重）
    NeuralNetwork nn = NeuralNetwork::createPreTrained(layers,activations);
    //std::cout << "Loaded pre-trained network.\n";
#else
    // 创建并训练新网络
    NeuralNetwork nn(layers, activations);
    nn.train(my_algorithm, input_dimension, output_dimension, 1e6, 0.01, 1e5);

    // 打印训练后的权重，以便复制到 createPreTrained 中
    std::cout << "\n--- Copy the following arrays into NeuralNetwork::createPreTrained() ---\n";
    nn.printWeights();
    std::cout << "--- End of copy ---\n\n";
#endif

    while (1)
    {
        std::vector<double> inputs(input_dimension);
        double n;
        std::cin >> n;
        if (n > input_dimension) return 0;
        for (int i = 0; i < n; i++)
        {
            std::cin >> inputs[i];
        }
		normalize_input(inputs); // 归一化输入
        std::vector<double> output = nn.predict(inputs);
		denormalize_output(output); // 反归一化输出
        std::cout << output[0];

        for (int i = 0; i < n; i++)
        {
            inputs[i] *= input_u;
        }
        std::cout << "True:" << my_algorithm(inputs)[0] << std::endl;
    }
    return 0;
}