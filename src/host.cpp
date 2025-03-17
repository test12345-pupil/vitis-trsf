#include "cmdlineparser.h"
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define S 32
#define D_INPUT 64
#define D_MODEL 64
#define D_FFN 256
#define N_BLOCK 6
#define N_LAYERS 3

void softmax(float* input, int size) {
    float max_val = input[0];
    float sum = 0.0f;

    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    for (int i = 0; i < size; i++) {
        input[i] = (input[i] - max_val);
        sum += input[i];
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void layer_norm(float* input, float* output, int size) {
    float mean = 0.0f;
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    for (int i = 0; i < size; i++) {
        variance += (input[i] - mean) * (input[i] - mean);
    }
    variance /= size;
    float std_dev = std::sqrt(variance + 1e-7f);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / std_dev;
    }
}

void host_mha(float* D, float* M_Q, float* M_K, float* M_V, float* Output, int s, int d_model) {
    float Q[S][D_MODEL];
    float K[S][D_MODEL];
    float V[S][D_MODEL];
    float Score[S][S];

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            Q[i][j] = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                Q[i][j] += D[i * D_MODEL + k] * M_Q[k * D_MODEL + j];
            }
        }
    }

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            K[i][j] = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                K[i][j] += D[i * D_MODEL + k] * M_K[k * D_MODEL + j];
            }
        }
    }

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            V[i][j] = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                V[i][j] += D[i * D_MODEL + k] * M_V[k * D_MODEL + j];
            }
        }
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D_MODEL));
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            Score[i][j] = 0.0f;
            for (int k = 0; k < D_MODEL; k++) {
                Score[i][j] += Q[i][k] * K[j][k];
            }
            Score[i][j] *= scale;
        }
    }

    for (int i = 0; i < S; i++) {
        softmax(Score[i], S);
    }

    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            Output[i * D_MODEL + j] = 0.0f;
            for (int k = 0; k < S; k++) {
                Output[i * D_MODEL + j] += Score[i][k] * V[k][j];
            }
        }
    }
}

void host_transformer_block(float* input, float* output,
                            float* M_Q, float* M_K, float* M_V,
                            float* W1, float* W2,
                            int s, int d_input, int d_model, int d_ffn) {
    std::vector<float> temp1(s * d_model, 0.0f);
    std::vector<float> temp2(s * d_model, 0.0f);

    host_mha(input, M_Q, M_K, M_V, temp1.data(), s, d_model);

    for (int i = 0; i < s * d_model; i++) {
        temp1[i] = input[i] + temp1[i];
    }
    layer_norm(temp1.data(), temp2.data(), s * d_model);

    std::vector<float> ffn_intermediate(s * d_ffn, 0.0f);
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < d_ffn; j++) {
            ffn_intermediate[i * d_ffn + j] = 0.0f;
            for (int k = 0; k < d_model; k++) {
                ffn_intermediate[i * d_ffn + j] += temp2[i * d_model + k] * W1[k * d_ffn + j];
            }
        }
    }

    for (int i = 0; i < s * d_ffn; i++) {
        ffn_intermediate[i] = std::max(0.0f, ffn_intermediate[i]);
    }

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < d_model; j++) {
            output[i * d_model + j] = 0.0f;
            for (int k = 0; k < d_ffn; k++) {
                output[i * d_model + j] += ffn_intermediate[i * d_ffn + k] * W2[k * d_model + j];
            }
        }
    }

    for (int i = 0; i < s * d_model; i++) {
        output[i] = temp2[i] + output[i];
    }
    layer_norm(output, output, s * d_model);
}

void host_transformer_layers(float* input, float* output,
                             float* M_Q, float* M_K, float* M_V,
                             float* W1, float* W2,
                             int s, int d_input, int d_model, int d_ffn) {
    // current存放当前层的输入，next存放当前层计算结果
    std::vector<float> current(input, input + s * d_model);
    std::vector<float> next(s * d_model, 0.0f);
    for (int l = 0; l < N_LAYERS; l++) {
        host_transformer_block(current.data(), next.data(),
                               M_Q + l * (d_input * d_model),
                               M_K + l * (d_input * d_model),
                               M_V + l * (d_input * d_model),
                               W1 + l * (d_model * d_ffn),
                               W2 + l * (d_ffn * d_model),
                               s, d_input, d_model, d_ffn);
        current.swap(next);
    }
    std::copy(current.begin(), current.end(), output);
}

int main(int argc, char** argv) {
    sda::utils::CmdLineParser parser;
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    auto device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binaryFile);
    auto krnl = xrt::kernel(device, uuid, "transformer");

    size_t size_input_bytes = sizeof(float) * S * D_INPUT;
    size_t size_output_bytes = sizeof(float) * S * D_MODEL;
    // 多层时，每层权重需要额外扩展第一维
    size_t size_M_Q_bytes = sizeof(float) * N_LAYERS * D_INPUT * D_MODEL;
    size_t size_M_K_bytes = sizeof(float) * N_LAYERS * D_INPUT * D_MODEL;
    size_t size_M_V_bytes = sizeof(float) * N_LAYERS * D_INPUT * D_MODEL;
    size_t size_W1_bytes  = sizeof(float) * N_LAYERS * D_MODEL * D_FFN;
    size_t size_W2_bytes  = sizeof(float) * N_LAYERS * D_FFN * D_MODEL;

    auto bo_M_Q = xrt::bo(device, size_M_Q_bytes, krnl.group_id(2));
    auto bo_M_K = xrt::bo(device, size_M_K_bytes, krnl.group_id(3));
    auto bo_M_V = xrt::bo(device, size_M_V_bytes, krnl.group_id(4));
    auto bo_W1  = xrt::bo(device, size_W1_bytes,  krnl.group_id(5));
    auto bo_W2  = xrt::bo(device, size_W2_bytes,  krnl.group_id(6));
    auto bo_M_Q_map = bo_M_Q.map<float*>();
    auto bo_M_K_map = bo_M_K.map<float*>();
    auto bo_M_V_map = bo_M_V.map<float*>();
    auto bo_W1_map  = bo_W1.map<float*>();
    auto bo_W2_map  = bo_W2.map<float*>();

    // 初始化各层的权重数据（简单赋值，实际可根据需要调整）
    for (int l = 0; l < N_LAYERS; l++) {
        for (int i = 0; i < D_INPUT * D_MODEL; i++) {
            bo_M_Q_map[l * (D_INPUT * D_MODEL) + i] = static_cast<float>(i) / (D_INPUT * D_MODEL);
            bo_M_K_map[l * (D_INPUT * D_MODEL) + i] = static_cast<float>(i) / (D_INPUT * D_MODEL);
            bo_M_V_map[l * (D_INPUT * D_MODEL) + i] = static_cast<float>(i) / (D_INPUT * D_MODEL);
        }
        for (int i = 0; i < D_MODEL * D_FFN; i++) {
            bo_W1_map[l * (D_MODEL * D_FFN) + i] = static_cast<float>(i) / (D_MODEL * D_FFN);
        }
        for (int i = 0; i < D_FFN * D_MODEL; i++) {
            bo_W2_map[l * (D_FFN * D_MODEL) + i] = static_cast<float>(i) / (D_FFN * D_MODEL);
        }
    }
    const int num_tests = 5;
    for (int test = 0; test < num_tests; test++) {


        bo_M_Q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_M_K.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_M_V.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_W1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_W2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto bo_input = xrt::bo(device, size_input_bytes, krnl.group_id(0));
        auto bo_output = xrt::bo(device, size_output_bytes, krnl.group_id(1));
        auto bo_input_map = bo_input.map<float*>();
        auto bo_output_map = bo_output.map<float*>();

        for (int i = 0; i < S * D_INPUT; i++) {
            bo_input_map[i] = static_cast<float>((1+1)*i % (S * D_INPUT)) / (S * D_INPUT);
        }

        bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        std::fill(bo_output_map, bo_output_map + S * D_MODEL, 0.0f);
        bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // 调用kernel，注意传入的权重指针现在包含了多层数据
        auto run = krnl(bo_input, bo_output, bo_M_Q, bo_M_K, bo_M_V, bo_W1, bo_W2, S, D_INPUT, D_MODEL, D_FFN);
        run.wait(); 
     
        bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        std::vector<float> host_output(S * D_MODEL, 0.0f);
        host_transformer_layers(bo_input_map, host_output.data(),
                                bo_M_Q_map, bo_M_K_map, bo_M_V_map,
                                bo_W1_map, bo_W2_map,
                                S, D_INPUT, D_MODEL, D_FFN);

        bool passed = true;
        for (int i = 0; i < S * D_MODEL; i++) {
            std::cout << bo_output_map[i] << ' ' << host_output[i] << '\n';
            if (std::abs(bo_output_map[i] - host_output[i]) > 1e-5) {
                passed = false;
                std::cout << "Mismatch at index " << i << ": Kernel=" << bo_output_map[i]
                          << ", Host=" << host_output[i] << std::endl;
                break;
            }
        }

        if (passed) {
            std::cout << "Test " << test << " passed!" << std::endl;
        } else {
            std::cout << "Test " << test << " failed!" << std::endl;
        }
    }

    return 0;
}
