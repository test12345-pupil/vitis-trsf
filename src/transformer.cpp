#include <stdint.h>
#include <hls_stream.h>
#include <iostream>
#include <hls_math.h>

#define S 32
#define D_INPUT 64
#define D_MODEL 64
#define D_FFN 256
#define N_LAYERS 3

const int c_s = S;
const int c_d_input = D_INPUT;
const int c_d_model = D_MODEL;
const int c_d_ffn = D_FFN;

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
    float std_dev = hls::sqrt(variance + 1e-7f);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) / std_dev;
    }
}

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

extern "C" {
void transformer(float* input, float* output,
                 float* M_Q, float* M_K, float* M_V,
                 float* W1, float* W2,
                 int s, int d_input, int d_model, int d_ffn) {
    #pragma HLS INTERFACE m_axi port=input  bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output bundle=gmem1
    #pragma HLS INTERFACE m_axi port=M_Q    bundle=gmem2
    #pragma HLS INTERFACE m_axi port=M_K    bundle=gmem3
    #pragma HLS INTERFACE m_axi port=M_V    bundle=gmem4
    #pragma HLS INTERFACE m_axi port=W1     bundle=gmem5
    #pragma HLS INTERFACE m_axi port=W2     bundle=gmem6
    #pragma HLS INTERFACE s_axilite port=input  bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=M_Q    bundle=control
    #pragma HLS INTERFACE s_axilite port=M_K    bundle=control
    #pragma HLS INTERFACE s_axilite port=M_V    bundle=control
    #pragma HLS INTERFACE s_axilite port=W1     bundle=control
    #pragma HLS INTERFACE s_axilite port=W2     bundle=control
    #pragma HLS INTERFACE s_axilite port=s      bundle=control
    #pragma HLS INTERFACE s_axilite port=d_input  bundle=control
    #pragma HLS INTERFACE s_axilite port=d_model  bundle=control
    #pragma HLS INTERFACE s_axilite port=d_ffn    bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // 使用局部二维数组作为中间数据存储，每层计算完成后作为下一层的输入
    float current[S][D_MODEL];
    float next[S][D_MODEL];

    // 将外部输入复制到current
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            current[i][j] = input[i * D_MODEL + j];
        }
    }

    // 对每一层依次计算
    for (int l = 0; l < N_LAYERS; l++) {
        // 根据层数计算各权重的偏移指针
        float* local_M_Q = M_Q + l * (D_INPUT * D_MODEL);
        float* local_M_K = M_K + l * (D_INPUT * D_MODEL);
        float* local_M_V = M_V + l * (D_INPUT * D_MODEL);
        float* local_W1  = W1  + l * (D_MODEL * D_FFN);
        float* local_W2  = W2  + l * (D_FFN * D_MODEL);

        float Q[S][D_MODEL];
        float K[S][D_MODEL];
        float V[S][D_MODEL];
        float Score[S][S];

        // 计算Q矩阵
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                Q[i][j] = 0.0f;
                for (int k = 0; k < D_MODEL; k++) {
                    Q[i][j] += current[i][k] * local_M_Q[k * D_MODEL + j];
                }
            }
        }
        // 计算K矩阵
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                K[i][j] = 0.0f;
                for (int k = 0; k < D_MODEL; k++) {
                    K[i][j] += current[i][k] * local_M_K[k * D_MODEL + j];
                }
            }
        }
        // 计算V矩阵
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                V[i][j] = 0.0f;
                for (int k = 0; k < D_MODEL; k++) {
                    V[i][j] += current[i][k] * local_M_V[k * D_MODEL + j];
                }
            }
        }

        float scale = 1.0f / hls::sqrt(static_cast<float>(D_MODEL));
        // 计算注意力得分Score
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
        float temp1[S][D_MODEL];
        // 计算Attention输出
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                temp1[i][j] = 0.0f;
                for (int k = 0; k < S; k++) {
                    temp1[i][j] += Score[i][k] * V[k][j];
                }
            }
        }
        // 残差连接
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                temp1[i][j] = current[i][j] + temp1[i][j];
            }
        }
        float temp2[S][D_MODEL];
        layer_norm(&temp1[0][0], &temp2[0][0], S * D_MODEL);

        // 前馈神经网络部分
        float ffn_intermediate[S][D_FFN];
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_FFN; j++) {
                ffn_intermediate[i][j] = 0.0f;
                for (int k = 0; k < D_MODEL; k++) {
                    ffn_intermediate[i][j] += temp2[i][k] * local_W1[k * D_FFN + j];
                }
            }
        }
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_FFN; j++) {
                ffn_intermediate[i][j] = hls::max(0.0f, ffn_intermediate[i][j]);
            }
        }
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                next[i][j] = 0.0f;
                for (int k = 0; k < D_FFN; k++) {
                    next[i][j] += ffn_intermediate[i][k] * local_W2[k * D_MODEL + j];
                }
            }
        }
        // 残差连接
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                next[i][j] = temp2[i][j] + next[i][j];
            }
        }
        layer_norm(&next[0][0], &next[0][0], S * D_MODEL);

        // 为下一层准备，将当前层的输出复制到current
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < D_MODEL; j++) {
                current[i][j] = next[i][j];
            }
        }
    }
    // 最终将最后一层的输出写回外部内存
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            output[i * D_MODEL + j] = current[i][j];
        }
    }
}
}
