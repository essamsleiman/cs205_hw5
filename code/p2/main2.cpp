#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
using namespace std;

vector<vector<double>>* matrices[3];
vector<double>* bias;

// Store your resultant predictions here
vector<vector<double>>* resMatrix;


vector<double> parseLine(const string& line) {
    vector<double> row;
    stringstream ss(line);
    string value;

    while (getline(ss, value, ';')) {
        row.push_back(stod(value));
    }

    return row;
}

vector<vector<double>>* loadMatrixFromFile(const string& filename) {
    ifstream file(filename);
    string line;
    auto* matrix = new vector<vector<double>>;

    if (file.is_open()) {
        while (getline(file, line)) {
            matrix->push_back(parseLine(line));
        }
        file.close();
    } else {
        cerr << "Unable to open file." << endl;
    }

    return matrix;
}

vector<double>* loadVectorFromFile(const string& filename){
    ifstream file(filename);
    string line;
    auto* vec = new vector<double>;

    while (getline(file, line)) {
        vec->push_back(stod(line));
    }

    return vec;
}

void parser(){
    const string f1 = "features.dat";
    const string f2 = "kernel.dat";
    const string f3 = "class_predictions_ground_truth.dat";
    const string f4 = "bias.dat";

    vector<vector<double>>* m1 = loadMatrixFromFile(f1);
    vector<vector<double>>* m2 = loadMatrixFromFile(f2);
    vector<vector<double>>* m3 = loadMatrixFromFile(f3);
    vector<double>* b = loadVectorFromFile(f4);

    cout << "Matrix 1 Size: " << m1->size() << endl;
    cout << "Matrix 2 Size: " << m2->size() << endl;
    cout << "Matrix 3 Size: " << m3->size() << endl;
    cout << "Bias Size: " << b->size() << endl;

    matrices[0] = m1;
    matrices[1] = m2;
    matrices[2] = m3;
    bias = b;
}

bool isApproximatelyEqual(double a, double b){
	return abs(a - b) < 1e-7;
}

bool checkResult(){
    // compares ResMatrix to the Ground Truth

    assert(resMatrix && matrices[2] && "One of the matrices is a nullptr");

    assert(resMatrix->size() == matrices[2]->size() && "The Matrix Size Does Not Match");

    for (size_t i = 0; i < matrices[2]->size(); ++i) {
        const std::vector<double>& vec1 = (*matrices[2])[i];
        const std::vector<double>& vec2 = (*resMatrix)[i];

        if (vec1.size() != vec2.size() || !std::equal(vec1.begin(), vec1.end(), vec2.begin(), isApproximatelyEqual)) {
            // Check each vector size and compare element-wise
            return false;
        }
    }

    return true;

}


__global__ void compute_logits(double *input_features, double *weights, double *biases, double *logits, int num_samples, int feature_dim, int num_classes)
{
    const uint sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint class_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx < num_samples && class_idx < num_classes)
    {
        double linear_combination = biases[class_idx];
        int weight_offset = class_idx;
        int feature_offset = sample_idx * feature_dim;

        for (int feature_idx = 0; feature_idx < feature_dim; ++feature_idx)
        {
           linear_combination += input_features[feature_offset + feature_idx] * weights[feature_idx * num_classes + weight_offset];
        }
        
        logits[sample_idx * num_classes + class_idx] = linear_combination;
    }
}


void copyDataToGPU(const std::vector<std::vector<double>>& inputFeature,
                   double* gpuFeatures,
                   const std::vector<std::vector<double>>& inputKernel,
                   double* gpuKernel,
                   const std::vector<double>* inputBias,
                   double* gpuBias)
{
    // bias vector to GPU
    int biasSize = inputBias->size();
    for (int i = 0; i < biasSize; i++)
    {
        gpuBias[i] = (*inputBias)[i];
    }

    // kernel matrix to GPU
    int kernelRowCount = inputKernel.size();
    int kernelColCount = inputKernel[0].size();
    for (int i = 0; i < kernelRowCount; i++)
    {
        for (int k = 0; k < kernelColCount; k++)
        {
            gpuKernel[i * kernelColCount + k] = inputKernel[i][k];
        }
    }

    // feature matrix to GPU
    int featureRowCount = inputFeature.size();
    int featureColCount = inputFeature[0].size();
    for (int i = 0; i < featureRowCount; i++)
    {
        for (int k = 0; k < featureColCount; k++)
        {
            gpuFeatures[i * featureColCount + k] = inputFeature[i][k];
        }
    }
}

std::vector<std::vector<double>> CopyDataToDevice(const double* d_result, int feature_size, int bias_size) {
    
    // Allocate vector 
    std::vector<std::vector<double>> result(feature_size, std::vector<double>(bias_size));

    // sychronize
    cudaDeviceSynchronize();

    // copy data to allocated vector
    for (int i = 0; i < feature_size; i++) {
        for (int j = 0; j < bias_size; j++) {
            result[i][j] = d_result[i * bias_size + j];
        }
    }

    return result;
}

int main()
{
    parser();

    // CUDA naive implementation
    auto feature = *matrices[0];
    auto kernel = *matrices[1];

    // allocate the memory on gpu using cudaMallocManaged for features, kernel, bias, and result
    double *d_bias, *d_features, *d_kernel,  *d_result;
    cudaMallocManaged(&d_bias, bias->size() * sizeof(double));
    cudaMallocManaged(&d_kernel, kernel.size() * kernel[0].size() * sizeof(double));
    cudaMallocManaged(&d_result, feature.size() * bias->size() * sizeof(double));
    cudaMallocManaged(&d_features, feature.size() * feature[0].size() * sizeof(double));

    // then copy data to gpu
    copyDataToGPU(feature, d_features, kernel, d_kernel, bias, d_bias);

    // define dim
    dim3 blockDim(32, 32, 1);
    dim3 gridDim((feature.size() + blockDim.x - 1) / blockDim.x, (bias->size() + blockDim.y - 1) / blockDim.y, 1);

    for (int iter = 0; iter < 10; iter++)
    {
        auto start = chrono::steady_clock::now();

        // cuda kernel
        compute_logits<<<gridDim, blockDim>>>(d_features, d_kernel, d_bias, d_result, feature.size(), kernel.size(), bias->size());

        // wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        auto end = chrono::steady_clock::now();

        // print the time taken for each iteration
        cout << "Iteration " << iter << " Time taken: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

        // copy results back to host only after final iteration to check results
        if (iter == 9)
        {
            std::vector<std::vector<double>> res = CopyDataToDevice(d_result, feature.size(), bias->size());
            resMatrix = &res;

            if (checkResult())
            {
                cout << "TEST SUCCESSFUL." << endl;
            }
            else
            {
                cout << "The Matrices don't match." << endl;
            }
        }
    }

    // don't forget to free allocated memory
    cudaFree(d_features);
    cudaFree(d_kernel);
    cudaFree(d_bias);
    cudaFree(d_result);

    return 0;
}
