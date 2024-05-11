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

int main()
{

    parser();

    // TODO: Your code here. Use matrices[0] to access features.dat and matrices[1] to access kernel.dat

    auto* feature = matrices[0];
    auto* kernel = matrices[1];

    for (int iter = 0; iter < 10; iter ++){
        vector<vector<double>> res(feature->size(), vector<double>(bias->size()));

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < feature->size(); ++i)
        {
            for (size_t k = 0; k < kernel->size(); ++k)
            {
                for (size_t j = 0; j < bias->size(); ++j)
                {
                    
                    // Calculate the product of feature and kernel
                    double feature_kernel_product = (*feature)[i][k] * (*kernel)[k][j];

                    // Check if this is the first element in kernel and add bias if true
                    double bias_addition = 0.0;
                    if (k == 0) {
                        bias_addition = (*bias)[j];
                    }

                    // Update the result matrix element
                    res[i][j] += feature_kernel_product + bias_addition;
                }
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::printf("Time taken is: %ld ms \n", std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());

        resMatrix = &res;


        if (iter == 0)
        {
            // You can modify the function declarations to pass resMatrix as a parameter if you would like
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

    return 0;
}
