#include <iostream>
using namespace std;

vector<double> GenerateRandomW(int arrayLen){
    vector<double> array;
    for(int i = 0; i < arrayLen; i++){
        array.push_back(rand() % 20 / 1.0);
    }
    return array;
}

double GenerateRandomB(){
    return rand() % 20 / 1.0;
}

int sign(double x){
    if(x > 0){
        return 1;
    }else if(x < 0){
        return -1;
    }else{
        return 0;
    }
}

double MultiDFunctionNotAccurate(vector<double>& W, vector<vector<double> >& X, double B, int arrayLen, int dataRow){
    double sum = 0;
    for(int i = 0; i < arrayLen; i++){
        sum += W[i] * X[i][dataRow];
    }
    return sum + B + ((rand() % 50 / 1.0) - 25.0 );
}

double MultiDFunction(vector<double>& W, vector<vector<double> >& X, double B, int arrayLen, int dataRow){
    double sum = 0;
    for(int i = 0; i < arrayLen; i++){
        sum += W[i] * X[i][dataRow];
    }
    return sum + B;
}

vector<double> TrainModel(int arrayLen, int dataRows, double learningRate, vector<vector<double> >& Data, vector<double>& w, double b, double penaltyStrength, int epoch){
    vector<double> weights = GenerateRandomW(arrayLen);
    double bias = GenerateRandomB();

    double errorSum = 0.0;
    for(int k = 0; k < epoch; k++){
        for(int j = 0; j < arrayLen; j++){
            errorSum = 0;
            for(int i = 0; i < dataRows; i++){
                // Use an appropriate function to fetch Y for training
                errorSum +=  (MultiDFunction(weights, Data, bias, arrayLen, i) - MultiDFunctionNotAccurate(w, Data, b, arrayLen, i)) * Data[j][i];
            }
            errorSum = errorSum / dataRows;
            errorSum = errorSum * 2;
            weights[j] = weights[j] - learningRate * (errorSum + (penaltyStrength * sign(weights[j])));
        }
        errorSum = 0;
        for(int i = 0; i < dataRows; i++){
            // Use an appropriate function to fetch Y for training
            errorSum +=  (MultiDFunction(weights, Data, bias, arrayLen, i) - MultiDFunctionNotAccurate(w, Data, b, arrayLen, i));
        }
        errorSum = errorSum / dataRows;
        errorSum = errorSum * 2;
        bias = bias - (learningRate * errorSum);
    }
    cout << "\nPredicted Weights : " << endl;
    for(int i = 0; i < arrayLen; i++){
        cout << (weights[i]) << endl;
    }
    cout << "\nPredicted Bias : " << endl;
    cout << bias << endl;
    weights.push_back(bias);
    return weights;
}

double GetAbsoluteMeanError(int arrayLen, int dataRows, double learningRate, vector<vector<double> >& Data, vector<double>& w, double b, double penaltyStrength, int epoch){
    vector<double> weights = TrainModel(6,6, 0.0001, Data, w, b, 0.01, 1000000);
    double bias = weights[weights.size()];
    weights.pop_back();
    double sum = 0;
    for(int i = 0; i < dataRows; i++){
        sum += abs(MultiDFunction(weights, Data, bias, arrayLen, i) - MultiDFunctionNotAccurate(w, Data, b, arrayLen, i));
    }
    return sum/dataRows;
}



int main(){

    // Below was used for testing during development 
    srand(5);
    vector<double> w= {18,3,4,5,17,44};
    double b = 100;
    cout << "\nActual Weights : " << endl;
    for(int i = 0; i < w.size(); i++){
        cout << (w[i]) << endl;
    }
    cout << "\nActual Bias : " << endl;
    cout << b << endl;
    vector<vector<double> > data = {
        {11, 2, 6, 4, 6, 17},
        {1, 21, 3, 45,  6, 5},
        {41, 2, 30, 43, 6, 7},
        {1, 41, 3, 46, 6, 2},
        {41, 24, 90, 43, 7, 6},
        {1, 21, 3, 45,  6, 5}
    };
    cout << "\nMean Absolute Error : " << endl;
    cout << GetAbsoluteMeanError(6, 6, 0.0001, data, w, b, 0.01, 1000000) << endl;
}