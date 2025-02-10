#include <iostream>
using namespace std;

// W1 represents gradient
// W2 represents y-intercept

int* GenerateRandomXCoordinates(int arrayLen){
    int* Xarray = new int[arrayLen];

    for(int i = 0; i < arrayLen; i++ ){
        Xarray[i] = rand() % 11;
    }

    return Xarray;
}
int* GenerateRandomYCoordinates(int arrayLen){
    int* Yarray = new int[arrayLen];

    for(int i = 0; i < arrayLen; i++ ){
        Yarray[i] = rand() % 11;
    }

    return Yarray;
}

double GenerateW1(){
    return rand() % 20 / 100.0;
}

double GenerateW2(){
    return rand() % 20 / 100.0;
}

double LinearFunction(double W1, double W2, int X){
    return W1 * X + W2;
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

double DifferentialW1(double W1, double W2, int* XCoordinates, int* YCoordinates, int arrayLen, double penaltyStrength, double actualGradient, double actualIntercept){
    int meanSquaredDifference = 0;
    for(int i = 0; i < arrayLen; i++){
         meanSquaredDifference +=  (LinearFunction(W1, W2, XCoordinates[i]) - LinearFunction(actualGradient, actualIntercept, XCoordinates[i])) * XCoordinates[i];
    }
    meanSquaredDifference /= arrayLen;
    return meanSquaredDifference*2 + (penaltyStrength * sign(W1));
}

double DifferentialW2(double W1, double W2, int* XCoordinates, int* YCoordinates, int arrayLen, double penaltyStrength, double actualGradient, double actualIntercept){
    int meanSquaredDifference = 0;
    for(int i = 0; i < arrayLen; i++){
         meanSquaredDifference +=  (LinearFunction(W1, W2, XCoordinates[i]) - LinearFunction(actualGradient, actualIntercept, XCoordinates[i]));
    }
    meanSquaredDifference /= arrayLen;
    return meanSquaredDifference*2;
}

double GetNewW1(double W1, double W2, int* XCoordinates, int* YCoordinates, int arrayLen, double learningRate, double penaltyStrength, double actualGradient, double actualIntercept){
    return W1 - learningRate * DifferentialW1(W1, W2, XCoordinates, YCoordinates, arrayLen, penaltyStrength, actualGradient, actualIntercept);
}

double GetNewW2(double W1, double W2, int* XCoordinates, int* YCoordinates, int arrayLen, double learningRate, double penaltyStrength, double actualGradient, double actualIntercept){
    return W2 - learningRate * DifferentialW2(W1, W2, XCoordinates, YCoordinates, arrayLen, penaltyStrength, actualGradient, actualIntercept);
}

double MeanAbsoluteError(double W1, double W2, double actualGradient, double actualIntercept, int arrayLen){
    double error = 0;
    int* TestXCoordinates = GenerateRandomXCoordinates(arrayLen);
    for(int i = 0; i < arrayLen; i++){
        error += abs(LinearFunction(W1, W2, TestXCoordinates[i]) - LinearFunction(actualGradient, actualIntercept, TestXCoordinates[i]));
    }
    return error / arrayLen;

}

int main(){
    srand(time(0));
    double PENALTY_STRENGTH = 18;
    double GRADIENT_W1 = 45;
    double INTERCEPT_W2 = 2;
    int EPOCH = 1000;
    int* XCoordinates = GenerateRandomXCoordinates(100);
    int* YCoordinates = GenerateRandomYCoordinates(100);
    double W1 = GenerateW1();
    double W2 = GenerateW2();  
    cout << "Initial W1: " << GRADIENT_W1 << endl;
    cout << "Initial W2: " << INTERCEPT_W2 << endl;
    for(int i = 0; i < EPOCH; i++){
        W1 = GetNewW1(W1, W2, XCoordinates, YCoordinates, 100, 0.01, PENALTY_STRENGTH, GRADIENT_W1, INTERCEPT_W2);
        W2 = GetNewW2(W1, W2, XCoordinates, YCoordinates, 100, 0.01, PENALTY_STRENGTH, GRADIENT_W1, INTERCEPT_W2);
    }
    cout << "Predicted W1: " << W1 << endl;
    cout << "Predicted W2: " << W2 << endl;
    cout << "Mean Absolute Error (MAE) of model: " << MeanAbsoluteError(W1, W2, GRADIENT_W1, INTERCEPT_W2, 100) << endl;
}