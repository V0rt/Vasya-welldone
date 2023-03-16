#ifndef DETECTOR_PYPLOT_H
#define DETECTOR_PYPLOT_H

#include <complex>
#include <vector>
#include "matplotlibcpp.h"
#include <thread>

using namespace std;
namespace plt = matplotlibcpp;

class pyplot {

public:
    pyplot(string name = "");

    ~pyplot();

    void plot_vector(float *in, size_t length);

    void plot_vector(complex<float> *in, size_t length);

    void plot_vector(vector<float> &in, size_t length = 0);

    void plot_vector(vector<complex<float>> &in, size_t length = 0);

    void plot_matrix(float *in, size_t rows, size_t columns);

    void vector_animate();

    void matrix_animate();

    void start_animate();

    void plot_dots(vector<complex<float>> in);

    void plot_dots(vector<float> x, vector<float> y);

    void show() { plt::show(); }

    void pause(float sec) { plt::pause(sec); }

private:
    float d_pause = 0.1;
    string name = "";
    long ax;
};


#endif //DETECTOR_PYPLOT_H
