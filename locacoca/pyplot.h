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
    pyplot();

    ~pyplot();

    void plot_vector(float *in, size_t length);

    void plot_vector(vector<float> &in, size_t length = 0);

    void plot_vector(complex<float> *in, size_t length);

    void plot_vector(vector<complex<float>> &in, size_t length = 0);

    void plot_matrix();

    void vector_animate();

    void matrix_animate();

    void start_animate();

private:
    float d_pause = 0.1;
    string name;
};


#endif //DETECTOR_PYPLOT_H
