#include "pyplot.h"


pyplot::pyplot(string name) {
    ax = plt::figure();
    if (!name.empty()) this->name = name;
}

pyplot::~pyplot() {

}

void pyplot::plot_vector(float *in, size_t length) {
    vector<float> buffer(in, in + length);
    plt::figure(ax);
    plt::cla();
    if (!name.empty()) plt::title(name);
    plt::plot(buffer);
    plt::pause(d_pause);
}

void pyplot::plot_vector(complex<float> *in, size_t length) {
    vector<float> buffer_r(length, 0);
    vector<float> buffer_i(length, 0);
    for (int i = 0; i < length; ++i) {
        buffer_r[i] = real(in[i]);
        buffer_i[i] = imag(in[i]);
    }
    plt::figure(ax);
    plt::cla();
    if (!name.empty()) plt::title(name);
    plt::plot(buffer_r);
    plt::plot(buffer_i);
    plt::pause(d_pause);
}

void pyplot::plot_vector(vector<float> &in, size_t length) {
    vector<float> buffer;
    if (length) {
        buffer = vector<float>(&in[0], &in[length]);
    } else {
        buffer = in;
    }
    plt::figure(ax);
    plt::cla();
    if (!name.empty()) plt::title(name);
    plt::ylim(-0.05, 0.05);
    plt::plot(buffer);
    plt::pause(d_pause);
}

void pyplot::plot_vector(vector<complex<float>> &in, size_t length) {
    vector<float> buffer_r;
    vector<float> buffer_i;
    if (length) {
        buffer_r.resize(length);
        buffer_i.resize(length);
        for (int i = 0; i < length; ++i) {
            buffer_r[i] = real(in[i]);
            buffer_i[i] = imag(in[i]);
        }
    } else {
        buffer_r.resize(in.size());
        buffer_i.resize(in.size());
        for (int i = 0; i < in.size(); ++i) {
            buffer_r[i] = real(in[i]);
            buffer_i[i] = imag(in[i]);
        }
    }
    plt::figure(ax);
    plt::cla();
    if (!name.empty()) plt::title(name);
    plt::plot(buffer_r);
    plt::plot(buffer_i);
    plt::pause(d_pause);
}

void pyplot::plot_matrix(float *in, size_t rows, size_t columns) {
    plt::figure(ax);
    plt::cla();
    if (!name.empty()) plt::title(name);
    plt::imshow(in, rows, columns, 1);
    plt::pause(d_pause);
}

void pyplot::plot_dots(vector<complex<float>> &in) {
    // TODO: Надо вариант с заданными границами
    vector<float> x;
    vector<float> y;
    for (auto &s : in) {
        x.push_back(real(s));
        y.push_back(imag(s));
    }
    plot_dots(x, y);
}

void pyplot::plot_dots(vector<float> &x, vector<float> &y) {
    // TODO: Надо вариант с заданными границами
    plt::figure(ax);
    plt::cla();
    plt::plot(x, y, "o");
    plt::grid(true);
    plt::pause(d_pause);
}

void pyplot::vector_animate() {

}

void pyplot::matrix_animate() {

}

void pyplot::start_animate() {

}

void pyplot::wait(float sec) {
    plt::pause(sec);
}


/// =================== DISPLAY PLOT =====================

display_plot::display_plot(size_t num_rows, size_t num_columns) :
        num_r(num_rows), num_c(num_columns) {

}

void
display_plot::plot_blured(float *in) {
    plt::subplot(2, 3, 1);
    plt::cla();
    plt::imshow(in, num_r, num_c, 1);

    plt::title("Blured image");
    plt::pause(0.01);
}

void
display_plot::plot_binary(float *in) {
    plt::subplot(2, 3, 2);
    plt::cla();
    plt::imshow(in, num_r, num_c, 1);

    plt::title("Binary image");
    plt::pause(0.01);
}

void
display_plot::plot_zones(float *in) {
    plt::subplot(2, 3, 3);
    plt::cla();
    plt::imshow(in, num_r, num_c, 1);

    plt::title("Signals mask");
    plt::pause(0.01);
}

void
display_plot::plot_signals(vector<complex<float>> signals) {
    vector<float> x, y;
    for (auto &s : signals) {
        x.push_back(real(s));
        y.push_back(imag(s));
    }

    plt::subplot(2, 3, 5);
    plt::cla();
    plt::plot(x, y, "x");

    plt::title("Signals");
    plt::grid(true);
    plt::xlim(0, xlim);
    plt::ylim(0, ylim);

    plt::pause(0.01);
}

void
display_plot::plot_centroids(vector<complex<float>> centroids) {
    vector<float> x, y;
    for (auto &s : centroids) {
        x.push_back(real(s));
        y.push_back(imag(s));
    }

    plt::subplot(2, 3, 6);
    plt::cla();
    plt::plot(x, y, "x");

    plt::title("Centroids");
    plt::grid(true);
    plt::xlim(0, xlim);
    plt::ylim(0, ylim);

    plt::pause(0.01);
}
