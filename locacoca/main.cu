#include "matplotlibcpp.h"
#include "functors.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/complex.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <iostream>


using namespace std;
namespace plt = matplotlibcpp;

struct complex_sinusoid {
    double freq; // frequency
    double phase_inc; // phase increment
    double samp_rate;

    __host__ __device__

    cf operator()(const int &i) const {
        double phase = i * phase_inc; // current phase
        return thrust::polar(1.0, freq * 2.0 * M_PI * i / samp_rate + phase);
    }
};

struct complex_multiply {
    __host__ __device__

    cf operator()(const cf &x, const cf &y) const {
        return x * y;
    }
};

void plot(vector<complex<float>> const &ref) {
    vector<float> i(ref.size());
    vector<float> r(ref.size());
    for (int ii = 0; ii < ref.size(); ++ii) {
        cf a = ref[ii];
        i[ii] = a.imag();
        r[ii] = a.real();
    }
    plt::plot(i);
    plt::plot(r);
    plt::pause(0.01);
//    plt::show();
};

void plot(vcf const &ref) {
    vector<complex<float>> h_v(ref.size());
    tr::copy(ref.begin(), ref.end(), h_v.begin());
    plot(h_v);
}

int main() {
    vector<float> a(1000);

//    std::generate(std::execution::par_unseq, inputData.begin(), inputData.end(), []()-> Complex {
//        thread_local std::default_random_engine generator; // thread_local so we don't have to do any locking
//        thread_local std::normal_distribution<double> distribution(0.0, 0.5); // mean = 0.0, stddev = 0.5
//        return Complex(distribution(generator), distribution(generator));
//    });


    //    auto source = make_shared<File_source>("/mnt/raid/REC/900/TBS-880-940_25msps_gain_08_step_5e6.cf32");
    std::cout << "alive" << std::endl;
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    double tx_freq = 10e9;
    double samp_rate = 25e6;
    double mov_freq_low = 10; // m/s
    double mov_freq_high = 200 * 1000 / 3600; // m/s

    double dop_freq_low = 2 * mov_freq_low * tx_freq / 3e8; // Hz
    double dop_freq_high = 2 * mov_freq_high * tx_freq / 3e8; // Hz
    double dop_freq_range = dop_freq_high * 2;
    double phase_inc = 0;

    int N = 2048;
    int Nk = 256;
    int Nr = 128;
    int ROW = N * Nk;
    int SIZE = ROW * Nr;
    double dop_freq_step = dop_freq_range / (float) Nr;

    auto fft_fun = cuda::fft(N);
    auto ifft_fun = cuda::ifft(N);

    auto lfm_iter = tr::make_transform_iterator(tr::make_counting_iterator(0), ff::gen_lfm(N));

    // Create a vector of complex sinusoid iterators
    std::vector<thrust::transform_iterator<complex_sinusoid, thrust::counting_iterator<int>>> sinusoid_iters;
    for (int r = 0; r < Nr; r++) {
        double freq = dop_freq_low + r * dop_freq_step; // frequency for current sinusoid
        complex_sinusoid sinusoid_func = {freq, phase_inc, samp_rate};
        thrust::counting_iterator<int> iter(0);
        thrust::transform_iterator<complex_sinusoid, thrust::counting_iterator<int>> sinusoid_iter(iter, sinusoid_func);
        sinusoid_iters.push_back(sinusoid_iter);
    }

    // делаем один импульс лчм
    vcf chirp_fft(lfm_iter, lfm_iter + N);
    // считаем от него спектр
    fft_fun(chirp_fft, chirp_fft);
    // делаем conj над спектром
    tr::transform(chirp_fft.begin(), chirp_fft.end(), chirp_fft.begin(), ff::fn_conj());
    // фильтр готов

    // принятый блок
    vcf packet(lfm_iter, lfm_iter + ROW);



    // принятый блок перемноженный на расстройки
    vcf packet_sinus(ROW * Nr, cf(0));

    // TODO: немного пошуметь и позамирать

    // вектор накопитель для всех расстроек
    vcf acc(N * Nr, cf(0));

    // результат свертки для всех расстроек
    vcf conv(N * Nr, cf(0));


    while (true) {
        start = std::chrono::steady_clock::now();

        // перемножаем с расстройками
        for (int r = 0; r < Nr; r++) {
            thrust::transform(packet.begin(),
                              packet.end(),
                              sinusoid_iters[r],
                              packet_sinus.begin() + ROW * r,
                              complex_multiply());
        }

        // суммируем в каждой расстройке
        for (int r = 0; r < Nr; r++) {
            // блоки длиной N в acc
            for (int block = 0; block < Nk; block++) {
                tr::transform(packet_sinus.begin() + block * N * r,
                              packet_sinus.begin() + (block + 1) * N * r,
                              acc.begin() + N * r,
                              acc.begin() + N * r,
                              tr::plus<cf>());
            }
        }
        plot(acc);
        /// сворачиваем с оригиналом
        // сначала в conv кладем спектр от просуммированного сигнала
        for (int r = 0; r < Nr; ++r) {
            fft_fun(tr::raw_pointer_cast(conv.data()) + N * r,
                    tr::raw_pointer_cast(acc.data()) + N * r);
        }

        // перемножаем спектр с фильтром
        tr::transform(conv.begin(), conv.end(), chirp_fft.begin(), conv.begin(), tr::multiplies<cf>());

        // обратное преобразование
        for (int r = 0; r < Nr; ++r) {
            ifft_fun(tr::raw_pointer_cast(conv.data()) + N * r);
        }

        vf inreal(N);
        tr::transform(conv.begin(), conv.end(), inreal.begin(), ff::fn_abs());

        end = std::chrono::steady_clock::now();
        std::cout << "T: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " µs" << std::endl;
    }
}