#include "functors.h"
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/generate.h>
#include <thrust/complex.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <iostream>


int main() {
//    auto source = make_shared<File_source>("/mnt/raid/REC/900/TBS-880-940_25msps_gain_08_step_5e6.cf32");

    int N = 1024;
    int Nk = 64;
    int Nr = 64;
    int ROW = N * Nk;
    int SIZE = ROW * Nr;
    int num_rows = 1024;
    int num_columns = 1024;
    int buffer_size = num_rows * num_columns;
//    source->get_data_block(buffer_size);

    auto lfm_iter = tr::make_transform_iterator(tr::make_counting_iterator(0), ff::gen_lfm(N));

    std::cout << "alive" << std::endl;

    vcf packet(lfm_iter, lfm_iter + ROW);
    vcf chirp_fft(lfm_iter, lfm_iter + 200);
    tr::copy(lfm_iter, lfm_iter + N, chirp_fft.begin());

    vcf acc(N, cf(0));
    vcf conv(N, cf(0));

    auto fft_fun = cuda::fft(N);
    auto ifft_fun = cuda::ifft(N);

    fft_fun(chirp_fft, chirp_fft);
    tr::transform(chirp_fft.begin(), chirp_fft.end(), chirp_fft.begin(), ff::fn_conj());
//    tr::fill(packet.begin(), packet.end(), cf(1, 1));

//    суммируем по строкам
    for (int part = 0; part < Nk; part++) {
        tr::transform(packet.begin() + part * N,
                      packet.begin() + (part + 1) * N,
                      acc.data(),
                      acc.data(),
                      tr::plus<cf>());
    }

    // сворачиваем с оригиналом
    // сначала в conv кладем спектр от сигнала
    fft_fun(conv, acc);
    // перемножение с оригиналом
    tr::transform(conv.begin(), conv.end(), chirp_fft.begin(), conv.begin(), tr::multiplies<cf>());

    ifft_fun(conv);

    vf inreal(N);
    tr::transform(conv.begin(), conv.end(), inreal.begin(), ff::fn_abs());

    auto max = tr::max_element(inreal.begin(), inreal.end());
    auto pos = tr::distance(inreal.begin(), max);

    std::cout << "max_pos " << pos << std::endl;
}