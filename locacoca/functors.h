#ifndef NEEDLE_FUNCTORS_H
#define NEEDLE_FUNCTORS_H

#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/async/reduce.h>

namespace tr = thrust;
typedef thrust::complex<float> cf;
typedef thrust::device_vector<thrust::complex<float>> vcf;
typedef thrust::device_vector<float> vf;


namespace ff {
/***************************/
/* MATRIX TRANSPOSE */
/***************************/
#define BLOCK_DIM 16

    __global__ void
    transpose(float *odata, float *idata, int width, int height) {
        __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

        // read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
        unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
        unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
        if ((xIndex < width) && (yIndex < height)) {
            unsigned int index_in = yIndex * width + xIndex;
            block[threadIdx.y][threadIdx.x] = idata[index_in];
        }

        // synchronise to ensure all writes to block[][] have completed
        __syncthreads();

        // write the transposed matrix tile to global memory (odata) in linear order
        xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
        yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if ((xIndex < height) && (yIndex < width)) {
            unsigned int index_out = yIndex * height + xIndex;
            odata[index_out] = block[threadIdx.x][threadIdx.y];
        }
    }


/***************************/
/* GPU KERNEL WITH CACHING */
/***************************/
    __global__ void
    d_convolution_1D_caching(const float *__restrict__ d_Signal,
                             const float *__restrict__ d_ConvKernel,
                             float *__restrict__ d_Result_GPU,
                             const int N, const int K) {
#define BLOCKSIZE 8
///        d_convolution_1D_caching<<<1 + (N / BLOCKSIZE), BLOCKSIZE>>>(d_Signal, d_ConvKernel, d_Result_GPU, N, K);

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        __shared__ float d_Tile[BLOCKSIZE];

        d_Tile[threadIdx.x] = d_Signal[i];
        __syncthreads();

        float temp = 0.f;

        int N_start_point = i - (K / 2);

        for (int j = 0; j < K; j++)
            if (N_start_point + j >= 0 && N_start_point + j < N) {

                if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))

                    // --- The signal element is in the tile loaded in the shared memory
                    temp += d_Tile[threadIdx.x + j - (K / 2)] * d_ConvKernel[j];

                else

                    // --- The signal element is not in the tile loaded in the shared memory
                    temp += d_Signal[N_start_point + j] * d_ConvKernel[j];

            }

        d_Result_GPU[i] = temp;
    }


/*********************************/
/* MATRIX REDUCTION BY ROWS */
/*********************************/
    template<typename T>
    struct fn_matrix_reduce {
        thrust::device_ptr<T> ptr;
        int size;

        __host__ __device__

        fn_matrix_reduce(thrust::device_ptr<T> p, int s) : ptr(p), size(s) {}

        __host__ __device__

        float operator()(int row_index) {
            auto begin = ptr + size * row_index;
            return thrust::reduce(thrust::device, begin, begin + size, (T) 0);
        }
    };

/****************/
/* SIN OPERATOR */
/****************/
    class sin_op {

        float fk_, Fs_;

    public:

        sin_op(float fk, float Fs) {
            fk_ = fk;
            Fs_ = Fs;
        }

        __host__ __device__ float operator()(float x) const { return sin(2.f * M_PI * x * fk_ / Fs_); }
    };

/*****************/
/* SINC OPERATOR */
/*****************/
    class sinc_op {

        float fc_, Fs_;

    public:

        sinc_op(float fc, float Fs) {
            fc_ = fc;
            Fs_ = Fs;
        }

        __host__ __device__ float operator()(float x) const {
            if (x == 0) return (2.f * fc_ / Fs_);
            else return (2.f * fc_ / Fs_) * sin(2.f * M_PI * fc_ * x / Fs_) / (2.f * M_PI * fc_ * x / Fs_);
        }
    };

/********************/
/* HAMMING OPERATOR */
/********************/
    class hamming_op {

        int L_;

    public:

        hamming_op(int L) { L_ = L; }

        __host__ __device__ float operator()(int x) const {
            return 0.54 - 0.46 * cos(2.f * M_PI * x / (L_ - 1));
        }
    };

/*********************************/
/* MULTIPLY CUFFTCOMPLEX NUMBERS */
/*********************************/
    struct multiply_cufftComplex {
        __device__ cufftComplex operator()(const cufftComplex &a, const cufftComplex &b) const {
            cufftComplex r;
            r.x = a.x * b.x - a.y * b.y;
            r.y = a.x * b.y + a.y * b.x;
            return r;
        }
    };

/*********************************/
/* INTEGER MODULUS */
/*********************************/
    struct mod : tr::unary_function<int, int> {
        int N;

        __host__ __device__
        mod(int n) : N(n) {};

        __host__ __device__
        int operator()(int i) { return i % N; }
    };

/*********************************/
/* INTEGER DIVISION */
/*********************************/
    struct div : tr::unary_function<int, int> {
        int N;

        __host__ __device__
        div(int n) : N(n) {};

        __host__ __device__
        int operator()(int i) { return (int) i / N; }
    };

    struct fn_abs : tr::unary_function<cf, float> {
        __host__ __device__
        float operator()(const cf &val) { return tr::abs<float>(val); }
    };

    struct fn_conj : tr::unary_function<cf, cf> {
        __host__ __device__
        cf operator()(const cf &val) { return tr::conj(val); }
    };


/*********************************/
/* SINUS GENERATOR */
/*********************************/
    struct gen_sin : tr::unary_function<int, tr::complex<float>> {
        __host__ __device__

        gen_sin(float ph_inc) : inc(ph_inc) {}

        __host__ __device__

        tr::complex<float> operator()(int i) {
            return tr::exp(tr::complex<float>(0, 2.0 * M_PI * i * inc));
        }

    private:
        float inc = 0;
    };

/*********************************/
/* LFM GENERATOR */
/*********************************/
    struct gen_lfm : tr::unary_function<int, tr::complex<float>> {
        __host__ __device__

        gen_lfm(int length) : len(length) {}

        __host__ __device__

        tr::complex<float> operator()(int i) {
            return tr::exp(tr::complex<float>(0, M_PI * (i - len / 2) * (i - len / 2) / (float) len));
        }

    private:
        float len;
    };
}

namespace cuda {
    class fft : tr::unary_function<vcf, vcf> {
    private:
        int N;
        cufftHandle plan;
    public:
        __host__
        fft(size_t n, int batch = 1) : N(n) {
            cufftPlan1d(&plan, N, CUFFT_C2C, batch);
        }

        __host__
        void operator()(vcf &vec) {
            cufftExecC2C(plan,
                         (cufftComplex *) tr::raw_pointer_cast(vec.data()),
                         (cufftComplex *) tr::raw_pointer_cast(vec.data()),
                         CUFFT_FORWARD);
        }

        __host__
        void operator()(vcf &dst, const vcf &src) {
            cufftExecC2C(plan,
                         (cufftComplex *) tr::raw_pointer_cast(src.data()),
                         (cufftComplex *) tr::raw_pointer_cast(dst.data()),
                         CUFFT_FORWARD);
        }
    };

    class ifft : tr::unary_function<vcf, vcf> {
    private:
        int N;
        cufftHandle plan;
    public:
        __host__
        ifft(size_t n, int batch = 1) : N(n) {
            cufftPlan1d(&plan, N, CUFFT_C2C, batch);
        }

        __host__
        void operator()(vcf &vec) {
            cufftExecC2C(plan,
                         (cufftComplex *) tr::raw_pointer_cast(vec.data()),
                         (cufftComplex *) tr::raw_pointer_cast(vec.data()),
                         CUFFT_INVERSE);
        }

        __host__
        void operator()(vcf &dst, const vcf &src) {
            cufftExecC2C(plan,
                         (cufftComplex *) tr::raw_pointer_cast(src.data()),
                         (cufftComplex *) tr::raw_pointer_cast(dst.data()),
                         CUFFT_INVERSE);
        }
    };
}
#endif //NEEDLE_FUNCTORS_H
