#include <cassert>
#include <cstdlib>
#include <cinttypes>
#include <random>
#include <algorithm>
#include <thread>
#include <array>
#include <cstring>
#include <pthread.h>
#include <iostream>

#include "kestrel.h"
#include "q.h"

#define MATRIX_M (TILE_M * 8)
#define MATRIX_N (TILE_N * 8)
#define MATRIX_K (TILE_K * 8)

int8_t* a_quants;  // [ MATRIX_M * MATRIX_K ]
float* a_scales;   // [ MATRIX_M * MATRIX_K/64 ]
int8_t* b_quants;  // [ MATRIX_N * MATRIX_K ]
float* b_scales;   // [ MATRIX_N * MATRIX_K/64 ]

#pragma clang target("arch=kestrel")
void jacc_internal(float *t_acc, int32_t *t_c, uint64_t tile, uint64_t tk) {
    const uint64_t tiles_n = MATRIX_N / TILE_N;
    const uint64_t tile_n = tile % tiles_n;
    const uint64_t tile_m = tile / tiles_n;
	for (int i = 0; i < TILE_M; i++) {
		for (int j = 0; j < TILE_N; j++) {
		    t_acc[i * TILE_N + j] +=
			t_c[i * TILE_N + j]
			* a_scales[(tile_m * TILE_M + i) * MATRIX_K / 64 + tk / 64]
			* b_scales[(tile_n * TILE_N + j) * MATRIX_K / 64 + tk / 64];
		}}
}

void acc(auto t_acc, auto t_c, uint64_t tile, uint64_t tk) {
    jacc_internal(t_acc.ptr, t_c.ptr, tile, tk);
}

#pragma clang target("arch=kestrel")
uint64_t jacc(uint64_t subtask_id, uint64_t DID, void *data) {
    // decode data to (auto t_acc, auto t_c, uint64_t tile, uint64_t tk)
    uint64_t *arg = (uint64_t *)data;
    float *t_acc;
    int32_t *t_c;
    uint64_t tile;
    uint64_t tk;

    t_acc = (float *)arg[0];
    t_c = (int32_t *)arg[1];
    tile = arg[2];
    tk = arg[3];

    // std::cout << "acc " << t_acc << std::endl;

    // do acc reduction
    jacc_internal(t_acc, t_c, tile, tk);

    return 2;
}

/* single row only */
#pragma clang target("arch=kestrel")
void jsoftmax_internal(float *tile_c) {
    uint64_t j;
    double sum;
    float max_val = tile_c[0];
    for (j = 0; j < TILE_N; j++) {
        if (tile_c[j] > max_val) {
            max_val = tile_c[j];
        }
    }
    sum = 0;
    for (j = 0; j < TILE_N; j++) {
        tile_c[j] = std::exp(tile_c[j] - max_val);
        sum += tile_c[j];
    }
    for (j = 0; j < TILE_N; j++) {
        tile_c[j] /= static_cast<float>(sum);
    }
}

#pragma clang target("arch=kestrel")
uint64_t jsoftmax(uint64_t subtask_id, uint64_t DID, void *data) {
    jsoftmax_internal((float *)((uint64_t *)data)[0]);
    return 3;
}

#pragma clang target("arch=kestrel")
void jzero_internal(uint64_t *s, int n) {
    memset((void *)s, 0, n);
}

#pragma clang target("arch=kestrel")
uint64_t jzero(uint64_t subtask_id, uint64_t DID, void *data) {
    // decode data to (t_acc)
    uint64_t *arg = (uint64_t *)data;
    uint64_t *t_acc;
    uint64_t len;

    // std::cout << "memset " << t_acc << std::endl;

    t_acc = (uint64_t *)arg[0];
    len = (uint64_t)arg[1];
    jzero_internal(t_acc, len);
    return 1;
}

#pragma clang target("arch=kestrel")
uint64_t matmul_int8_kernel(
    uint64_t kernel_did,
    uint64_t thread_id,
    uint64_t thread_count,
    float* c
) {
    assert(MATRIX_M % TILE_M == 0);
    assert(MATRIX_N % TILE_N == 0);
    assert(MATRIX_K % TILE_K == 0);

    const uint64_t tiles_m = MATRIX_M / TILE_M;
    const uint64_t tiles_n = MATRIX_N / TILE_N;
    const uint64_t tiles = tiles_m * tiles_n;

    const uint64_t tile0 = thread_id * tiles / thread_count;
    const uint64_t tile1 = (thread_id + 1) * tiles / thread_count;

    auto t_a = tmem.alloc<int8_t, TILE_M, TILE_K>();
    auto t_b = tmem.alloc<int8_t, TILE_N, TILE_K>();
    auto t_c = tmem.alloc<int32_t, TILE_M, TILE_N>();
    auto t_acc = tmem.alloc<float, TILE_M, TILE_N>();

    int i1 = 0;
    DID_T last_did3 = 0;
    for (uint64_t tile = tile0; tile < tile1; ++tile, i1++) {
#ifdef DEBUG
        printf("====> thread_id=%lu tile=%lu, tile_end=%lu\n",thread_id, tile, tile1);
#endif
        const uint64_t tile_n = tile % tiles_n;
        const uint64_t tile_m = tile / tiles_n;

        float* tile_c = c + (tile_m * TILE_M * MATRIX_N) + tile_n * TILE_N;

        QK_CORE_NT_2(jzero, (uint64_t *)t_acc.ptr, TILE_M * TILE_N * sizeof(float),1,0,0,0);
        int last_did1 = 0, last_did2 = 0;
        for (uint64_t tk = 0; tk < MATRIX_K; tk += TILE_K) {
            printf("----> thread_id=%lu tk=%lu, k=%d, stride=%d\n",thread_id, tk,MATRIX_K,TILE_K);
            const int8_t* tile_a = a_quants + tile_m * TILE_M * MATRIX_K + tk;
            const int8_t* tile_b = b_quants + tile_n * TILE_N * MATRIX_K + tk;
	    QK_CORE_NT_2(jzero, (uint64_t *)t_c.ptr, TILE_M * TILE_N * sizeof(int32_t),2,last_did1,0,0);
            QK_LOAD_NT((uint64_t)t_a.ptr, (uint64_t)tile_a, MATRIX_K, TILE_K, TILE_M,3,last_did2,0);
            QK_LOAD_NT((uint64_t)t_b.ptr, (uint64_t)tile_b, MATRIX_K, TILE_K, TILE_N,4,last_did2,0);
            QK_AICE_MATMUL_NT((int8_t *)t_a.ptr, (int8_t *)t_b.ptr, (int32_t *)t_c.ptr,5,2,3,4);
            last_did2 = 5;
            QK_CORE_NT_4(jacc,(float *)t_acc.ptr,(int32_t *)t_c.ptr,tile,tk,6,5,1,0);
            last_did1 = 6;
        }
	QK_STORE_NT((uint64_t)t_acc.ptr, (uint64_t)tile_c, MATRIX_N*sizeof(float), TILE_N*sizeof(float), TILE_M,7,last_did1,0);
	last_did3 = 7;
    }
    QK_RETURN(last_did3);
}

void matmul_int8_reference(
    float* c
) {
    for (uint64_t i = 0; i < MATRIX_M; i++) {
    for (uint64_t j = 0; j < MATRIX_N; j++) {
        float sum = 0;
        for (uint64_t tk = 0; tk < MATRIX_K; ++tk) {
            float a = a_quants[i * MATRIX_K + tk] * a_scales[(i * MATRIX_K + tk) / 64];
            float b = b_quants[j * MATRIX_K + tk] * b_scales[(j * MATRIX_K + tk) / 64];
            sum += a * b;
        }
        c[i * MATRIX_N + j] = sum;
    }}
}

void compare(const float* c, const float* cref, uint64_t m, uint64_t n) {
    for (uint64_t i = 0; i < MATRIX_M; i++) {
    for (uint64_t j = 0; j < MATRIX_N; j++) {
        float tc = c[i * MATRIX_N + j];
        float rc = cref[i * MATRIX_N + j];
        double rel = (tc - rc) / rc;
        if (rc > 1e-10 && fabs(rel) > 1e-1) {
            fprintf(stderr, "%" PRIu64 ",%" PRIu64 ": %g != %g  RE=%g\n", i, j, tc, rc, rel);
            exit(1);
        }
    }}
}

#ifndef THREADS
#define THREADS 8
#endif

#pragma clang target("arch=kestrel")
uint64_t asl_kernel(
    uint64_t kernel_did,
    uint64_t kdoid1,
    uint64_t thread_id,
    uint64_t thread_count) {
    QK_N2ONE_CK(0,128,1,kdoid1);
    if (thread_id == 0) {
        printf("I'm asl kernel!\n");
    }
    QK_RETURN(1);
}

void softmax_reference(float* c) {
    uint64_t i,s;

    for (i = 0; i < MATRIX_M; i++)
        for (s = 0; s < MATRIX_N; s += TILE_N)
            jsoftmax_internal(c + i*MATRIX_N + s);
}

#pragma clang target("arch=kestrel")
uint64_t softmax_kernel(
    uint64_t kernel_did,
    uint64_t kdoid1,
    uint64_t thread_id,
    uint64_t thread_count,
    float* c)
{
    uint64_t i;

    const uint64_t tiles_m = MATRIX_M / TILE_M;
    const uint64_t tiles_n = MATRIX_N / TILE_N;
    const uint64_t tiles = tiles_m * tiles_n;

    const uint64_t tile0 = thread_id * tiles / thread_count;
    const uint64_t tile1 = (thread_id + 1) * tiles / thread_count;

    QK_ONE2ONE_CK(1,kdoid1);
    DID_T last_did = 1;
    for (uint64_t tile = tile0; tile < tile1; ++tile) {
        const uint64_t tile_n = tile % tiles_n;
        const uint64_t tile_m = tile / tiles_n;

        float* tile_c = c + (tile_m * TILE_M * MATRIX_N) + tile_n * TILE_N;
        for (i = 0; i < TILE_M; i++) {
            QK_CORE_NT_1(jsoftmax,tile_c+i*MATRIX_N,2,last_did,0,0);
	    last_did = 2;
	}
    }
    QK_RETURN(last_did);
}

#pragma clang target("arch=kestrel")
void matmul_softmax(
    uint64_t thread_id,
    uint64_t thread_count,
    float* c)
{
    uint64_t kdid[3];
    kdid[0] = matmul_int8_kernel(0, thread_id, thread_count, c);
    kdid[1] = asl_kernel(1, kdid[0], thread_id, thread_count);
    kdid[2] = softmax_kernel(2, kdid[0], thread_id, thread_count, c);
    Q_STOP_NT(kdid[2],0,0);
}

void test(void) {
    std::vector<int8_t> qa(MATRIX_M * MATRIX_K);
    std::vector<int8_t> qb(MATRIX_N * MATRIX_K);
    std::vector<float> sa(MATRIX_M * MATRIX_K / TILE_K);
    std::vector<float> sb(MATRIX_N * MATRIX_K / TILE_K);
    std::vector<float> c(MATRIX_M * MATRIX_N);
    std::vector<float> cref(MATRIX_M * MATRIX_N);

    // std::random_device rndd;
    std::mt19937 rnde{0}; //rndd()};
    std::uniform_int_distribution<int8_t> dist8{-128, 127};
    std::uniform_real_distribution<float> distf{-1.0f, 1.0f};

    std::generate(qa.begin(), qa.end(), [&]() { return dist8(rnde); });
    std::generate(qb.begin(), qb.end(), [&]() { return dist8(rnde); });
    std::generate(sa.begin(), sa.end(), [&]() { return distf(rnde); });
    std::generate(sb.begin(), sb.end(), [&]() { return distf(rnde); });

    // TODO: g.appendCopyToDevice. Copy data from host to HBM.
    a_quants = qa.data();
    a_scales = sa.data(),
    b_quants = qb.data();
    b_scales = sb.data();

    // TODO: g.appendLaunchKernel. Load kernel blob to kestrel HBM
    // kestrel_kernel = (void *)matmul_softmax;

    // TODO: g.execute. Trigger the kernel run on kestrel
    std::array<std::thread, THREADS> threads;
    for (int i = 0; i < THREADS; i++) {
        threads[i] = std::thread([&, i]() {
	    matmul_softmax(i, THREADS, c.data());
        });
    }
    for (int i = 0; i < THREADS; i++) {
        threads[i].join();
    }

    // TODO: g.appendCopyToHost. Copy data back from HBM to host.
    // ...

    // Compare with host reference
    matmul_int8_reference(cref.data());
    softmax_reference(cref.data());
    compare(c.data(), cref.data(), MATRIX_M, MATRIX_N);
}

int main() {
    Q_INIT();
    test();
    Q_EXIT();

    printf("PASS!\n");
    return 0;
}
