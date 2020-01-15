#include <random>
#include <chrono>
#include <omp.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "Timer.hpp"

template <typename RanGen>
std::vector<float> randu(unsigned n, RanGen&&  gen) {
    std::uniform_real_distribution<> dist;
    std::vector<float> res(n);
    for (auto& x: res) x = dist(gen);
    return res;
}

float InnerProduct(const float* a, const float *b, unsigned dim) {
unsigned i;
float c= 0;
#pragma omp parallel for reduction(+:c)
for (i=0; i<dim; i++) {
 c += a[i] * b[i];
}

return c;
}

float InnerProduct_WO(const float* a, const float *b, unsigned dim) {
unsigned i;
float c= 0;

for (i=0; i<dim; i++) {
 c += a[i] * b[i];
}

return c;
}

constexpr unsigned G_SEED = 101u;
int main() {
    std::mt19937_64 gen(G_SEED);
    unsigned n = 1e4;
    auto a = randu(n, gen);
    auto b = randu(n, gen);
HighResolutionTimer timer;
timer.restart();
auto c1 = InnerProduct(&a[0],&b[0],n);
auto e1 = timer.elapsed();

timer.restart();
auto c2 = InnerProduct_WO(&a[0],&b[0],n);
auto e2 = timer.elapsed();

printf("With OpenMP: %.2f (res: %.6f)\nWithout: %.2f (res: %.6f)\n", e1, c1, e2, c2);

return 0;
}