// File       : helper.h
// Description: Utilities (do not change code here)
// Copyright 2024 Harvard University. All Rights Reserved.
#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <vector>

// YOU DO NOT NEED TO CHANGE THIS CODE

struct Stats {
    double min, max, mean, p50, std, skew, kurtosis;
    Stats()
        : min(0.0), max(0.0), mean(0.0), p50(0.0), std(0.0), skew(0.0),
          kurtosis(0.0)
    {
    }
};

Stats statistics(std::vector<double> data)
{
    Stats s;
    double M2 = 0.0;
    double M3 = 0.0;
    double M4 = 0.0;
    size_t k = 0;
    for (const auto v : data) {
        const size_t k0 = k;
        ++k;
        const double delta0 = v - s.mean;
        const double delta1 = delta0 / k;
        const double delta2 = delta1 * delta1;
        const double tmp = delta0 * delta1 * k0;
        s.mean += delta1;
        M4 += tmp * delta2 * (k * k - 3.0 * k + 3.0) + 6.0 * delta2 * M2 -
              4.0 * delta1 * M3;
        M3 += tmp * delta1 * (k - 2.0) - 3.0 * delta1 * M2;
        M2 += tmp;
    }
    if (k > 1) {
        s.std = std::sqrt(M2 / (k - 1.0));
    }
    s.skew = std::sqrt(static_cast<double>(k)) * M3 / (std::pow(M2, 1.5));
    s.kurtosis = k * M4 / (M2 * M2) - 3.0;
    std::sort(data.begin(), data.end());
    s.min = data.front();
    if (0 == data.size() % 2) {
        s.p50 = 0.5 * (data[data.size() / 2] + data[data.size() / 2 + 1]);
    } else {
        s.p50 = data[data.size() / 2];
    }
    s.max = data.back();
    return s;
}

// timer
double get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6; // seconds
}
