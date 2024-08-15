#pragma once

#include "ncnn/net.h"

class NcnnVad {
    public:
        NcnnVad(const char* param_path, const char* bin_path, bool use_gpu = false);
        ~NcnnVad(){};
    private:
        ncnn::Net vad_net;

        // h & c
        std::vector<float> h;
        std::vector<float> c;

    public:
        float infer_samples(const std::vector<float>& sample_data);
};