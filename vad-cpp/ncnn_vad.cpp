#include "ncnn_vad.h"

NcnnVad::NcnnVad(const char* param_path, const char* bin_path, bool use_gpu)
{
    // net config
    vad_net.opt.use_vulkan_compute = use_gpu;
    vad_net.opt.use_fp16_packed = true;
    vad_net.opt.use_fp16_storage = true;
    vad_net.opt.use_fp16_arithmetic = true;
    vad_net.opt.num_threads = 1;

    // load model
    vad_net.load_param(param_path);
    vad_net.load_model(bin_path);    


    // init h & c
    h.resize(2 * 1 * 64, 0.0f);
    c.resize(2 * 1 * 64, 0.0f);
}


float
NcnnVad::infer_samples(const std::vector<float>& sample_data)
{
    bool is_speech = false;

    // ncnn mat for input
    ncnn::Mat in0(512, 1, 1);
    ncnn::Mat ncnn_h(64, 1, 2);
    ncnn::Mat ncnn_c(64, 1, 2);

    // in0
    memcpy(in0.data, sample_data.data(), 512 * sizeof(float));
    // h & c
    memcpy(ncnn_h.data, h.data(), h.size() * sizeof(float));
    memcpy(ncnn_c.data, c.data(), c.size() * sizeof(float));


    ncnn::Extractor ex = vad_net.create_extractor();

    ncnn::Mat out0, out1, out2;
    {
        // set input
        ex.input("in0", in0);
        ex.input("in1", ncnn_h);
        ex.input("in2", ncnn_c);

        // get output
        ex.extract("out0", out0);
        ex.extract("out1", out1);
        ex.extract("out2", out2);
    }

    // update h & c
    memcpy(h.data(), out1.data, h.size() * sizeof(float));
    memcpy(c.data(), out2.data, c.size() * sizeof(float));


    return out0[0];
}