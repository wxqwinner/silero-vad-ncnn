#include <iostream>
#include <algorithm>
#include "ncnn_vad.h"
#include "wav_reader.hpp"


#define EXPECTED_SAMPLE_RATE 16000
#define EXPECTED_CHUNK_SIZE (0.032 * EXPECTED_SAMPLE_RATE)

static void normalizeSamples(const std::vector<short>& samples_short, std::vector<float>& samples_float) {
    const float normFactor = 1.0f / 32768.0f;
    samples_float.clear();
    for (short sample : samples_short) {
        samples_float.push_back(static_cast<float>(sample) * normFactor);
    }
}

int main(int argc, char** argv) {
    if (argc!= 4)
    {
        fprintf(stderr, "Usage: %s <model.param> <model.bin>  <wave_file>\n", argv[0]);
        return -1;
    }
    

    // create vad net
    NcnnVad vad_net(argv[1], argv[2], false);

    // Read audio data
    int sampleRate, numSamples;
    std::vector<short> audioData = readWAV(argv[3], sampleRate, numSamples);
    fprintf(stderr, "Sample rate: %d\n", sampleRate);
    fprintf(stderr, "Number of samples: %d\n", numSamples);

    std::vector<float> samples_float;
    int k = 0;
    while (k < numSamples) {
        

        std::vector<short> samples_short(audioData.begin() + k, audioData.begin() + k + EXPECTED_CHUNK_SIZE);
        normalizeSamples(samples_short, samples_float);

        // Run inference
        float speech_score = vad_net.infer_samples(samples_float);
        if (speech_score > 0.5)
        {
            fprintf(stderr, "voice\n");
        }
        else
        {
            fprintf(stderr, "unvoice\n");
        }
        

        
        // k += EXPECTED_CHUNK_SIZE;
        k = std::min(k + (int)EXPECTED_CHUNK_SIZE, numSamples);
    }
    
    return 0;
}