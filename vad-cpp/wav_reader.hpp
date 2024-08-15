#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

// wav reader class
struct WAVHeader {
    char riff[4];
    int chunkSize;
    char wave[4];
    char fmt[4];
    int subchunk1Size;
    short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    char data[4];
    int dataSize;
};

// read wav file and return samples
std::vector<short> readWAV(const std::string& filename, int& sampleRate, int& numSamples) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "can not open file " << filename << std::endl;
        return {};
    }

    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    if (std::strncmp(header.riff, "RIFF", 4) != 0 || std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "unsupported file format" << std::endl;
        return {};
    }

    sampleRate = header.sampleRate;
    numSamples = header.dataSize / (header.bitsPerSample / 8);

    std::vector<short> samples(numSamples);
    file.read(reinterpret_cast<char*>(samples.data()), header.dataSize);

    return samples;
}
