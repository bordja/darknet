#include "image_fstream.h"
#include <fstream>
#include <iostream>

/*extern "C"*/ static std::ifstream in_stream;
/*extern "C"*/ static std::ofstream out_stream;
/*extern "C"*/ static std::ifstream timestamps;
#ifdef __cplusplus
extern "C" {
#endif
    
    extern "C" void fstream_open(const char *filename, int mode) {
        if (mode == INPUT) {
            std::cout << "Opening input stream " <<filename<< std::endl;
            in_stream.open(filename, std::ios_base::binary);
            if (in_stream.is_open()) {
                std::cout << "Input stream open" << std::endl;
            }
            else {
                std::cout << "Can not open input stream" << std::endl;
            }
        }
        else if (mode == OUTPUT) {
            std::cout << "Opening output stream " <<filename<< std::endl;
            out_stream.open(filename, std::ios_base::binary);
            if (out_stream.is_open()) {
                std::cout << "Output stream open" << std::endl;
            }
            else {
                std::cout << "Can not open output stream" << std::endl;
            }
        }
        else if (mode == TIMESTAMPS) {
            std::cout << "Opening timestamp stream " <<filename<< std::endl;
            timestamps.open(filename, std::ios_base::binary);
            if (timestamps.is_open()) {
                std::cout << "Timestamp stream open" << std::endl;
            }
            else {
                std::cout << "Can not open timestamp stream" << std::endl;
            }
        }
        else {
            std::cout << "Invalid mode for file stream" << std::endl;
            exit(-1);
        }
        return;
    }

    extern "C" int fstream_is_open(int mode) {
        if (mode == INPUT)
            return in_stream.is_open();
        else if (mode == OUTPUT)
            return out_stream.is_open();
        else if (mode == TIMESTAMPS)
            return timestamps.is_open();
        else {
            std::cout << "Invalid mode for file stream" << std::endl;
            exit(-1);
        }
    }

    extern "C" int fstream_eof(int mode) {
        if (mode == INPUT)
            return in_stream.eof();
        else if (mode == TIMESTAMPS)
            return timestamps.eof();
    }

    extern "C" void fstream_read(unsigned char* out, int size, int mode) {
        if (mode == INPUT)
            in_stream.read((char*)out, size);
        else if (mode == TIMESTAMPS)
            timestamps.read((char*)out, size);
        return;
    }

    extern "C" void fstream_write(const char* buffer, int size) {
        out_stream.write(buffer, size);
        return;
    }

    extern "C" void fstream_close(int mode) {
        if (mode == INPUT)
            in_stream.close();
        else if (mode == OUTPUT)
            out_stream.close();
        else if (mode == TIMESTAMPS)
            timestamps.close();
        else {
            std::cout << "Invalid mode for file stream" << std::endl;
            exit(-1);
        }
        return;
    }
#ifdef __cplusplus
}
#endif