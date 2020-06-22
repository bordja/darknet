#include "image_fstream.h"
#include <fstream>
#include <iostream>
/*extern "C"*/ static std::ifstream stream;
#ifdef __cplusplus
extern "C" {
#endif
    
    extern "C" void fstream_open(const char *filename) {
        std::cout << "Opening stream " <<filename<< std::endl;
        stream.open(filename, std::ios_base::binary);
        if (stream.is_open()) {
            std::cout << "Stream open" << std::endl;
        }
        else {
            std::cout << "Can not open stream" << std::endl;
        }
        return;
    }

    extern "C" int fstream_is_open() {
        return stream.is_open();
    }

    extern "C" int fstream_eof() {
        return stream.eof();
    }

    extern "C" void fstream_read(unsigned char* out, int size) {
        stream.read((char*)out, size);
        return;
    }
#ifdef __cplusplus
}
#endif