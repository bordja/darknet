#ifndef IMAGE_FSTREAM_H
#define IMAGE_FSTREAM_H

#define INPUT  0
#define OUTPUT 1

#ifdef __cplusplus
extern "C" {
#endif

void fstream_open(const char *filename, int mode);
int fstream_is_open(int mode);
int fstream_eof(void);
void fstream_read(unsigned char* out, int size);
void fstream_write(const char* buffer, int size);
void fstream_close(int mode);

#ifdef __cplusplus
}
#endif
#endif