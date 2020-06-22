#ifndef IMAGE_FSTREAM_H
#define IMAGE_FSTREAM_H

#ifdef __cplusplus
extern "C" {
#endif

void fstream_open(const char *filename);
int fstream_is_open(void);
int fstream_eof(void);
void fstream_read(unsigned char* out, int size);

#ifdef __cplusplus
}
#endif
#endif