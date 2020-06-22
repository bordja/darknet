#ifndef __PERSPECTIVE_TRANSFORM_H__
#define __PERSPECTIVE_TRANSFORM_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void cv_copy_to_input_perspective(void* input);
void cv_copy_from_output_perspective(void* output);
void pixel_perspective_transform(int x, int y, int* x_new, int* y_new);
bool mouse_click_and_param_init(void* init_bgr_frame, const char* cv_window_name);
void get_perspective_transform(void);

#ifdef __cplusplus
}
#endif
#endif /*__PERSPECTIVE_TRANSFORM_H__*/