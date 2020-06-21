#ifndef __PERSPECTIVE_TRANSFORM_H__
#define __PERSPECTIVE_TRANSFORM_H__

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void cv_copy_to_input_perspective(void* input);
extern "C" void pixel_perspective_transform(void* perspective_img, int x, int y, int* x_new, int* y_new);
extern "C" bool mouse_click_and_param_init(void* init_bgr_frame, const char* cv_window_name);
extern "C" void get_perspective_transform(void* perspective_img);

#ifdef __cplusplus
}
#endif
#endif /*__PERSPECTIVE_TRANSFORM_H__*/