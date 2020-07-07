#ifndef __PERSPECTIVE_TRANSFORM_H__
#define __PERSPECTIVE_TRANSFORM_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAMERA_1_POLE_1_ID           3284453
#define CAMERA_1_POLE_2_ID           3284225
#define CAMERA_1_POLE_3_ID           3284454
#define CAMERA_1_POLE_4_ID           3284224

#define CAMERA_2_POLE_1_ID           3284453
#define CAMERA_2_POLE_2_ID           3284225
#define CAMERA_2_POLE_3_ID           3284454
#define CAMERA_2_POLE_4_ID           3284224

#define CAMERA_3_POLE_1_ID           3284453
#define CAMERA_3_POLE_2_ID           3284225
#define CAMERA_3_POLE_3_ID           3284454
#define CAMERA_3_POLE_4_ID           3284224

#define CAMERA_4_POLE_1_ID           3284453
#define CAMERA_4_POLE_2_ID           3284225
#define CAMERA_4_POLE_3_ID           3284454
#define CAMERA_4_POLE_4_ID           3284224

typedef enum cv_Color
{
    PURPLE,
    LIGHT_BLUE
}cv_Color;

extern int pole_ids_init[4];

void cv_copy_to_input_perspective(void* input);
void cv_copy_from_output_perspective(void* output);
void deinit_perspective_params(void);
void pixel_perspective_transform(int x, int y, int* x_new, int* y_new, cv_Color color);
bool mouse_click_and_param_init(void* init_bgr_frame, const char* cv_window_name);
void get_perspective_transform(void);

#ifdef __cplusplus
}
#endif
#endif /*__PERSPECTIVE_TRANSFORM_H__*/