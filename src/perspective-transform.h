#ifndef __PERSPECTIVE_TRANSFORM_H__
#define __PERSPECTIVE_TRANSFORM_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//#define CAMERA_1_POLE_1_ID           3284220
//#define CAMERA_1_POLE_2_ID           3284458
//#define CAMERA_1_POLE_3_ID           3284221
//#define CAMERA_1_POLE_4_ID           3284457

/* Camera IDs for KPIs */
#define CAMERA_1_POLE_1_ID           3284221
#define CAMERA_1_POLE_2_ID           3284457
#define CAMERA_1_POLE_3_ID           3284222
#define CAMERA_1_POLE_4_ID           3284456

#define CAMERA_2_POLE_1_ID           3284453
#define CAMERA_2_POLE_2_ID           3284225
#define CAMERA_2_POLE_3_ID           3284454
#define CAMERA_2_POLE_4_ID           3284224

#define CAMERA_3_POLE_1_ID           3284456
#define CAMERA_3_POLE_2_ID           3284222
#define CAMERA_3_POLE_3_ID           3284457
#define CAMERA_3_POLE_4_ID           3284221

#define CAMERA_4_POLE_1_ID           3284218
#define CAMERA_4_POLE_2_ID           3284460
#define CAMERA_4_POLE_3_ID           3284219
#define CAMERA_4_POLE_4_ID           3284459

typedef enum cv_Color
{
    PURPLE,
    LIGHT_BLUE
}cv_Color;

typedef struct cv_Quadrangle
{
    int x0, y0;     // top-left
    int x1, y1;     // top-right
    int x2, y2;     // bottom-left
    int x3, y3;     // bottom-right
    int xc, yc;     // center
}cv_Quadrangle;

extern int pole_ids_init[4];
extern uint16_t pole_perspective_loc_x[4];
extern uint16_t pole_perspective_loc_y[4];

void cv_copy_to_input_perspective(void* input);
void cv_copy_from_output_perspective(void* output);
void deinit_perspective_params(void);
int pixel_perspective_transform(int x, int y, int* x_new, int* y_new);
int detection_perspective_transform(int x0, int y0, int x_center, int y_center, int width, int height, cv_Quadrangle* out);
void conversion_quad_rect(int x0, int y0, int width, int height, cv_Quadrangle* out);
bool mouse_click_and_param_init(void* init_bgr_frame, const char* cv_window_name);
void get_perspective_transform(void);

#ifdef __cplusplus
}
#endif
#endif /*__PERSPECTIVE_TRANSFORM_H__*/