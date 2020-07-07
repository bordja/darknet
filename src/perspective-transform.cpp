#include "IPM.h"
#include "mouse-event.h"
#include "perspective-transform.h"

#define WIDTH               1920
#define HEIGHT              1080
#define POLE_DISTANCE       (HEIGHT / 5)
#define FRAME_SIZE_UYVY     (WIDTH * HEIGHT * 2)
#define PI                  3.1415926

#ifdef __cplusplus
extern "C" {
#endif
#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>

int pole_ids_init[4] = {POLE_1_ID, POLE_2_ID, POLE_3_ID, POLE_4_ID};

static cv::Mat input_perspective;

/* Params for I perspective transformation (Warp) */
static cv::Mat result_warp;
static bool defined_warp_pole = false;
static cv::Point2f dst_warp[4];
static cv::Point2f pole_locations_warp[4];

/* Params for II perspective transformation (Inverse) */
static cv::Mat result_inverse;
static bool defined_inverse_pole = false;
static std::vector<cv::Point2f> src_inverse;
static std::vector<cv::Point2f> dst_inverse;
static std::vector<cv::Point2f> pole_locations_inverse;

/* Params for III perspective transformation (Finetune) */
static cv::Mat result_finetune;
static bool defined_finetune_pole = false;
static std::vector<cv::Point2f> src_finetune;
static std::vector<cv::Point2f> dst_finetune;
static std::vector<cv::Point2f> dst_finetune_next_rect_1;
static std::vector<cv::Point2f> dst_finetune_next_rect_2;
static std::vector<cv::Point2f> dst_finetune_next_rect_3;

enum line_Mofidy {
    SPREAD = 0,
    SPREAD_UP = 1,
    SPREAD_DOWN = 2,
    COMPRESS = 3,
    TRANSLATE_UP = 4,
    TRANSLATE_DOWN = 5,
    SPREAD_MAX = 6
};

enum CameraView
{
    LEFT,
    RIGHT
};

static void vertical_road_line_modification(cv::Point2f src[], cv::Point2f dst[],
int weight, enum line_Mofidy mode, enum CameraView camView)
{
    double w_0_x, w_0_y, w_0, w_1_x, w_1_y, w_1, w;
    int diff_x = abs(src[0].x - src[1].x);
    int diff_y = abs(src[0].y - src[1].y);
    double theta = atan2(diff_y, diff_x);

    switch (camView)
    {
    case LEFT:
        switch (mode)
        {
        case SPREAD:
            dst[0].x = int(src[0].x - weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            dst[1].x = int(src[1].x + weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case SPREAD_UP:
            dst[0].x = int(src[0].x - weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            break;
        case SPREAD_DOWN:
            dst[1].x = int(src[1].x + weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case COMPRESS:
            dst[0].x = int(src[0].x + weight * cos(theta));
            dst[0].y = int(src[0].y + weight * sin(theta));
            dst[1].x = int(src[1].x - weight * cos(theta));
            dst[1].y = int(src[1].y - weight * sin(theta));
            break;
        case TRANSLATE_UP:
            dst[0].x = int(src[0].x - weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            dst[1].x = int(src[1].x - weight * cos(theta));
            dst[1].y = int(src[1].y - weight * sin(theta));
            break;
        case TRANSLATE_DOWN:
            dst[0].x = int(src[0].x + weight * cos(theta));
            dst[0].y = int(src[0].y + weight * sin(theta));
            dst[1].x = int(src[1].x + weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case SPREAD_MAX:
            w_0_x = (src[0].x - 0) / cos(theta);
            w_0_y = (src[0].y - 0) / sin(theta);
            w_0 = std::min(w_0_x, w_0_y);
            w_1_x = (WIDTH - src[1].x) / cos(theta);
            w_1_y = (HEIGHT - src[1].y) / sin(theta);
            w_1 = std::min(w_1_x, w_1_y);
            w = std::min(w_0, w_1);
            dst[0].x = int(src[0].x - w * cos(theta));
            dst[0].y = int(src[0].y - w * sin(theta));
            dst[1].x = int(src[1].x + w * cos(theta));
            dst[1].y = int(src[1].y + w * sin(theta));
            break;
        default:
            std::cout << "LEFT: Unrecognized line modification mode!" << std::endl;
            break;
        }
        break;
    case RIGHT:
        switch (mode)
        {
        case SPREAD:
            dst[0].x = int(src[0].x + weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            dst[1].x = int(src[1].x - weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case SPREAD_UP:
            dst[0].x = int(src[0].x + weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            break;
        case SPREAD_DOWN:
            dst[1].x = int(src[1].x - weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case COMPRESS:
            dst[0].x = int(src[0].x - weight * cos(theta));
            dst[0].y = int(src[0].y + weight * sin(theta));
            dst[1].x = int(src[1].x + weight * cos(theta));
            dst[1].y = int(src[1].y - weight * sin(theta));
            break;
        case TRANSLATE_UP:
            dst[0].x = int(src[0].x + weight * cos(theta));
            dst[0].y = int(src[0].y - weight * sin(theta));
            dst[1].x = int(src[1].x + weight * cos(theta));
            dst[1].y = int(src[1].y - weight * sin(theta));
            break;
        case TRANSLATE_DOWN:
            dst[0].x = int(src[0].x - weight * cos(theta));
            dst[0].y = int(src[0].y + weight * sin(theta));
            dst[1].x = int(src[1].x - weight * cos(theta));
            dst[1].y = int(src[1].y + weight * sin(theta));
            break;
        case SPREAD_MAX:
            w_0_x = (WIDTH - src[0].x) / cos(theta);
            w_0_y = (src[0].y - 0) / sin(theta);
            w_0 = std::min(w_0_x, w_0_y);
            w_1_x = (src[1].x - 0) / cos(theta);
            w_1_y = (HEIGHT - src[1].y) / sin(theta);
            w_1 = std::min(w_1_x, w_1_y);
            w = std::min(w_0, w_1);
            dst[0].x = int(src[0].x + w * cos(theta));
            dst[0].y = int(src[0].y - w * sin(theta));
            dst[1].x = int(src[1].x - w * cos(theta));
            dst[1].y = int(src[1].y + w * sin(theta));
            break;
        default:
            std::cout << "RIGHT: Unrecognized line modification mode!" << std::endl;
            break;
        }
        break;
    default:
        std::cout << __func__ << " : Unrecognized angle." << std::endl;
        break;
    }
    if (dst[0].x >= WIDTH)
        dst[0].x = WIDTH - 1;
    if (dst[0].y >= HEIGHT)
        dst[0].y = HEIGHT - 1;
    if (dst[1].x >= WIDTH)
        dst[1].x = WIDTH - 1;
    if (dst[1].y >= HEIGHT)
        dst[1].y = HEIGHT - 1;
}

static void prepare_destination_warp(int start_point_high, int modification_point_high,
    int start_point_low, int modification_point_low)
{

    int line_high = (int)cv::norm(coordinates[start_point_high] - coordinates[modification_point_high]);
    int line_low = (int)cv::norm(coordinates[start_point_low] - coordinates[modification_point_low]);
    int line_offset_high = (WIDTH - line_high) / 2;
    int line_offset_low = (WIDTH - line_low) / 2;

    dst_warp[0] = cv::Point2f(line_offset_high, coordinates[start_point_high].y);
    dst_warp[1] = cv::Point2f(line_offset_high + line_high, coordinates[start_point_high].y);
    dst_warp[2] = cv::Point2f(line_offset_low, coordinates[start_point_low].y);
    dst_warp[3] = cv::Point2f(line_offset_low + line_low, coordinates[start_point_low].y);
}

static void prepare_source_destination_inverse(int start_point_high, int modification_point_high,
    int start_point_low, int modification_point_low)
{
    int line_high = (int)cv::norm(coordinates[start_point_high] - coordinates[modification_point_high]);
    int line_offset_high = (WIDTH - line_high) / 2;

    cv::Point2f new_points_warp[NUM_KEY_POINTS];
    int offset_height_inverse = abs(coordinates[start_point_high].y - coordinates[start_point_low].y) / 2;

    new_points_warp[0] = cv::Point2f(line_offset_high, HEIGHT - offset_height_inverse);
    new_points_warp[1] = cv::Point2f(line_offset_high + line_high, HEIGHT - offset_height_inverse);
    new_points_warp[2] = cv::Point2f(line_offset_high, HEIGHT);
    new_points_warp[3] = cv::Point2f(line_offset_high + line_high, HEIGHT);

    src_inverse.push_back(dst_warp[2]);
    src_inverse.push_back(dst_warp[3]);
    src_inverse.push_back(dst_warp[1]);
    src_inverse.push_back(dst_warp[0]);

    dst_inverse.push_back(new_points_warp[2]);
    dst_inverse.push_back(new_points_warp[3]);
    dst_inverse.push_back(new_points_warp[1]);
    dst_inverse.push_back(new_points_warp[0]);
}

static void next_rect(std::vector<cv::Point2f> src_points, std::vector<cv::Point2f>& dst_points)
{
    cv::Point2f tmp[4];

    tmp[0] = src_points[3];
    tmp[1] = src_points[2];

    int diff_x_1 = abs(src_points[2].x - src_points[1].x);
    int diff_y_1 = abs(src_points[2].y - src_points[1].y);
    int distance_1 = cv::norm(src_points[2] - src_points[1]);
    double theta_1 = atan2(diff_y_1, diff_x_1);

    tmp[2].x = int(src_points[2].x - distance_1 * cos(theta_1));
    tmp[2].y = int(src_points[2].y - distance_1 * sin(theta_1));

    int diff_x_2 = abs(src_points[3].x - src_points[0].x);
    int diff_y_2 = abs(src_points[3].y - src_points[0].y);
    int distance_2 = cv::norm(src_points[3] - src_points[0]);
    double theta_2 = atan2(diff_y_2, diff_x_2);

    tmp[3].x = int(src_points[3].x - distance_2 * cos(theta_2));
    tmp[3].y = int(src_points[3].y - distance_2 * sin(theta_2));

    dst_points.push_back(tmp[0]);
    dst_points.push_back(tmp[1]);
    dst_points.push_back(tmp[2]);
    dst_points.push_back(tmp[3]);
}

static void perspective_transform_element_wise(cv::Point2f src, cv::Point2f& dest, cv::Mat warpMatrix)
{
    dest.x = (int)((warpMatrix.at<double>(0,0)*src.x + warpMatrix.at<double>(0,1)*src.y + warpMatrix.at<double>(0,2)) /
            (warpMatrix.at<double>(2,0)*src.x + warpMatrix.at<double>(2,1)*src.y + warpMatrix.at<double>(2,2)));
    dest.y = (int)((warpMatrix.at<double>(1,0)*src.x + warpMatrix.at<double>(1,1)*src.y + warpMatrix.at<double>(1,2)) /
            (warpMatrix.at<double>(2,0)*src.x + warpMatrix.at<double>(2,1)*src.y + warpMatrix.at<double>(2,2)));
}

extern "C" void cv_copy_to_input_perspective(void* input)
{
    input_perspective = *(cv::Mat*)input;
}

extern "C" void cv_copy_from_output_perspective(void* output)
{
    *(cv::Mat*)output = result_finetune;
}

extern "C" void pixel_perspective_transform(int x, int y, int* x_new, int* y_new, cv_Color color)
{
    cv::Point2f input(x,y);
    cv::Point2f warp;
    cv::Point2f inverse;
    cv::Point2f finetune;

    /* I -> Warp perspective */
    cv::Mat warpMatrix = cv::getPerspectiveTransform(coordinates, dst_warp);
    perspective_transform_element_wise(input, warp, warpMatrix);

    /* II -> Inverse perspective */
    IPM ipm(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), src_inverse, dst_inverse);
    perspective_transform_element_wise(warp, inverse, ipm.getH());

    /* III -> Finetune perspective */
    IPM ipm_finetune(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), src_finetune, dst_finetune);
    perspective_transform_element_wise(inverse, finetune, ipm_finetune.getH());

    *x_new = finetune.x;
    *y_new = finetune.y;

    if (color == PURPLE)
        cv::circle(result_finetune, finetune, 8, cv::Scalar(255, 0, 255), -1);
    else if (color == LIGHT_BLUE)
        cv::circle(result_finetune, finetune, 8, cv::Scalar(255, 255, 0), -1);
}

extern "C" bool mouse_click_and_param_init(void* init_bgr_frame, const char* cv_window_name)
{
    cv::Mat* opencv_bgr_frame = (cv::Mat*)init_bgr_frame;
    static bool clickEventFinished = false;

    cv::namedWindow(cv_window_name, cv::WINDOW_AUTOSIZE);
    if (!clickEventFinished)
    {
        if(!click_finished)
        {
            cv::setMouseCallback(cv_window_name, MouseCallBackFunc, (void*)&opencv_bgr_frame);
        }
        else
        {
            for (int n = 0; n < NUM_KEY_POINTS; n++)
                pole_locations_warp[n] = coordinates[n];

            enum CameraView cameraView;
            int start_point_low = 0;
            int start_point_high = 0;
            int modification_point_low;
            int modification_point_high;
            for (int s = 1; s < NUM_KEY_POINTS; s++)
            {
                if (coordinates[s].y > coordinates[start_point_low].y)
                    start_point_low = s;
                else if (coordinates[s].y < coordinates[start_point_high].y)
                    start_point_high = s;
            }

            if ((start_point_low + 1) < NUM_KEY_POINTS)
            {
                cameraView = LEFT;
                modification_point_low = start_point_low + 1;
                modification_point_high = start_point_high - 1;
            }
            else
            {
                cameraView = RIGHT;
                modification_point_low = start_point_low - 1;
                modification_point_high = start_point_high + 1;
            }

            cv::Point2f line_modify_l[2];
            cv::Point2f line_modify_r[2];
            double line_distance_l;
            double line_distance_r;
            double line_ratio;
            int distance = 3 * POLE_DISTANCE / 2;
            switch (cameraView)
            {
            case LEFT:
                line_modify_l[0] = coordinates[start_point_high - 1];
                line_modify_l[1] = coordinates[start_point_low];
                line_modify_r[0] = coordinates[start_point_high];
                line_modify_r[1] = coordinates[start_point_low + 1];
                line_distance_l = cv::norm(line_modify_l[0] - line_modify_l[1]);
                line_distance_r = cv::norm(line_modify_r[0] - line_modify_r[1]);

                if (line_distance_l > line_distance_r)
                {
                    line_ratio = line_distance_l / line_distance_r;
                    line_distance_l = distance;
                    line_distance_r = (int)(distance / line_ratio);
                }
                else
                {
                    line_ratio = line_distance_r / line_distance_l;
                    line_distance_l = (int)(distance / line_ratio);
                    line_distance_r = distance;
                }

                vertical_road_line_modification(line_modify_l, line_modify_l, (int)line_distance_l, SPREAD, LEFT);
                vertical_road_line_modification(line_modify_r, line_modify_r, (int)line_distance_r, SPREAD, LEFT);
                coordinates[start_point_high - 1] = line_modify_l[0];
                coordinates[start_point_low] = line_modify_l[1];
                coordinates[start_point_high] = line_modify_r[0];
                coordinates[start_point_low + 1] = line_modify_r[1];
                break;
            case RIGHT:
                line_modify_l[0] = coordinates[start_point_high];
                line_modify_l[1] = coordinates[start_point_low - 1];
                line_modify_r[0] = coordinates[start_point_high + 1];
                line_modify_r[1] = coordinates[start_point_low];
                line_distance_l = cv::norm(line_modify_l[0] - line_modify_l[1]);
                line_distance_r = cv::norm(line_modify_r[0] - line_modify_r[1]);

                if (line_distance_l > line_distance_r)
                {
                    line_ratio = line_distance_l / line_distance_r;
                    line_distance_l = distance;
                    line_distance_r = (int)(distance / line_ratio);
                }
                else
                {
                    line_ratio = line_distance_r / line_distance_l;
                    line_distance_l = (int)(distance / line_ratio);
                    line_distance_r = distance;
                }

                vertical_road_line_modification(line_modify_l, line_modify_l, (int)line_distance_l, SPREAD, RIGHT);
                vertical_road_line_modification(line_modify_r, line_modify_r, (int)line_distance_r, SPREAD, RIGHT);
                coordinates[start_point_high] = line_modify_l[0];
                coordinates[start_point_low - 1] = line_modify_l[1];
                coordinates[start_point_high + 1] = line_modify_r[0];
                coordinates[start_point_low] = line_modify_r[1];
                break;
            }

            prepare_destination_warp(start_point_high, modification_point_high,
                start_point_low, modification_point_low);
            prepare_source_destination_inverse(start_point_high, modification_point_high,
                start_point_low, modification_point_low);

            clickEventFinished = true;
        }
    }
    for (int i = 0; i < mouse_move_cnt; i++)
        circle(*opencv_bgr_frame, coordinates[i], 16, cv::Scalar(0, 0, 255), -1);
    cv::imshow(cv_window_name, *opencv_bgr_frame);
    cv::waitKey(1);

    return clickEventFinished;
}

extern "C" void get_perspective_transform(void)
{
    cv::Mat warpMatrix = cv::getPerspectiveTransform(coordinates, dst_warp);
    if (!defined_warp_pole)
    {
        for (int i = 0; i < NUM_KEY_POINTS; i++)
            perspective_transform_element_wise(pole_locations_warp[i], pole_locations_warp[i], warpMatrix);
        defined_warp_pole = true;
    }
    cv::warpPerspective(input_perspective, result_warp, warpMatrix, result_warp.size());
    for (int m = 0; m < mouse_move_cnt; m++)
        circle(result_warp, pole_locations_warp[m], 5, cv::Scalar(255, 0, 255), -1);

    IPM ipm(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), src_inverse, dst_inverse);
    if (!defined_inverse_pole)
    {
        pole_locations_inverse.push_back(pole_locations_warp[0]);
        pole_locations_inverse.push_back(pole_locations_warp[1]);
        pole_locations_inverse.push_back(pole_locations_warp[2]);
        pole_locations_inverse.push_back(pole_locations_warp[3]);
        for (int i = 0; i < NUM_KEY_POINTS; i++)
            perspective_transform_element_wise(pole_locations_inverse[i], pole_locations_inverse[i], ipm.getH());
        defined_inverse_pole = true;
    }
    ipm.applyHomography(result_warp, result_inverse);

    if (!defined_finetune_pole)
    {
        src_finetune.push_back(pole_locations_inverse[2]);
        src_finetune.push_back(pole_locations_inverse[3]);
        src_finetune.push_back(pole_locations_inverse[1]);
        src_finetune.push_back(pole_locations_inverse[0]);

        int Yl = std::min(src_finetune[0].y, src_finetune[1].y);
        dst_finetune.push_back(cv::Point2f(src_finetune[0].x, Yl));
        dst_finetune.push_back(cv::Point2f(src_finetune[1].x, Yl));
        dst_finetune.push_back(cv::Point2f(src_finetune[1].x, Yl - POLE_DISTANCE));
        dst_finetune.push_back(cv::Point2f(src_finetune[0].x, Yl - POLE_DISTANCE));

        next_rect(dst_finetune, dst_finetune_next_rect_1);
        next_rect(dst_finetune_next_rect_1, dst_finetune_next_rect_2);
        next_rect(dst_finetune_next_rect_2, dst_finetune_next_rect_3);
        defined_finetune_pole = true;
    }
    IPM ipm_finetune(cv::Size(WIDTH, HEIGHT), cv::Size(WIDTH, HEIGHT), src_finetune, dst_finetune);
    ipm_finetune.applyHomography(result_inverse, result_finetune);

    ipm.drawPoints(src_inverse, result_warp, cv::Scalar(0,205,205));
    ipm.drawPoints(dst_inverse, result_inverse, cv::Scalar(0,205,205));

    ipm.drawPoints(dst_finetune, result_finetune, cv::Scalar(0,205,205));
    ipm.drawPoints(dst_finetune_next_rect_1, result_finetune, cv::Scalar(255,0,0));
    ipm.drawPoints(dst_finetune_next_rect_2, result_finetune, cv::Scalar(0,0,255));
    ipm.drawPoints(dst_finetune_next_rect_3, result_finetune, cv::Scalar(120,120,120));
}
#ifdef __cplusplus
}
#endif