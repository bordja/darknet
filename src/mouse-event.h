#ifndef __MOUSE_EVENT_H__
#define __MOUSE_EVENT_H__

#include <opencv2/opencv.hpp>

#define NUM_KEY_POINTS  4

extern int mouse_move_cnt;
extern bool click_finished;

/**
 * coordinates[0] -> top-left
 * coordinates[1] -> top-right
 * coordinates[2] -> bottom-left
 * coordinates[3] -> bottom-right
 */
extern cv::Point2f coordinates[NUM_KEY_POINTS];
void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata);

#endif /* __MOUSE_EVENT_H__ */