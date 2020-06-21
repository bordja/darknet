#include "mouse-event.h"
#include <iostream>

int mouse_move_cnt = 0;
bool click_finished = false;
cv::Point2f coordinates[NUM_KEY_POINTS] = {cv::Point(0,0), cv::Point(0,0), cv::Point(0,0), cv::Point(0,0)};

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (++mouse_move_cnt <= NUM_KEY_POINTS)
        {
            cv::Mat *opencv_bgr_frame = (cv::Mat*)userdata;
            coordinates[mouse_move_cnt-1].x = x;
            coordinates[mouse_move_cnt-1].y = y;
            if (mouse_move_cnt == NUM_KEY_POINTS)
                click_finished = true;
        }
        else
        {
            std::cout << "No more clicking!" << std::endl;
        }
    }    
}