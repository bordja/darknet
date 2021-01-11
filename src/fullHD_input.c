#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "darknet.h"
#include "option_list.h"
#include "fullHD_input.h"
#include "perspective-transform.h"
#include "tcp-ip-client.h"
#include <unistd.h>

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

#define TRESH (0.3F)
#define NFRAMES 3
#define WIDTH 1920
#define HEIGHT 1080
#define FRAME_SIZE_UYVY (WIDTH * HEIGHT * 2)

#define USE_QUADS

#ifdef OPENCV

#include "http_stream.h"
static float* predictions[NFRAMES];
static int demo_index = 0;
static mat_cv* cv_images[NFRAMES];
static char **demo_names;
static image **demo_alphabet;
static int demo_classes;
static float demo_thresh = 0;

static network net;
static int net_classes;
static image in_s;
static image det_s;

static int nboxes = 0;
static detection *dets = NULL;

static float *avg;
static volatile int flag_exit;
static long long int frame_id = 0;
int frame_skip = 0;
mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;
mat_cv* perspective_img;

cv_Detect car_perspective_detections[MAX_CAR_DETS];
cv_Detect person_perspective_detections[MAX_PERSON_DETS];

static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread_fullHD = 0;
static volatile int run_detect_in_thread_fullHD = 0;

void run_fullHD(char *cfgfile, char *weightfile, float thresh, char **names, int classes,
    const int8_t cameraID, const char *in_filename, const char* in_timestamps);

void *fetch_in_thread_fullHD(void *ptr);
void *fetch_in_thread_sync_fullHD(void *ptr);
void *detect_in_thread_fullHD(void *ptr);
void *detect_in_thread_sync_fullHD(void *ptr);
double get_wall_time_fullHD();

#define OUTPUT_PERSPECTIVE_PATH "/home/usorac/Desktop/Faculty/Master-rad/03-yolov4/CKPT-5/out_perspective_dets_v8/"
#define MAX_PATH_LENGTH 512
#define VIDEO_FPS 5

unsigned char uyvy_frame[FRAME_SIZE_UYVY];
void fullHD_input(int argc, char **argv)
{
    char *datacfg = argv[2];
    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;

    list *options = read_data_cfg(datacfg);
    int classes = option_find_int(options, "classes", 20);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    /* Camera 1 - Yolov4 */
    pole_ids_init[0] = CAMERA_1_POLE_1_ID;
    pole_ids_init[1] = CAMERA_1_POLE_2_ID;
    pole_ids_init[2] = CAMERA_1_POLE_3_ID;
    pole_ids_init[3] = CAMERA_1_POLE_4_ID;
    const int8_t cameraID_1 = 1;
    const char* in_filename_1 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-L/Out1.yuv";
    const char* in_timestamps_1 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-L/timestamp_1";
    run_fullHD(cfg, weights, TRESH, names, classes, cameraID_1, in_filename_1, in_timestamps_1);

    /* Camera 2 - Yolov4 */
    pole_ids_init[0] = CAMERA_2_POLE_1_ID;
    pole_ids_init[1] = CAMERA_2_POLE_2_ID;
    pole_ids_init[2] = CAMERA_2_POLE_3_ID;
    pole_ids_init[3] = CAMERA_2_POLE_4_ID;
    const int8_t cameraID_2 = 2;
    const char* in_filename_2 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-L/Out2.yuv";
    const char* in_timestamps_2 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-L/timestamp_2";
    run_fullHD(cfg, weights, TRESH, names, classes, cameraID_2, in_filename_2, in_timestamps_2);

    /* Camera 3 - Yolov4 */
    pole_ids_init[0] = CAMERA_3_POLE_1_ID;
    pole_ids_init[1] = CAMERA_3_POLE_2_ID;
    pole_ids_init[2] = CAMERA_3_POLE_3_ID;
    pole_ids_init[3] = CAMERA_3_POLE_4_ID;
    const int8_t cameraID_3 = 3;
    const char* in_filename_3 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-R/Out2.yuv";
    const char* in_timestamps_3 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-R/timestamp_2";
    run_fullHD(cfg, weights, TRESH, names, classes, cameraID_3, in_filename_3, in_timestamps_3);

    /* Camera 4 - Yolov4 */
    pole_ids_init[0] = CAMERA_4_POLE_1_ID;
    pole_ids_init[1] = CAMERA_4_POLE_2_ID;
    pole_ids_init[2] = CAMERA_4_POLE_3_ID;
    pole_ids_init[3] = CAMERA_4_POLE_4_ID;
    const int8_t cameraID_4 = 4;
    const char* in_filename_4 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-R/Out1.yuv";
    const char* in_timestamps_4 = "/home/usorac/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-08-16/KPI-1/DFG-R/timestamp_1";
    run_fullHD(cfg, weights, TRESH, names, classes, cameraID_4, in_filename_4, in_timestamps_4);

    free_ptrs((void **)names, net_classes);
}

void run_fullHD(char *cfgfile, char *weightfile, float thresh, char **names, int classes,
    const int8_t cameraID, const char *in_filename, const char* in_timestamps) {

    in_img = det_img = show_img = perspective_img = NULL;

    image **alphabet = load_alphabet();

    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    int delay = frame_skip;
    printf("Full HD\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }

    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    fstream_open(in_filename, INPUT);
    fstream_open(in_timestamps, TIMESTAMPS);
    
    if (fstream_is_open(INPUT) && fstream_is_open(TIMESTAMPS))
    {
        layer l = net.layers[net.n - 1];
        int j;

        avg = (float *)calloc(l.outputs, sizeof(float));
        for (j = 0; j < NFRAMES; ++j) predictions[j] = (float *)calloc(l.outputs, sizeof(float));

        if (l.classes != demo_classes) {
            printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
            getchar();
            exit(0);
        }

        flag_exit = 0;

        custom_thread_t fetch_thread = NULL;
        custom_thread_t detect_thread = NULL;
        if (custom_create_thread(&fetch_thread, 0, fetch_in_thread_fullHD, 0)) error("Thread creation failed");
        if (custom_create_thread(&detect_thread, 0, detect_in_thread_fullHD, 0)) error("Thread creation failed");

        fetch_in_thread_sync_fullHD(0);
        det_img = in_img;
        det_s = in_s;

        fetch_in_thread_sync_fullHD(0);
        detect_in_thread_sync_fullHD(0);
        det_img = in_img;
        det_s = in_s;

        for (j = 0; j < NFRAMES / 2; ++j) {
            free_detections(dets, nboxes);
            fetch_in_thread_sync_fullHD(0);
            detect_in_thread_sync_fullHD(0);
            det_img = in_img;
            det_s = in_s;
        }

        int count = 0;
        int full_screen = 0;
        create_window_cv("FullHD", full_screen, WIDTH, HEIGHT);

        /*
         * - Ouput perspective format:
         * 
         *  ----------------------------------------------------
         *  [0]                    -> Camera ID (1B)           |
         *                                                     |
         *  [1:4]                  -> POLE_1_ID (4B)           |
         *  [5:8]                  -> POLE_2_ID (4B)           |
         *  [9:12]                 -> POLE_3_ID (4B)           |
         *  [13:16]                -> POLE_4_ID (4B)           |
         *                                                     |
         *  [17:18]                -> pole_rel_1_x (2B)        |
         *  [19:20]                -> pole_rel_1_y (2B)        |- Camera/Stream Info
         *                                                     |
         *  [21:22]                -> pole_rel_2_x (2B)        |
         *  [23:24]                -> pole_rel_2_y (2B)        |
         *                                                     |
         *  [25:26]                -> pole_rel_3_x (2B)        |
         *  [27:28]                -> pole_rel_3_y (2B)        |
         *                                                     |
         *  [29:30]                -> pole_rel_4_x (2B)        |
         *  [31:32]                -> pole_rel_4_y (2B)        |
         *  ----------------------------------------------------
         *
         *  ----------------------------------------------------
         *  [33:40]                -> Frame_0_Timestamp (8B)   |
         *  [41:42]                -> PersonNumber (2B)        |
         *                                                     |
         *  [43:44]                -> x0 (2B)                  |
         *  [45:46]                -> y0 (2B)                  |
         *  [47:48]                -> width_0 (2B)             |
         *  [49:50]                -> height_0 (2B)            |
         *                                                     |
         *  [51:52]                -> x1 (2B)                  |
         *  [53:54]                -> y1 (2B)                  |
         *  [55:56]                -> width_1 (2B)             |
         *  [57:58]                -> height_1 (2B)            |
         *               ...                                   |
         *                                                     |- Frame_0 Info
         *  [(43+8*50+0):(43+8*50+1)]    -> CarNumer (2B)      |
         *                                                     |
         *  [(43+8*50+2):(43+8*50+3)]    -> x0 (2B)            |
         *  [(43+8*50+4):(43+8*50+5)]    -> y0 (2B)            |
         *  [(43+8*50+6):(43+8*50+7)]    -> width_0 (2B)       |
         *  [(43+8*50+8):(43+8*50+9)]    -> height_0 (2B)      |
         *                                                     |
         *  [(43+8*50+10):(43+8*50+11)]  -> x1 (2B)            |
         *  [(43+8*50+12):(43+8*50+13)]  -> y1 (2B)            |
         *  [(43+8*50+14):(43+8*50+15)]  -> width_1 (2B)       |
         *  [(43+8*50+16):(43+8*50+17)]  -> height_1 (2B)      |
         *                                                     |
         *               ...                                   |
         *                                                     |
         *  ----------------------------------------------------
         *               ...
         */
        const char out_perspective[MAX_PATH_LENGTH];
        snprintf(out_perspective, MAX_PATH_LENGTH - 1, OUTPUT_PERSPECTIVE_PATH "out_perspective_%d", cameraID);
        fstream_open(out_perspective, OUTPUT);
        if (!fstream_is_open(OUTPUT))
        {
            printf("Failed to open file: %s\n", out_perspective);
            return;
        }

        /* Opening input video */
        write_cv* in_video = NULL;
        const char in_video_path[MAX_PATH_LENGTH];
        snprintf(in_video_path, MAX_PATH_LENGTH - 1, OUTPUT_PERSPECTIVE_PATH "in_%d.mp4", cameraID);
        in_video = create_video_writer(in_video_path, 'M', 'J', 'P', 'G', VIDEO_FPS, WIDTH, HEIGHT, 1);

        /* Opening output video (perspective) */
        write_cv* out_perspective_video = NULL;
        const char out_perspective_video_path[MAX_PATH_LENGTH];
        snprintf(out_perspective_video_path, MAX_PATH_LENGTH - 1, OUTPUT_PERSPECTIVE_PATH "out_perspective_%d.mp4", cameraID);
        out_perspective_video = create_video_writer(out_perspective_video_path, 'M', 'J', 'P', 'G', VIDEO_FPS, WIDTH, HEIGHT, 1);

        float avg_fps = 0;
        bool finished_clicking = false;
        bool window_created = false;
        bool poles_sent = false;

        fstream_write((char*)&cameraID, sizeof(cameraID));

        int num_poles_init = (&pole_ids_init)[1] - pole_ids_init;
        for (int i = 0; i < num_poles_init; i++)
        {
            uint32_t pole_id = (uint32_t)pole_ids_init[i];
            fstream_write((char*)&pole_id, sizeof(pole_id));
        }

        while (!fstream_eof(INPUT) && !fstream_eof(TIMESTAMPS)) {
            ++count;
            {
                const float nms = .45;
                int local_nboxes = nboxes;
                detection *local_dets = dets;
                this_thread_yield();

                custom_atomic_store_int(&run_fetch_in_thread_fullHD, 1);
                custom_atomic_store_int(&run_detect_in_thread_fullHD, 1);

                if (nms) {
                    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                    else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }

                ++frame_id;
                uint64_t current_timestamp;
                fstream_read((char*)&current_timestamp, sizeof(current_timestamp), TIMESTAMPS);

                if (det_img != NULL)
                    cv_copy_to_input_perspective((void*)det_img);
                draw_detection_and_point(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

                if (!finished_clicking && show_img != NULL)
                    finished_clicking = mouse_click_and_param_init((void*)show_img, "FullHD");
                else if (show_img != NULL)
                {
                    draw_frame_ID(show_img, (int)frame_id);
                    write_frame_cv(in_video, show_img);
                    show_image_mat(show_img, "FullHD");
                    perspective_img = show_img;
                    if (!window_created)
                    {
                        create_window_cv("Perspective transform", full_screen, WIDTH, HEIGHT);
                        window_created = true;
                    }
                    get_perspective_transform();
                    cv_copy_from_output_perspective((void*)perspective_img);

                    if (!poles_sent)
                    {
                        int pole_nums = (&pole_perspective_loc_x)[1] - pole_perspective_loc_x;
                        for (int i = 0; i < pole_nums; i++)
                        {
                            fstream_write((char*)&(pole_perspective_loc_x[i]), sizeof(pole_perspective_loc_x[i]));
                            fstream_write((char*)&(pole_perspective_loc_y[i]), sizeof(pole_perspective_loc_y[i]));
                        }
                        poles_sent = true;
                    }

                    fstream_write((char*)&current_timestamp, sizeof(current_timestamp));

                    int person_indx = 0;
                    cv_Color color = PURPLE;
                    for (int dets = 0; dets < num_persons; dets++)
                    {
#ifdef USE_QUADS
                        int retVal = detection_perspective_transform((cv_Detect*)&person_detections[dets],
                                &person_perspective_detections[person_indx]);
#else
                        int retVal = pixel_perspective_transform(person_detections[dets].x_center, person_detections[dets].y_center,
                                &person_perspective_detections[person_indx].x0, &person_perspective_detections[person_indx].y0);
#endif
                        if (retVal == 0)
                        {
                            int person_class_id = 0;
                            point_cv quad_pts[4];
                            quad_pts[0].x = person_perspective_detections[person_indx].x0;
                            quad_pts[0].y = person_perspective_detections[person_indx].y0;
                            quad_pts[1].x = person_perspective_detections[person_indx].x0 + person_perspective_detections[person_indx].width;
                            quad_pts[1].y = person_perspective_detections[person_indx].y0;
                            quad_pts[2].x = person_perspective_detections[person_indx].x0 + person_perspective_detections[person_indx].width;
                            quad_pts[2].y = person_perspective_detections[person_indx].y0 + person_perspective_detections[person_indx].height;
                            quad_pts[3].x = person_perspective_detections[person_indx].x0;
                            quad_pts[3].y = person_perspective_detections[person_indx].y0 + person_perspective_detections[person_indx].height;
#ifdef USE_QUADS
                            draw_custom_shape(perspective_img, quad_pts, 4, person_class_id, "person", person_detections[dets].prob);
#else
                            draw_custom_shape(perspective_img, quad_pts, 1, person_class_id, "person", person_detections[dets].prob);
#endif
                            person_indx++;
                        }
                    }

                    uint16_t Persons = (uint16_t)person_indx;
                    fstream_write((char*)&Persons, sizeof(Persons));
                    for (int dets = 0; dets < MAX_PERSON_DETS; dets++)
                    {
                        uint16_t person_x0 = 0;
                        uint16_t person_y0 = 0;
                        uint16_t person_width = 0;
                        uint16_t person_height = 0;
                        if (dets < Persons)
                        {
                            person_x0 = (uint16_t)person_perspective_detections[dets].x0;
                            person_y0 = (uint16_t)person_perspective_detections[dets].y0;
                            person_width = (uint16_t)person_perspective_detections[dets].width;
                            person_height = (uint16_t)person_perspective_detections[dets].height;
                        }
                        fstream_write((char*)&person_x0, sizeof(person_x0));
                        fstream_write((char*)&person_y0, sizeof(person_y0));
                        fstream_write((char*)&person_width, sizeof(person_width));
                        fstream_write((char*)&person_height, sizeof(person_height));
                    }

                    int car_indx = 0;
                    color = LIGHT_BLUE;
                    for (int dets = 0; dets < num_cars; dets++)
                    {
#ifdef USE_QUADS
                        int retVal = detection_perspective_transform((cv_Detect*)&car_detections[dets],
                                &car_perspective_detections[car_indx]);
#else
                        int retVal = pixel_perspective_transform(car_detections[dets].x_center, car_detections[dets].y_center,
                                &car_perspective_detections[car_indx].x0, &car_perspective_detections[car_indx].y0);
#endif
                        if (retVal == 0)
                        {
                            int x_center = car_perspective_detections[car_indx].x_center;
                            if ((x_center > pole_perspective_loc_x[0]) && (x_center < pole_perspective_loc_x[1]))
                            {
                                int car_class_id = 2;
                                point_cv quad_pts[4];
                                quad_pts[0].x = car_perspective_detections[car_indx].x0;
                                quad_pts[0].y = car_perspective_detections[car_indx].y0;
                                quad_pts[1].x = car_perspective_detections[car_indx].x0 + car_perspective_detections[car_indx].width;
                                quad_pts[1].y = car_perspective_detections[car_indx].y0;
                                quad_pts[2].x = car_perspective_detections[car_indx].x0 + car_perspective_detections[car_indx].width;
                                quad_pts[2].y = car_perspective_detections[car_indx].y0 + car_perspective_detections[car_indx].height;
                                quad_pts[3].x = car_perspective_detections[car_indx].x0;
                                quad_pts[3].y = car_perspective_detections[car_indx].y0 + car_perspective_detections[car_indx].height;
#ifdef USE_QUADS
                                draw_custom_shape(perspective_img, quad_pts, 4, car_class_id, "car", car_detections[dets].prob);
#else
                                draw_custom_shape(perspective_img, quad_pts, 1, car_class_id, "car", car_detections[dets].prob);
#endif
                                car_indx++;
                            }
                        }
                    }

                    uint16_t Cars = (uint16_t)car_indx;
                    fstream_write((char*)&Cars, sizeof(Cars));
                    for (int dets = 0; dets < MAX_CAR_DETS; dets++)
                    {
                        uint16_t car_x0 = 0;
                        uint16_t car_y0 = 0;
                        uint16_t car_width = 0;
                        uint16_t car_height = 0;
                        if (dets < Cars)
                        {
                            car_x0 = (uint16_t)car_perspective_detections[dets].x0;
                            car_y0 = (uint16_t)car_perspective_detections[dets].y0;
                            car_width = (uint16_t)car_perspective_detections[dets].width;
                            car_height = (uint16_t)car_perspective_detections[dets].height;
                        }
                        fstream_write((char*)&car_x0, sizeof(car_x0));
                        fstream_write((char*)&car_y0, sizeof(car_y0));
                        fstream_write((char*)&car_width, sizeof(car_width));
                        fstream_write((char*)&car_height, sizeof(car_height));
                    }
                    draw_frame_ID(perspective_img, (int)frame_id);
                    write_frame_cv(out_perspective_video, perspective_img);
                    show_image_mat(perspective_img, "Perspective transform");
                }
                for (int dets = 0; dets < num_cars; dets++)
                {
                    car_detections[dets].x0 = 0;
                    car_detections[dets].y0 = 0;
                    car_detections[dets].width = 0;
                    car_detections[dets].height = 0;
                }
                for (int dets = 0; dets < num_persons; dets++)
                {
                    person_detections[dets].x0 = 0;
                    person_detections[dets].y0 = 0;
                    person_detections[dets].width = 0;
                    person_detections[dets].height = 0;
                }
                free_detections(local_dets, local_nboxes);

                int c = wait_key_cv(1);
                if (c == 10) {
                    if (frame_skip == 0) frame_skip = 60;
                    else if (frame_skip == 4) frame_skip = 0;
                    else if (frame_skip == 60) frame_skip = 4;
                    else frame_skip = 0;
                }
                else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                {
                    flag_exit = 1;
                    destroy_all_windows_cv();
                }

                while (custom_atomic_load_int(&run_detect_in_thread_fullHD)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);
                }

                while (custom_atomic_load_int(&run_fetch_in_thread_fullHD)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);
                }
                free_image(det_s);

                if (flag_exit == 1) break;

                release_mat(&show_img);
                show_img = det_img;

                det_img = in_img;
                det_s = in_s;
            }
            --delay;
        }
        printf("\ninput video stream closed. \n");
        fstream_close(INPUT);
        printf("input timestamps file closed\n");
        fstream_close(TIMESTAMPS);
        printf("output perspective detctions file closed. \n");
        fstream_close(OUTPUT);
        release_video_writer(&in_video);
        release_video_writer(&out_perspective_video);

        this_thread_sleep_for(thread_wait_ms);

        custom_join(detect_thread, 0);
        custom_join(fetch_thread, 0);

        // free memory
        free_image(in_s);
        free_detections(dets, nboxes);

        free(avg);
        for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
        demo_index = (NFRAMES + demo_index - 1) % NFRAMES;
        for (j = 0; j < NFRAMES; ++j) {
            release_mat(&cv_images[j]);
        }

        int i;
        const int nsize = 8;
        for (j = 0; j < nsize; ++j) {
            for (i = 32; i < 127; ++i) {
                free_image(alphabet[j][i]);
            }
            free(alphabet[j]);
        }
        free(alphabet);

        net_classes = net.layers[net.n - 1].classes;
        free_network(net);
        printf("Memory freed \n");

        deinit_perspective_params();
        return;
    }

}

void *fetch_in_thread_fullHD(void *ptr)
{
    
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_fetch_in_thread_fullHD)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }
        
        fstream_read(uyvy_frame, FRAME_SIZE_UYVY, INPUT);
        in_s = get_uyvy_image_from_stream_resize(uyvy_frame, net.w, net.h, net.c, &in_img);

        if (fstream_eof(INPUT)) {
            printf("Stream closed.\n");
            custom_atomic_store_int(&flag_exit, 1);
            custom_atomic_store_int(&run_fetch_in_thread_fullHD, 0);
            return 0;
        }

        custom_atomic_store_int(&run_fetch_in_thread_fullHD, 0);
    }
    return 0;
}

void *fetch_in_thread_sync_fullHD(void *ptr)
{
    custom_atomic_store_int(&run_fetch_in_thread_fullHD, 1);
    while (custom_atomic_load_int(&run_fetch_in_thread_fullHD)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

void *detect_in_thread_fullHD(void *ptr)
{
    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_detect_in_thread_fullHD)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

        layer l = net.layers[net.n - 1];
        float *X = det_s.data;
        float *prediction = network_predict(net, X);

        memcpy(predictions[demo_index], prediction, l.outputs * sizeof(float));
        mean_arrays(predictions, NFRAMES, l.outputs, avg);
        l.output = avg;

        cv_images[demo_index] = det_img;
        det_img = cv_images[(demo_index + NFRAMES / 2 + 1) % NFRAMES];
        demo_index = (demo_index + 1) % NFRAMES;

        
        dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

        custom_atomic_store_int(&run_detect_in_thread_fullHD, 0);
    }

    return 0;
}

void *detect_in_thread_sync_fullHD(void *ptr)
{
    custom_atomic_store_int(&run_detect_in_thread_fullHD, 1);
    while (custom_atomic_load_int(&run_detect_in_thread_fullHD)) this_thread_sleep_for(thread_wait_ms);
    return 0;
}

double get_wall_time_fullHD()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}
#endif