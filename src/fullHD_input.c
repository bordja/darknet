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

#define TRESH (0.6F)
#define NFRAMES 3
#define WIDTH 1920
#define HEIGHT 1080
#define FRAME_SIZE_UYVY (WIDTH * HEIGHT * 2)

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
static image in_s;
static image det_s;

static int nboxes = 0;
static detection *dets = NULL;

static float *avg;
static volatile int flag_exit;
static long long int frame_id = 0;
int frame_skip = 40;
mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;
mat_cv* perspective_img;

point_cv car_perspective_detections[300];
point_cv person_perspective_detections[300];
//std::ifstream yuv_stream;

static const int thread_wait_ms = 1;
static volatile int run_fetch_in_thread_fullHD = 0;
static volatile int run_detect_in_thread_fullHD = 0;

void run_fullHD(char *cfgfile, char *weightfile, float thresh, const char *filename, char **names, int classes);
void *fetch_in_thread_fullHD(void *ptr);
void *fetch_in_thread_sync_fullHD(void *ptr);
void *detect_in_thread_fullHD(void *ptr);
void *detect_in_thread_sync_fullHD(void *ptr);
double get_wall_time_fullHD();

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
    //const char* filename = "E:/darknet/data/frames0.yuv";
    const char* filename = "/home/rtrk/Desktop/Faculty/Master-rad/01-frame-grabber/camera-inputs/stalak-2020-05-21/SSD-1/Out2.yuv";

    run_fullHD(cfg, weights, TRESH, filename, names, classes);
}

void run_fullHD(char *cfgfile, char *weightfile, float thresh, const char *filename, char **names, int classes) {

    in_img = det_img = show_img = perspective_img = NULL;

    image **alphabet = load_alphabet();

    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    int delay = frame_skip;
    printf("Full HD\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    if (tcp_init_client() != 0)
    {
        printf("Connect client failed.\n");
        return;
    }

    int numbers = 0;
    while (true)
    {
        printf("SOCK_WRITE: %d\n", ++numbers);
        write(sockfd, &numbers, sizeof(numbers));
        sleep(1);
    }

    //yuv_stream.open(filename, std::ios_base::binary);
    fstream_open(filename);
    
    if (fstream_is_open())
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

        fetch_in_thread_sync_fullHD(0); //fetch_in_thread(0);
        det_img = in_img;
        det_s = in_s;

        fetch_in_thread_sync_fullHD(0); //fetch_in_thread(0);
        detect_in_thread_sync_fullHD(0); //fetch_in_thread(0);
        det_img = in_img;
        det_s = in_s;

        for (j = 0; j < NFRAMES / 2; ++j) {
            free_detections(dets, nboxes);
            fetch_in_thread_sync_fullHD(0); //fetch_in_thread(0);
            detect_in_thread_sync_fullHD(0); //fetch_in_thread(0);
            det_img = in_img;
            det_s = in_s;
        }

        int count = 0;
        int full_screen = 0;
        create_window_cv("FullHD", full_screen, WIDTH, HEIGHT);

        float avg_fps = 0;
        bool finished_clicking = false;
        bool window_created = false;

        while (!fstream_eof()) {
            ++count;
            if (count % frame_skip == 1)
            {
                const float nms = .45;    // 0.4F
                int local_nboxes = nboxes;
                detection *local_dets = dets;
                this_thread_yield();

                custom_atomic_store_int(&run_fetch_in_thread_fullHD, 1); // if (custom_create_thread(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
                custom_atomic_store_int(&run_detect_in_thread_fullHD, 1); // if (custom_create_thread(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

                //if (nms) do_nms_obj(local_dets, local_nboxes, l.classes, nms);    // bad results
                if (nms) {
                    if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                    else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
                }

                ++frame_id;
                if (det_img != NULL)
                    cv_copy_to_input_perspective((void*)det_img);
                draw_detection_and_point(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);

                if (!finished_clicking && show_img != NULL)
                    finished_clicking = mouse_click_and_param_init((void*)show_img, "FullHD");
                else if (show_img != NULL)
                {

                    show_image_mat(show_img, "FullHD");
                    perspective_img = show_img;
                    if (!window_created)
                    {
                        create_window_cv("Perspective transform", full_screen, WIDTH, HEIGHT);
                        window_created = true;
                    }
                    get_perspective_transform();
                    cv_Color color = PURPLE;
                    for (int dets = 0; dets < num_cars; dets++)
                    {
                        pixel_perspective_transform(car_detections[dets].x, car_detections[dets].y,
                            &car_perspective_detections[dets].x, &car_perspective_detections[dets].y, color);
                    }
                    color = LIGHT_BLUE;
                    for (int dets = 0; dets < num_persons; dets++)
                    {
                        pixel_perspective_transform(person_detections[dets].x, person_detections[dets].y,
                            &person_perspective_detections[dets].x, &person_perspective_detections[dets].y, color);
                    }
                    cv_copy_from_output_perspective((void*)perspective_img);
                    show_image_mat(perspective_img, "Perspective transform");
                }
                for (int dets = 0; dets < num_cars; dets++)
                {
                    car_detections[dets].x = 0;
                    car_detections[dets].y = 0;
                }
                for (int dets = 0; dets < num_persons; dets++)
                {
                    person_detections[dets].x = 0;
                    person_detections[dets].y = 0;
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
                }


                while (custom_atomic_load_int(&run_detect_in_thread_fullHD)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);   // custom_join(detect_thread, 0);
                }

                while (custom_atomic_load_int(&run_fetch_in_thread_fullHD)) {
                    if (avg_fps > 180) this_thread_yield();
                    else this_thread_sleep_for(thread_wait_ms);   // custom_join(fetch_thread, 0);
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
        printf("input video stream closed. \n");

        this_thread_sleep_for(thread_wait_ms);

        custom_join(detect_thread, 0);
        custom_join(fetch_thread, 0);

        // free memory
        free_image(in_s);
        free_detections(dets, nboxes);
        tcp_deinit_client();

        free(avg);
        for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
        demo_index = (NFRAMES + demo_index - 1) % NFRAMES;
        for (j = 0; j < NFRAMES; ++j) {
            release_mat(&cv_images[j]);
        }

        free_ptrs((void **)names, net.layers[net.n - 1].classes);

        int i;
        const int nsize = 8;
        for (j = 0; j < nsize; ++j) {
            for (i = 32; i < 127; ++i) {
                free_image(alphabet[j][i]);
            }
            free(alphabet[j]);
        }
        free(alphabet);
        free_network(net);
        //cudaProfilerStop();
        printf("Memory freed \n");
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
        
        fstream_read(uyvy_frame, FRAME_SIZE_UYVY);
        //cv::Mat yuv_frame(HEIGHT, WIDTH, CV_8UC2, uyvy_frame);
        //cv::Mat bgr_frame(HEIGHT, WIDTH, CV_8UC3);
        
        in_s = get_uyvy_image_from_stream_resize(uyvy_frame, net.w, net.h, net.c, &in_img);

        if (fstream_eof()) {
            printf("Stream closed.\n");
            custom_atomic_store_int(&flag_exit, 1);
            custom_atomic_store_int(&run_fetch_in_thread_fullHD, 0);
            //exit(EXIT_FAILURE);
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