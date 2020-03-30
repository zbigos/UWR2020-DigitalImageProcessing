#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <inttypes.h>
#include <iostream>

using namespace cv;
using namespace std;

#include <math.h>


// This has been stolen from stackoverflow.
typedef struct RgbColor
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RgbColor;

typedef struct HsvColor
{
    unsigned char h;
    unsigned char s;
    unsigned char v;
} HsvColor;

typedef struct PixInfo
{
    unsigned char h;
    unsigned char s;
    unsigned char v;
    int16_t x, y;
    HsvColor crapoutpixinfo() { HsvColor o; o.h = h; o.s = s; o.v = v; return o; }
    void assign(HsvColor inject, int16_t _x, int16_t _y) {h = inject.h; s = inject.s; v = inject.v; x = _x; y = _y;}

    public: bool operator<(PixInfo a) { 
        if (a.v != v)
            return a.v > v;
        if (a.s != s)
            return a.s > s;
        return a.h > h;
    }
} PixInfo;


RgbColor HsvToRgb(HsvColor hsv)
{
    RgbColor rgb;
    unsigned char region, remainder, p, q, t;

    if (hsv.s == 0)
    {
        rgb.r = hsv.v;
        rgb.g = hsv.v;
        rgb.b = hsv.v;
        return rgb;
    }

    region = hsv.h / 43;
    remainder = (hsv.h - (region * 43)) * 6; 

    p = (hsv.v * (255 - hsv.s)) >> 8;
    q = (hsv.v * (255 - ((hsv.s * remainder) >> 8))) >> 8;
    t = (hsv.v * (255 - ((hsv.s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
        case 0:
            rgb.r = hsv.v; rgb.g = t; rgb.b = p;
            break;
        case 1:
            rgb.r = q; rgb.g = hsv.v; rgb.b = p;
            break;
        case 2:
            rgb.r = p; rgb.g = hsv.v; rgb.b = t;
            break;
        case 3:
            rgb.r = p; rgb.g = q; rgb.b = hsv.v;
            break;
        case 4:
            rgb.r = t; rgb.g = p; rgb.b = hsv.v;
            break;
        default:
            rgb.r = hsv.v; rgb.g = p; rgb.b = q;
            break;
    }

    return rgb;
}

HsvColor RgbToHsv(RgbColor rgb)
{
    HsvColor hsv;
    unsigned char rgbMin, rgbMax;

    rgbMin = rgb.r < rgb.g ? (rgb.r < rgb.b ? rgb.r : rgb.b) : (rgb.g < rgb.b ? rgb.g : rgb.b);
    rgbMax = rgb.r > rgb.g ? (rgb.r > rgb.b ? rgb.r : rgb.b) : (rgb.g > rgb.b ? rgb.g : rgb.b);

    hsv.v = rgbMax;
    if (hsv.v == 0)
    {
        hsv.h = 0;
        hsv.s = 0;
        return hsv;
    }

    hsv.s = 255 * long(rgbMax - rgbMin) / hsv.v;
    if (hsv.s == 0)
    {
        hsv.h = 0;
        return hsv;
    }

    if (rgbMax == rgb.r)
        hsv.h = 0 + 43 * (rgb.g - rgb.b) / (rgbMax - rgbMin);
    else if (rgbMax == rgb.g)
        hsv.h = 85 + 43 * (rgb.b - rgb.r) / (rgbMax - rgbMin);
    else
        hsv.h = 171 + 43 * (rgb.r - rgb.g) / (rgbMax - rgbMin);

    return hsv;
}
// end of stolen code, the rest is 100% mine


#define HISTVARTYPE int32_t
void calculate_histogram(HISTVARTYPE *histogram, Mat *image) {
    for (int i = 0; i < image->rows; i++)
        for(int j = 0; j < image->cols; j++) {
            // now we use the stolen code
            Vec3b source = image->at<Vec3b>(Point(i, j));
            RgbColor inject;
            /* counterintuitive, but opencv uses BGR */
            inject.r = source[2];
            inject.g = source[1];
            inject.b = source[0];
            HsvColor output = RgbToHsv(inject);
            histogram[output.v]++;
    }
}

void render_histogram(HISTVARTYPE *histogram, HISTVARTYPE *prefix, Mat *destination) {
    for (int i = 0; i < destination->rows; i++)
        for (int j = 0 ; j < destination->cols; j++) {
            Vec3b inject;
            inject[0] = 0; // b 
            inject[1] = ((destination->cols - histogram[i]) < j) ? 255 : 0; // g
            inject[2] = (((destination->cols - prefix[i]) > j - 6) && (destination->cols - prefix[i]) < j) ? 255 : 0; // g
            destination->at<Vec3b>(Point(i, j)) = inject; 
        }
}

void normalise_and_calculate_prefix(HISTVARTYPE *histogram, HISTVARTYPE *prefix, int32_t imsize, HISTVARTYPE *unconstrained) {
    HISTVARTYPE maximum = 0;
    for(int i = 0 ; i < imsize; i++)
        maximum = max(maximum, histogram[i]);
    // we get the velue for scaling the histogram

    int64_t suma = 0, cap = imsize;
    for(int i = 0 ; i < imsize; i++) {
        suma += histogram[i];
        prefix[i] = suma/cap;
        unconstrained[i] = suma;
    }

    for(int i = 0 ; i < imsize; i++)
        histogram[i] = (HISTVARTYPE)(512.0 * ((double)histogram[i]/(double)maximum));
}

void normalise_and_calculate_prefix(HISTVARTYPE *histogram, HISTVARTYPE *prefix, int32_t imsize) {
    HISTVARTYPE maximum = 0;
    for(int i = 0 ; i < imsize; i++)
        maximum = max(maximum, histogram[i]);
    // we get the velue for scaling the histogram

    int64_t suma = 0, cap = imsize;
    for(int i = 0 ; i < imsize; i++) {
        suma += histogram[i];
        prefix[i] = suma/cap;
    }

    for(int i = 0 ; i < imsize; i++)
        histogram[i] = (HISTVARTYPE)(512.0 * ((double)histogram[i]/(double)maximum));
}

void match_histograms(Mat *manipulated_image, HISTVARTYPE *source_histogram, HISTVARTYPE *source_histogram_prefix, HISTVARTYPE *target_histogram, HISTVARTYPE *target_histogram_prefix, HISTVARTYPE *result_histogram) {
    // extract all pixels from image and sort them by v.
    PixInfo *pixels = (PixInfo *)calloc(manipulated_image->rows * manipulated_image->cols, sizeof(PixInfo));
    int64_t iter = 0;
    for(int i = 0 ; i < manipulated_image->rows; i++)
        for(int j = 0 ; j < manipulated_image->cols; j++) {
            Vec3b source = manipulated_image->at<Vec3b>(Point(i, j));
            RgbColor inject;
            /* counterintuitive, but opencv uses BGR */
            inject.r = source[2];
            inject.g = source[1];
            inject.b = source[0];
            PixInfo output;
            output.assign(RgbToHsv(inject), i, j);
            pixels[iter] = output; 
            iter ++;
        }

    sort(pixels, pixels + (manipulated_image->rows * manipulated_image->cols));

    int32_t expendable = 0, allocated_space = 0;
    int32_t greedy_pointer = 0, greedy_fseek = 0;
    int32_t allocation_offset = 0;
    while(1) {
        if (allocated_space > expendable && allocated_space != 0) { // we have more allocated histogram space, we can spend it.
                while(expendable == 0) { // nothing to allocate, take some from the source image
                    int8_t currently_watching = pixels[greedy_pointer].v;
                    greedy_fseek = 0;
                    while(pixels[greedy_pointer + greedy_fseek].v == currently_watching) {
                        greedy_fseek++;
                        expendable ++;
                    }

                }

            // now we *must* have something to allocate
            for(int i = greedy_pointer; i < (greedy_pointer + greedy_fseek); i++) {
                pixels[i].v = allocation_offset;
                expendable ++;
            }

            greedy_pointer += greedy_fseek;
        } else { // we don't have allocated histogram space, we want to allocate more.
            allocated_space = target_histogram_prefix[allocation_offset];
            allocation_offset ++; // we declare that we will be writing to the right, but we can allocate more;
        }
    
        if (allocated_space > 510*510)
            break;
    }


    for(int i = 0 ; i < manipulated_image->rows; i++)
        for(int j = 0 ; j < manipulated_image->cols; j++) {
            Vec3b inject;
            inject[0] = 0; // b 
            inject[1] = 0; // g
            inject[2] = 0; // r
            manipulated_image->at<Vec3b>(Point(i, j)) = inject;                 
        }
    /* now we just stupidly overwrite the histogram */
    for(int i = 0 ; i < manipulated_image->rows * manipulated_image->cols; i++) {
        RgbColor o = HsvToRgb(pixels[i].crapoutpixinfo());
        Vec3b inject;
        inject[0] = o.b; // b 
        inject[1] = o.g; // g
        inject[2] = o.r; // r
        manipulated_image->at<Vec3b>(Point(pixels[i].x, pixels[i].y)) = inject;     
    }
    //    result_histogram[pixels[i].v]++;
    // greedily apply them to the image

    free(pixels);
}

void DitherImage(Mat *Input, Mat *Output) {
    for (int i = 0; i < Input->rows; i++)
        for(int j = 0 ; j < Input->cols; j++) {
            Vec3b inject;
            inject = Input->at<Vec3b>(Point(i, j));
            for(int k = 0 ; k < 3; k++)
                inject[k] ^= rand() % 8;
            Output->at<Vec3b>(Point(i, j)) = inject;                   
        }
}

int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout << " Usage: <image to modify> <image with source histogram>  " << endl;
     return -1;
    }

    Mat source = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat modifier = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if(!source.data || !modifier.data) {
        cout <<  "Could not open or find the image" << endl;
        exit(1);
    }

    if(source.rows * source.cols != modifier.rows * modifier.cols) {
        cout << "Images are of different size" << endl;
        exit(1);
    }

    HISTVARTYPE *histogram1 = (HISTVARTYPE *)calloc(source.cols, sizeof(HISTVARTYPE));
    HISTVARTYPE *histogram1psum = (HISTVARTYPE *)calloc(source.cols, sizeof(HISTVARTYPE));
    calculate_histogram(histogram1, &source);

    HISTVARTYPE *histogram2 = (HISTVARTYPE *)calloc(modifier.cols, sizeof(HISTVARTYPE));
    HISTVARTYPE *histogram2psum = (HISTVARTYPE *)calloc(modifier.cols, sizeof(HISTVARTYPE));
    HISTVARTYPE *histogram2psum_unconstrained = (HISTVARTYPE *)calloc(source.cols, sizeof(HISTVARTYPE));
    calculate_histogram(histogram2, &modifier);

    normalise_and_calculate_prefix(histogram1, histogram1psum, source.cols);
    normalise_and_calculate_prefix(histogram2, histogram2psum, modifier.cols, histogram2psum_unconstrained);

    Mat histogram1_render = source.clone();
    Mat histogram2_render = source.clone();
        /* not the best practice, but we make sure that we have mats of the exact same type */
    
    render_histogram(histogram1, histogram1psum, &histogram1_render);
    render_histogram(histogram2, histogram2psum, &histogram2_render);


    Mat result = source.clone();
    HISTVARTYPE *histogram3 = (HISTVARTYPE *)calloc(result.cols, sizeof(HISTVARTYPE));
    HISTVARTYPE *histogram3psum = (HISTVARTYPE *)calloc(result.cols, sizeof(HISTVARTYPE));
    Mat histogram3_render = source.clone();
    match_histograms(&result, histogram1, histogram1psum, histogram2, histogram2psum_unconstrained, histogram3);
    calculate_histogram(histogram3, &result);
    normalise_and_calculate_prefix(histogram3, histogram3psum, result.cols);
    render_histogram(histogram3, histogram3psum, &histogram3_render);

    Mat dithered = source.clone();
    Mat histogram4_render = source.clone();
    DitherImage(&result, &dithered);
    HISTVARTYPE *histogram4 = (HISTVARTYPE *)calloc(result.cols, sizeof(HISTVARTYPE));
    HISTVARTYPE *histogram4psum = (HISTVARTYPE *)calloc(result.cols, sizeof(HISTVARTYPE));
    calculate_histogram(histogram4, &result);
    normalise_and_calculate_prefix(histogram4, histogram4psum, source.cols);
    render_histogram(histogram4, histogram4psum, &histogram4_render);


    Mat H1, H2, H3, H4, HP1, HP2, O, Os;
    vconcat(source, histogram1_render, H1);
    vconcat(modifier, histogram2_render, H2);
    hconcat(H1, H2, HP1);
    vconcat(result, histogram3_render, H3);
    vconcat(dithered, histogram4_render, H4);
    hconcat(H3, H4, HP2);

    hconcat(HP1, HP2, O);

    namedWindow( "Histogram Matching", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Histogram Matching", O );                   // Show our image inside it.
    imwrite("output.jpeg", O);

    free(histogram1);
    free(histogram2);
    free(histogram3);

    free(histogram1psum);
    free(histogram2psum);
    free(histogram3psum);

    free(histogram2psum_unconstrained);


    while(1)
        waitKey(20);                                          // Wait for a keystroke in the window
    return 0;
}
