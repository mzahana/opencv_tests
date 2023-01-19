#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
    omp_set_num_threads(cv::getNumberOfCPUs());

    #pragma omp parallel for
    for (int i = 0; i < 8; i++)
    {
        printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
    }
    

    Mat src;

    // Load the image
    src = imread( argv[1],  IMREAD_GRAYSCALE );


    // Resize
    // Get the new image dimensions
    int width = 640;
    int height = 480;

    // Create the destination image
    Mat resized_img;

    // Resize the image
    resize(src, resized_img, Size(width, height), 0, 0, INTER_LINEAR);


    // auto start = std::chrono::high_resolution_clock::now();
    clock_t start = clock();

    // Create a vector to store the keypoints
    vector<KeyPoint> keypoints;
    Mat descriptor;
    Ptr<FeatureDetector> orb = ORB::create(500);

    // Extract ORB features
    orb->detect(resized_img, keypoints);

    clock_t detectionTotalTime = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    cout << "Feature detection is done in " << detectionTotalTime << " miliseconds." << endl;


    cout << "Size of keypoints " << keypoints.size() << "\n";

    // Draw the keypoints on the image
    Mat img_keypoints;
    drawKeypoints(resized_img, keypoints, img_keypoints);//, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // Show the image with keypoints
    imshow("Keypoints", img_keypoints);

    waitKey(0);
    return(0);
}
