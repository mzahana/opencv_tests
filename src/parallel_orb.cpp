#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace std;

void detectFeatures(const Mat& img, vector<KeyPoint>& keypoints) {
    Ptr<FeatureDetector> orb = ORB::create(1000);
    orb->detect(img, keypoints);
    printf("I am Thread %d\n", omp_get_thread_num());
}

int main( int argc, char** argv ) {

    // Create a variable to store the number of threads to use
    int numThreads =4; //(int)cv::getNumberOfCPUs();

    // omp_set_num_threads(cv::getNumberOfCPUs());
    omp_set_num_threads(numThreads);

    // Load the image
    Mat src;
    src = imread( argv[1],  IMREAD_GRAYSCALE );

    Ptr<FeatureDetector> orb = ORB::create(500);


    // Resize
    // Get the new image dimensions
    int width = 640;
    int height = 480;

    // Create the destination image
    Mat dst;

    // Resize the image
    resize(src, dst, Size(width, height), 0, 0, INTER_LINEAR);


    // Create a vector to store the keypoints
    vector<KeyPoint> keypoints;
    vector<vector<KeyPoint>> vec_kps;
    vec_kps.resize(numThreads);

    // Create an array of threads
    // thread threads[numThreads];

    // auto start = std::chrono::high_resolution_clock::now();
    clock_t start = clock();
    // Split the image into numThreads horizontal strips
    #pragma omp parallel for
    for (int i = 0; i < numThreads; i++) {
        int startRow = i * dst.rows / numThreads;
        int endRow = (i + 1) * dst.rows / numThreads;
        Mat strip = dst(Range(startRow, endRow), Range::all());
        cout << "Rows: " << strip.rows << " Cols: " << strip.cols << "\n";
        orb->detect(strip, vec_kps[i]);
        printf("Keypoints for strip %d = %d \n", i, (int)vec_kps[i].size());
    }

    // Wait for all threads to complete
    // for (int i = 0; i < numThreads; i++) {
    //     threads[i].join();
    // }

    // std::cout << "Duration: " << duration.count() << " milliseconds" << std::endl;

    // concatenate all keypoints vectors
    for (int i = 0; i < numThreads; i++) {
        keypoints.insert(keypoints.end(), vec_kps[i].begin(), vec_kps[i].end());
    }
    cout << "Size of keypoints " << keypoints.size() << "\n";

    clock_t detectionTotalTime = double(clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
    cout << "Feature detection is done in " << detectionTotalTime << " miliseconds." << endl;

    // Draw the keypoints on the image
    Mat img_keypoints;
    drawKeypoints(dst, keypoints, img_keypoints);//, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    // Show the image with keypoints
    imshow("Keypoints", img_keypoints);

    waitKey(0);
    return(0);
}
