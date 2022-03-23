#include <numeric>
#include "matching2D.hpp"
#include "dataStructures.h"
using namespace std;

inline const double secondsToMilliseconds(const double seconds) {
    return ((1000 * seconds) / 1.0);
}


void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType, DetectorInfo& matchInfo) {

    // configure matcher
    double t;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        constexpr bool crossCheck{ false };
        // const int normType{ ((descriptorFamily.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2) };
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        std::cout << "BF matching cross-check = " << crossCheck << std::endl;

    } else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descRef.type() != CV_32F) { descRef.convertTo(descRef, CV_32F); }
        if (descSource.type() != CV_32F) { descSource.convertTo(descSource, CV_32F); }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        std::cout << "FLANN matching" << std::endl;
    }

    if (selectorType.compare("SEL_NN") == 0) {
        t = static_cast<double>(cv::getTickCount());
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((static_cast<double>(cv::getTickCount())) - t) / cv::getTickFrequency();

    } else if (selectorType.compare("SEL_KNN") == 0) {
        std::vector<std::vector<cv::DMatch>> knnMatches;

        t = static_cast<double>(cv::getTickCount());
        matcher->knnMatch(descSource, descRef, knnMatches, 2);
        constexpr float threshold{ 0.8 };

        for (auto iterator{ std::begin(knnMatches) }; iterator != std::end(knnMatches); iterator++) {
            if ((*iterator).at(0).distance < (threshold * (*iterator).at(1).distance)) {
                matches.push_back((*iterator).at(0));
            }
        }

        t = ((static_cast<double>(cv::getTickCount())) - t) / cv::getTickFrequency();
    }
    matchInfo.nKeypoints =  matches.size();
    matchInfo.time =  secondsToMilliseconds(t);
    std::cout << "Matched Keypoints: " << matches.size() << "\n";
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, DetectorInfo& descInfo)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0) 
    {
        extractor = cv::ORB::create();

    } 
    else if (descriptorType.compare("FREAK") == 0) 
    {
        extractor = cv::xfeatures2d::FREAK::create();

    } 
    else if (descriptorType.compare("AKAZE") == 0) 
    {
        extractor = cv::AKAZE::create();

    } 
    // else if (descriptorType.compare("SIFT") == 0) 
    // {
    //     extractor = cv::xfeatures2d::SIFT::create();

    // } 
    else if (descriptorType.compare("BRIEF") == 0) 
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

    } 
    else
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    descInfo.time = secondsToMilliseconds(t);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, DetectorInfo &dectInfo)
{
    // DetectorInfo dectInfo;
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    dectInfo.nKeypoints = keypoints.size();
    dectInfo.time = secondsToMilliseconds(t);

    // return dectInfo;
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, DetectorInfo &dectInfo)
{
    // DetectorInfo dectInfo;
    // compute detector parameters based on image size
    int blockSize = 2;       
    int apertureSize = 3; 
    int scaledApertureSize = apertureSize * 2;
    int minResponse = 100;

    double k = 0.04;
    double overlapThreshold = 0.0;

    // Apply corner detection
    double t = (double)cv::getTickCount();
 
    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    bool foundOverlap{ false };

    for (int i{ 0 }; i < dstNorm.rows; i++) {
        for (int j{ 0 }; j < dstNorm.cols; j++) {

            const int response{ static_cast<int>(dstNorm.at<float>(i, j)) };

            if (response > minResponse) {
                cv::KeyPoint point;
                point.pt = cv::Point2f(i, j);
                point.size = scaledApertureSize;
                point.response = response;
                point.class_id = 0;

                foundOverlap = false;

                for (auto iterator{ std::begin(keypoints) }; iterator != std::end(keypoints); iterator++) {
                    if (cv::KeyPoint::overlap(point, (*iterator)) > overlapThreshold) {
                        foundOverlap = true;

                        if (point.response > (*iterator).response) {
                            *iterator = point;
                            break;
                        }
                    }
                }

                if (!foundOverlap) { keypoints.push_back(point); }
            }
        }

    }


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    dectInfo.nKeypoints = keypoints.size();
    dectInfo.time = secondsToMilliseconds(t);

    // return dectInfo;
}


void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis, DetectorInfo &dectInfo)
{
    // DetectorInfo dectInfo;
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0) {
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(30, true, type);

    } else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();

    } else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();

    } else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    }

    // else if (detectorType.compare("SIFT") == 0) {
    //     detector = cv::xfeatures2d::SIFT::create();



    // Apply corner detection
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);


    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType<<" detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    dectInfo.nKeypoints = keypoints.size();
    dectInfo.time = secondsToMilliseconds(t);

    // return dectInfo;
}