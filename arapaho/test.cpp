/*************************************************************************
 * arapaho                                                               *
 *                                                                       *
 * C++ API for Yolo v2                                                   *
 *                                                                       *
 * This test wrapper reads an image or video file and displays           *
 * detected regions in it.                                               *
 *                                                                       *
 * https://github.com/prabindh/darknet                                   *
 *                                                                       *
 * Forked from, https://github.com/pjreddie/darknet                      *
 *                                                                       *
 * Refer below file for build instructions                               *
 *                                                                       *
 * arapaho_readme.txt                                                    *
 *                                                                       *
 *************************************************************************/

#include "arapaho.hpp"
#include <string>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "PersonTrack.h"
#include "hungarian.h"
#include<climits>

// Use OpenCV for scaling the image (faster)
#define _ENABLE_OPENCV_SCALING

using namespace cv;

//
// Some configuration inputs
//
static char INPUT_DATA_FILE[]    = "input.data";
static char INPUT_CFG_FILE[]     = "input.cfg";
static char INPUT_WEIGHTS_FILE[] = "input.weights";
static char INPUT_AV_FILE[]      = "input.avi"; //"input.jpg"; //// Can take in either Video or Image file
#define MAX_OBJECTS_PER_FRAME (100)

#define TARGET_SHOW_FPS (10)

const int TRACK_INIT_TH = 100;

//
// Some utility functions
//
bool fileExists(const char *file)
{
    struct stat st;
    if(!file) return false;
    int result = stat(file, &st);
    return (0 == result);
}

//
// Main test wrapper for arapaho
//
int main()
{
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    box* boxes = 0;
    std::string* labels;
    vector<Rect> found, found_filtered;
    vector<PersonTrack> tracks;
    vector<vector<int>> costMatrix;
    double ticks = 0;


    // Early exits
    if(!fileExists(INPUT_DATA_FILE) || !fileExists(INPUT_CFG_FILE) || !fileExists(INPUT_WEIGHTS_FILE))
    {
        EPRINTF("Setup failed as input files do not exist or not readable!\n");
        return -1;
    }

    // Create arapaho
    ArapahoV2* p = new ArapahoV2();
    if(!p)
    {
        return -1;
    }

    // TODO - read from arapaho.cfg
    ArapahoV2Params ap;
    ap.datacfg = INPUT_DATA_FILE;
    ap.cfgfile = INPUT_CFG_FILE;
    ap.weightfile = INPUT_WEIGHTS_FILE;
    ap.nms = 0.4;
    ap.maxClasses = 2;

    // Always setup before detect
    ret = p->Setup(ap, expectedW, expectedH);
    if(false == ret)
    {
        EPRINTF("Setup failed!\n");
        if(p) delete p;
        p = 0;
        return -1;
    }

    // Steps below this, can be performed in a loop

    // loop
    // {
    //    setup arapahoImage;
    //    p->Detect(arapahoImage);
    //    p->GetBoxes;
    // }
    //

    // Setup image buffer here
    ArapahoV2ImageBuff arapahoImage;
    Mat image;

    // Setup show window
    namedWindow ( "Arapaho" , CV_WINDOW_AUTOSIZE );

    // open a video or image file
    VideoCapture cap ( INPUT_AV_FILE );
    if( ! cap.isOpened () )
    {
        EPRINTF("Could not load the AV file %s\n", INPUT_AV_FILE);
        if(p) delete p;
        p = 0;
        return -1;
    }
    // Detection loop
    while(1)
    {
      double precTick = ticks;
      ticks = (double) cv::getTickCount();

      double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds
        int imageWidthPixels = 0, imageHeightPixels = 0;
        bool success = cap.read(image);
        if(!success)
        {
            EPRINTF("cap.read failed/EoF - AV file %s\n", INPUT_AV_FILE);
            if(p) delete p;
            p = 0;
            waitKey();
            return -1;
        }
        if( image.empty() )
        {
            EPRINTF("image.empty error - AV file %s\n", INPUT_AV_FILE);
            if(p) delete p;
            p = 0;
            waitKey();
            return -1;
        }
        else
        {
            imageWidthPixels = image.size().width;
            imageHeightPixels = image.size().height;
            DPRINTF("Image data = %p, w = %d, h = %d\n", image.data, imageWidthPixels, imageHeightPixels);

            // Remember the time
            auto detectionStartTime = std::chrono::system_clock::now();

            // Process the image
            arapahoImage.bgr = image.data;
            arapahoImage.w = imageWidthPixels;
            arapahoImage.h = imageHeightPixels;
            arapahoImage.channels = 3;
            // Using expectedW/H, can optimise scaling using HW in platforms where available

            int numObjects = 0;

#ifdef _ENABLE_OPENCV_SCALING
            // Detect the objects in the image
            p->Detect(
                image,
                0.24,
                0.5,
                numObjects);
#else
            p->Detect(
                arapahoImage,
                0.24,
                0.5,
                numObjects);
#endif
            std::chrono::duration<double> detectionTime = (std::chrono::system_clock::now() - detectionStartTime);

            printf("==> Detected [%d] objects in [%f] seconds\n", numObjects, detectionTime.count());

            if(numObjects > 0 && numObjects < MAX_OBJECTS_PER_FRAME) // Realistic maximum
            {
                boxes = new box[numObjects];
                labels = new std::string[numObjects];
                if(!boxes)
                {
                    if(p) delete p;
                    p = 0;
                    return -1;
                }
                if(!labels)
                {
                    if(p) delete p;
                    p = 0;
                    if(boxes)
                    {
                        delete[] boxes;
                        boxes = NULL;
                    }
                    return -1;
                }

                // Get boxes and labels
                p->GetBoxes(
                    boxes,
                    labels,
                    numObjects
                    );

                int objId = 0;
                int leftTopX = 0, leftTopY = 0, rightBotX = 0,rightBotY = 0;

                // clear previous detections
                found.clear();
                found_filtered.clear();

                for (objId = 0; objId < numObjects; objId++)
                {
                    leftTopX = 1 + imageWidthPixels*(boxes[objId].x - boxes[objId].w / 2);
                    leftTopY = 1 + imageHeightPixels*(boxes[objId].y - boxes[objId].h / 2);
                    rightBotX = 1 + imageWidthPixels*(boxes[objId].x + boxes[objId].w / 2);
                    rightBotY = 1 + imageHeightPixels*(boxes[objId].y + boxes[objId].h / 2);
                    DPRINTF("Box #%d: center {x,y}, box {w,h} = [%f, %f, %f, %f]\n",
                            objId, boxes[objId].x, boxes[objId].y, boxes[objId].w, boxes[objId].h);
                    // Show image and overlay using OpenCV
                    // Show labels
                    if (labels[objId].c_str() && labels[objId]=="person")
                    {
                      // rectangle(image,
                      //     cvPoint(leftTopX, leftTopY),
                      //     cvPoint(rightBotX, rightBotY),
                      //     CV_RGB(255, 0, 0), 1, 8, 0);
                      //   DPRINTF("Label:%s\n\n", labels[objId].c_str());
                      //   putText(image, std::to_string(objId)  , cvPoint(leftTopX, leftTopY),
                            // FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
                        found.push_back(Rect(leftTopX,leftTopY,rightBotX-leftTopX,rightBotY-leftTopY));
                    }
                }

                size_t i, j;
                for (i=0; i<found.size(); i++)
                {
                    Rect r = found[i];
                    for (j=0; j<found.size(); j++)
                        if (j!=i && (r & found[j])==r)
                            break;
                    if (j==found.size())
                        found_filtered.push_back(r);
                }

                costMatrix.clear();

                // update tracks
                for(int j=0;j<tracks.size();j++)
                {
                    tracks[j].predict(dT);
                }

                for (i=0; i<found_filtered.size(); i++)
                {
                    Rect r = found_filtered[i];
                    r.x += cvRound(r.width*0.1);
                    r.width = cvRound(r.width*0.8);
                    r.y += cvRound(r.height*0.06);
                    r.height = cvRound(r.height*0.9);
                    found_filtered[i] = r;
                    rectangle(image, r.tl(), r.br(), cv::Scalar(0,255,0), 2);

                    if(tracks.size()==0) // Initializing Tracks for first detection
                    {
                        PersonTrack pt(r);
                        pt.should_keep(true);
                        pt.take_measure(r);
                        tracks.push_back(pt);
                    }
                    else // calculate the cost matrix
                    {
                        int min_d = INT_MAX;
                        vector<int> row(tracks.size());
                        row.clear();
                        for(int j=0;j<tracks.size();j++)
                        {
                            double d = sqrt(pow((double)(r.x-tracks[j].bbox.x),2.0) + pow((double)(r.y-tracks[j].bbox.y),2.0));
                            int di = cvRound(d);
                            if(di<min_d)
                                min_d = di;
                            row.push_back(di);
                        }
                        if(min_d>TRACK_INIT_TH) // initialize new track
                        {
                            PersonTrack pt(r);
                            pt.should_keep(true);
                            pt.take_measure(r);
                            tracks.push_back(pt);
                            double d = sqrt(pow((double)(r.x-pt.bbox.x),2.0) + pow((double)(r.y-pt.bbox.y),2.0));
                            int di = cvRound(d);
                            row.push_back(di);
                        }
                        costMatrix.push_back(row);
                    }
                }
                if(costMatrix.size()!=0)
                {
        //            cout << endl;
        //            for(int i=0;i<costMatrix.size();i++)
        //            {
        //                for(int j=0;j<costMatrix[0].size();j++)
        //                {
        //                    cout << costMatrix[i][j] << "\t";
        //                }
        //                cout << endl;
        //            }
        ////            cout << endl;
                    Hungarian hungarian(costMatrix,(int)found_filtered.size(),(int)tracks.size(),HUNGARIAN_MODE_MINIMIZE_COST);
                    hungarian.solve();
                    vector<vector<int>> assignment = hungarian.assignment();
                    cout << "cost: " << endl;
                    hungarian.print_cost();
                    cout << "assignment: " << endl;
                    hungarian.print_assignment();

                    for(int j=0;j<tracks.size();j++)
                    {
                        bool track_detected = false;
                        for(int i=0;i<found_filtered.size();i++)
                        {
                            if(assignment[i][j]==1)
                            {
                                tracks[j].take_measure(found_filtered[i]);
                                track_detected = true;
                                break;
                            }
                        }
                        if(!tracks[j].should_keep(track_detected)) // reject track based on rejection criteria
                        {
                            tracks.erase(tracks.begin()+j);
                            j--;
                        }
                    }

        //            for(vector<PersonTrack>::iterator i=tracks.begin();i<tracks.end();i++)
        //            {
        //                bool track_detected = false;
        //                for(vector<Rect>:: iterator j=found_filtered.begin();i<found_filtered.end();j++)
        //                {
        //
        //                }
        //            }
                }

                for(int j=0;j<tracks.size();j++)
                {
                    rectangle(image, tracks[j].bbox.tl(), tracks[j].bbox.br(), cv::Scalar(255,0,0), 2);
                }

                if (boxes)
                {
                    delete[] boxes;
                    boxes = NULL;
                }
                if (labels)
                {
                    delete[] labels;
                    labels = NULL;
                }

            }// If objects were detected
            imshow("Arapaho", image);
            waitKey((1000 / TARGET_SHOW_FPS));

        } //If a frame was read
    }// Detection loop

clean_exit:

    // Clear up things before exiting
    if(p) delete p;
    DPRINTF("Exiting...\n");
    return 0;
}
