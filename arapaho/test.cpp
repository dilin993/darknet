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
//#include "PersonTrack.h"
//#include "hungarian.h"
#include "udp_client.h"
#include<climits>
#include<ctime>
#include "pugixml.hpp"

// Use OpenCV for scaling the image (faster)
#define _ENABLE_OPENCV_SCALING

using namespace cv;
using namespace std;

//
// Some configuration inputs
//
char *INPUT_DATA_FILE    = nullptr;//"input.data";
char *INPUT_CFG_FILE   = nullptr;//"input.cfg";
char *INPUT_WEIGHTS_FILE = nullptr;//"input.weights";
char *INPUT_AV_FILE     = nullptr;//"input.avi"; //"input.jpg"; //// Can take in either Video or Image file
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


void copyToCstr(char * &cstr,string str)
{
  delete cstr;
  cstr = new char [str.length()+1];
  strcpy (cstr, str.c_str());
  cout << cstr << endl;
}

//
// Main test wrapper for arapaho
//
int main(int argc, const char * argv[])
{
    bool ret = false;
    int expectedW = 0, expectedH = 0;
    box* boxes = 0;
    std::string* labels;
    vector<Rect> found, found_filtered;
    uint16_t frameCount=0;

    string ip="127.0.0.1";
    int port=8089;

	if (argc > 1)
  {
		pugi::xml_document doc;
		pugi::xml_parse_result result = doc.load_file(argv[1]);
		if (result)
		{
			pugi::xml_node detectorConfig = doc.child("configuration").child("detector");
			ip = detectorConfig.attribute("ip").as_string();
			port = detectorConfig.attribute("port").as_int();

			pugi::xml_node inputConfig = doc.child("configuration").child("input");
      copyToCstr(INPUT_DATA_FILE,inputConfig.attribute("data").as_string());
      cout << "INPUT_DATA_FILE: " << INPUT_DATA_FILE << endl;
      copyToCstr(INPUT_CFG_FILE,inputConfig.attribute("cfg").as_string());
      //cout << "INPUT_CFG_FILE: " << INPUT_CFG_FILE << endl;
      copyToCstr(INPUT_WEIGHTS_FILE,inputConfig.attribute("weights").as_string());
      //cout << "IINPUT_WEIGHTS_FILE: " << INPUT_WEIGHTS_FILE << endl;
      copyToCstr(INPUT_AV_FILE,inputConfig.attribute("video").as_string());
      //cout << "INPUT_AV_FILE: " << INPUT_AV_FILE << endl;
		}

	}
  else
  {
    cout << "darknet: Please specify configuration file." << endl;
    cout << "\tusage: ./darknet <configuration file>" << endl;
    return -1;
  }

//    vector<PersonTrack> tracks;
//    vector<vector<int>> costMatrix;
//    double ticks = 0;
    udp_client::udp_client client1(ip.c_str(),port);
    cout << "connected to " << ip << ":" << port << endl << endl;


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
        	// frame variables
			string frameData = "";
			time_t t = time(0);
			struct tm *now = localtime(&t);
			char buffer[80];
			strftime(buffer, sizeof(buffer), "%d-%m-%Y %I:%M:%S", now);
			string timeStr(buffer);

			if (frameCount >= UINT16_MAX)
				frameCount = 0;

			// frame header
			frameData += "frame" + to_string(frameCount) + ";";
			frameData += timeStr + ";";
			frameCount++;



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



                for (i=0; i<found_filtered.size(); i++)
                {
                    Rect r = found_filtered[i];
//                    r.x += cvRound(r.width*0.1);
//                    r.width = cvRound(r.width*0.8);
//                    r.y += cvRound(r.height*0.06);
//                    r.height = cvRound(r.height*0.9);
//                    found_filtered[i] = r;
                    rectangle(image, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
                    string pos="";
                    pos += to_string(r.x) + "," +
                    	   to_string(r.y) + "," +
						   to_string(r.width) + "," +
						   to_string(r.height);
                    frameData += pos + ";";
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

            // transmit current frame data
            const char *cstr = frameData.c_str();
			cout << cstr << endl;
			client1.send(cstr, frameData.length());

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
