#include <iostream>
#include <opencv2/opencv.hpp>
#include "PersonTrack.h"
#include "hungarian.h"
#include<climits>

using namespace std;
using namespace cv;

const int TRACK_INIT_TH = 100;

int main (int argc, const char * argv[])
{
    VideoCapture cap;
    if(argc<2)
    {
        cout << "Using webcam for input." << endl;
        cap = VideoCapture(CV_CAP_ANY);
    }
    else
    {
        cout << "Using file: " << argv[1] << " for input." << endl;
        cap = VideoCapture(argv[1]);
    }

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    if (!cap.isOpened())
        return -1;

    Mat img;
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    double ticks = 0;
    namedWindow("video capture", CV_WINDOW_AUTOSIZE);

    vector<PersonTrack> tracks;
    vector<vector<int>> costMatrix;
    while (true)
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        cap >> img;
        if (!img.data)
            continue;

        vector<Rect> found, found_filtered;
        hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);

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
            rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);

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
            rectangle(img, tracks[j].bbox.tl(), tracks[j].bbox.br(), cv::Scalar(255,0,0), 2);
        }
        imshow("video capture", img);
        if (waitKey(20) >= 0)
            break;
    }
    return 0;
}