#include <Windows.h>
#include <Ole2.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <io.h>
#include <direct.h>

#include <Kinect.h>

#define DepWidth 512
#define DepHeight 424

// Kinect vairables
IKinectSensor* sensor;               // Kinect sensor
IMultiSourceFrameReader* reader;     // Kinect data source
ICoordinateMapper* mapper;           // Converts between depth, color, and 3d coordinates
SYSTEMTIME st;
UINT16 Depdata[DepWidth * DepHeight * 3];  // BGRA array containing the texture data

bool initKinect() {
    if (FAILED(GetDefaultKinectSensor(&sensor))) {
        return false;
    }
    if (sensor) {
        sensor->get_CoordinateMapper(&mapper);

        sensor->Open();
        sensor->OpenMultiSourceFrameReader(
            FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
            &reader);
        return reader;
    }
    else {
        return false;
    }
}

void getDepthData(IMultiSourceFrame* frame) {
    IDepthFrame* depthframe;
    IDepthFrameReference* frameref = NULL;
    frame->get_DepthFrameReference(&frameref);
    frameref->AcquireFrame(&depthframe);
    if (frameref) frameref->Release();
    if (!depthframe) return;

    depthframe->CopyFrameDataToArray(DepWidth * DepHeight, Depdata);

    if (depthframe) depthframe->Release();
}

void writeData() {
    //std::cout << "start writing data!" << std::endl;
    FILE* pFile1;
    if ((pFile1 = fopen("./data/realtime/Depdata.txt", "wb")) == NULL) {
        printf("cant open the file");
        exit(0);
    }
    fwrite(Depdata, sizeof(UINT16), DepWidth * DepHeight, pFile1);
    fclose(pFile1);
}

void getKinectData() {
    IMultiSourceFrame* frame = NULL;
    if (SUCCEEDED(reader->AcquireLatestFrame(&frame))) {
        GetLocalTime(&st);
        std::cout << st.wYear << "." << st.wMonth << "." << st.wDay << "."
            << st.wHour << "." << st.wMinute << "." << st.wSecond << "." << st.wMilliseconds << std::endl;
        getDepthData(frame);
        writeData();
    }
    if (frame) frame->Release();
}


int main() {
    if (!initKinect()) return 1;
    while (1) {
        getKinectData();
        std::ofstream myfile1("./data/realtime/timestamp.txt");
        if (myfile1.is_open()) {
            myfile1 << st.wMilliseconds << "\n";
        }
        myfile1.close();
    }
    return 0;
}