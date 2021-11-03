#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::string;

vector<string> row;

int WNDX = 32;
int WNDY = 32;

/*
Dataset paths
*/
string datasetTrainPath("banana-detection/bananas_train/images/");
string datasetTestPath("banana-detection/bananas_test/images/");
string datasetFile("banana-detection/bananas_train/label2.txt");
string patchDataset("patch_dataset/");
string patchData("patchesData.csv");
string overlapFile("overlap.csv");
string xTrainData("FinalDataset/images.csv");
string yTrainData("FinalDataset/labels.csv");

struct Patch
{
    int label;
    std::string patchName;
};

string getFileData(const string &path)
{
    auto stream = ostringstream{};
    ifstream file(path);
    stream << file.rdbuf();
    return stream.str();
}

// get labels and data from raw training dataset
std::map<int, std::vector<string>> readDatasetData()
{
    string contents;
    std::map<int, std::vector<string>> fileData;
    char delimiter = ',';

    contents = getFileData(datasetFile);

    istringstream stream(contents);
    std::vector<string> items;
    string record, filename2, label, xmin, ymin, xmax, ymax;

    int counter = 0;
    // read file line by line
    while (std::getline(stream, record))
    {
        istringstream line(record);
        // decomposition each line according to delimiter
        while (std::getline(line, filename2, delimiter))
        {
            items.push_back(filename2);

            std::getline(line, label, delimiter);
            items.push_back(label);

            std::getline(line, xmin, delimiter);
            items.push_back(xmin);

            std::getline(line, ymin, delimiter);
            items.push_back(ymin);

            std::getline(line, xmax, delimiter);
            items.push_back(xmax);

            std::getline(line, ymax);
            items.push_back(ymax);
        }

        fileData[counter] = items;
        items.clear();
        counter += 1;
    }
    return fileData;
}

/*
	This function is used for specifying label of each patch.
    - bnArea: Area of banana zone
    - rgArea: Area of patch
    - label 2 means that the patch can not be considered as 
      "Has Banana" or "Has not Banana"
*/
int calculateOverlap(int bnArea, int lenX, int lenY)
{
    int rgArea = lenX * lenY;
    if (rgArea == 0)
    {
        return 0;
    }
    float overlap = (float)rgArea / (float)bnArea;
    if (overlap >= 0.25)
    {
        return 1;
    }
    else if (overlap > 0.05 && overlap < 0.25) //to change
    {
        return 2;
    }
    else
    {
        return 0;
    }
}

/*
    There are 9 case that two rectangle can intersect, this function analyze these 9 case and if 
    banana zone and patch zone have intersection then execute calculateOverlap() function to 
    specify the label of patch based on amount of common area between its area and banana zone area
*/
int specifyLabel(string imgName, string saveName, int bnXmin, int bnYmin, int bnXmax, int bnYmax,
                 int rgXmin, int rgYmin, int rgXmax, int rgYmax) // argument change
{
    int bnArea = (bnXmax - bnXmin) * (bnYmax - bnYmin);
    ofstream overlap;
    overlap.open(overlapFile, std::ios_base::app);
    if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
         rgXmax >= bnXmin && rgXmax <= bnXmax &&
         rgYmax >= bnYmin && rgYmax <= bnYmax &&
         rgYmin >= bnYmin && rgYmin <= bnYmax))
    {
        int lenX = abs(rgXmin - rgXmax);
        int lenY = abs(rgYmax - rgYmin);
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type1"
                << "," << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY); //true
    }
    else if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
              rgXmax >= bnXmin && rgXmax <= bnXmax &&
              rgYmax >= bnYmin && rgYmax <= bnYmax &&
              rgYmin <= bnYmin))
    {
        int lenX = abs(rgXmin - rgXmax);
        int lenY = abs(rgYmax - bnYmin);
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type2,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY); //true
    }
    else if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
              rgXmax >= bnXmin && rgXmax <= bnXmax &&
              rgYmin >= bnYmin && rgYmin <= bnYmax &&
              rgYmax >= bnYmax))
    {
        int lenX = abs(rgXmin - rgXmax);
        int lenY = abs(rgYmin - bnYmax);
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type3,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY); //true
    }
    else if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
              rgXmax >= bnXmax &&
              rgYmin >= bnYmin && rgYmin <= bnYmax &&
              rgYmax >= bnYmin && rgYmax <= bnYmax))
    {
        int lenX = abs(rgXmin - bnXmax);
        int lenY = abs(rgYmax - rgYmin);
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type4,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY); //true
    }
    else if ((rgXmax >= bnXmin && rgXmax <= bnXmax &&
              rgXmin <= bnXmin &&
              rgYmin >= bnYmin && rgYmin <= bnYmax &&
              rgYmax >= bnYmin && rgYmax <= bnYmax))
    {
        int lenX = abs(rgXmax - bnXmin);
        int lenY = abs(rgYmax - rgYmin);
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type5,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY); //true
    }
    else if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
              rgYmin >= bnYmin && rgYmin <= bnYmax))
    {
        int lenX = abs(rgXmin - bnXmax);
        int lenY = abs(rgYmin - bnYmax); //true
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type7,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY);
    }
    else if ((rgXmin >= bnXmin && rgXmin <= bnXmax &&
              rgYmax >= bnYmin && rgYmax <= bnYmax))
    {
        int lenX = abs(rgXmin - bnXmax);
        int lenY = abs(rgYmax - bnYmin); //true
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type8,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY);
    }
    else if ((rgXmax >= bnXmin && rgXmax <= bnXmax &&
              rgYmax >= bnYmin && rgYmax <= bnYmax))
    {
        int lenX = abs(rgXmax - bnXmin);
        int lenY = abs(rgYmax - bnYmin); //true
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type9,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY);
    }
    else if ((rgXmax >= bnXmin && rgXmax <= bnXmax &&
              rgYmin >= bnYmin && rgYmin <= bnYmax))
    {
        int lenX = abs(rgXmax - bnXmin);
        int lenY = abs(rgYmin - bnYmax); //true
        overlap << imgName << "," << bnArea << "," << to_string(lenX * lenY) << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type10,"
                << saveName << "\n";
        overlap.close();
        return calculateOverlap(bnArea, lenX, lenY);
    }
    else
    {
        overlap << imgName << "," << bnArea << ","
                << "0"
                << "," << bnXmin << "," << bnYmin << "," << bnXmax << "," << bnYmax << "," << rgXmin << "," << rgYmin << "," << rgXmax << "," << rgYmax << ",type12,"
                << saveName << "\n";
        overlap.close();
        return 0;
    }
}

/*
    Convert each 256*256 pixels image to a group of 32*32 pixels patches
    and specify each patch label according to its overlapping with banana zone
*/
void createPatchesDataset(std::map<int, std::vector<string>> rawdatasetData)
{
    std::vector<string> rowData;
    Mat src;
    std::vector<Patch> patchImgs;

    int xmin = 0, ymin = 0;

    int xmax = 0, ymax = 0;

    int imgRow = 0, imgCol = 0;

    int label = 0;

    int counter = 0;

    ofstream patchFile;
    patchFile.open(patchData); // Create file

    ofstream outputFile("datasetf.csv");

    patchFile << "image"
              << ","
              << "label" << endl; // Headers
    string imgName;
    Patch patch;
    for (int name = 0; name < 1000; name++)
    {
        src = imread(datasetTrainPath + to_string(name) + ".png");
        imgRow = src.rows;
        imgCol = src.cols;
        rowData = rawdatasetData[name + 1]; // +1 for syncing the indices
        //Get banana zone coordinates
        xmin = stoi(rowData[2]);
        ymin = stoi(rowData[3]);
        xmax = stoi(rowData[4]);
        ymax = stoi(rowData[5]);
        /*
            For each of 256*256 training images this procedure is performed to 
            divide them into 32*32 patches with step 16. Also, if the patch has 
            banana the label will be 1.
            - (xmin,ymin,xmax,ymax) are the coordinates of banana zone
            - (x,y,x+32,y+32) are the coordinates of patch zone
            - specifyLabel() function specifies the label of patch based on
            area of intersection of banana and patch zone
            
        */
        cout << "image " << name << ".png reading..." << endl;
        for (int x = 0; x <= imgRow - WNDX; x += 16)
        {
            for (int y = 0; y <= imgCol - WNDY; y += 16)
            {
                string patchName = to_string(counter) + ".png";
                label = specifyLabel(to_string(name) + ".png", patchName,
                                     xmin, ymin, xmax, ymax, x, y, x + WNDX, y + WNDY);
                // crop patch from original image
                Rect patch(x, y, WNDX, WNDY);
                Mat croppedPatch = src(patch);
                // store each patch in new directory
                imwrite(patchDataset + patchName, croppedPatch);
                // store each patch with its label in file
                patchFile << patchName << "," << to_string(label) << "\n";
                counter++;
            }
        }
    }
    outputFile.close();
    patchFile.close(); // Close the file
}

/*
    Get information of the patches data file, created by createPatchesDataset() function
*/
std::map<int, std::vector<string>> getPatchesData()
{
    string filename(patchData);
    string contents;
    std::map<int, std::vector<string>> fileData;
    char delimiter = ',';

    contents = getFileData(filename);

    istringstream stream(contents);
    std::vector<string> items;
    string record, imgName, label;

    int counter = 0;
    // read file line by line
    while (std::getline(stream, record))
    {
        istringstream line(record);
        // decomposition each line according to delimiter
        while (std::getline(line, imgName, delimiter))
        {
            items.push_back(imgName);

            std::getline(line, label, delimiter);
            items.push_back(label);
        }
        fileData[counter] = items;
        items.clear();
        counter += 1;
    }
    return fileData;
}

/*
    - select patches with label 1 or 0 randomly and convert them to csv file
    - output of this function directly used as input of CNN
*/
void selectTrainingData(int preprocessType, std::map<int, std::vector<string>> patchImages, int count, int label)
{
    srand((unsigned)time(0));
    ofstream imageData(xTrainData, std::ios_base::app);
    ofstream labelData(yTrainData, std::ios_base::app);
    int counter = 0;
    if (label == 1)
    {
        counter = 2000;
    }
    else
    {
        counter = 4000;
    }
    Mat image;
    Mat temp;
    for (int i = 0; i < counter; i++) // tochange to e.g. 1500 or 1000
    {
        int result = (rand() % (count - 1));
        image = imread(patchDataset + patchImages[result][0]);
        if (preprocessType == 1)
        {
            temp = image.clone();
        }
        else if (preprocessType == 2)
        {
            cvtColor(image, temp, COLOR_BGR2GRAY);
        }
        else if (preprocessType == 3)
        {
            GaussianBlur(image, temp, Size(7, 7), 2, 2);
        }
        imageData << format(temp, cv::Formatter::FMT_CSV) << endl;
        labelData << to_string(label) << endl;
    }
    imageData.close();
    labelData.close();
}

/*
    Count the patches with label 0 and label 1 and run selectTrainingData() function
    for preparing X_train and y_train data to feed to CNN
    preprocessType: specifies the type of images to be stored and can be "rgb"=1, "grayscale"=2 or "gaussian"=3
*/
void preparePatchDataset(std::map<int, std::vector<string>> patchImages, int preprocessType)
{
    int size = patchImages.size();
    std::map<int, std::vector<string>> labelOne, labelZero;
    int countOne = 0, countZero = 0;
    for (int i = 0; i < size; i++)
    {
        if (patchImages[i][1] == "1")
        {
            labelOne[countOne] = patchImages[i];
            countOne++;
        }
        else
        {
            labelZero[countZero] = patchImages[i];
            countZero++;
        }
    }

    cout << "ones:" << countOne << "zero: " << countZero;
    selectTrainingData(preprocessType, labelOne, countOne, 1);
    selectTrainingData(preprocessType, labelZero, countZero, 0);
}

/*
    select 20 images randomly for testing different models of CNN
*/
void selectTestImages()
{
    srand((unsigned)time(0));
    Mat image, gray, gauss;
    cv::Mat mat;
    mat.rows = 7;
    mat.cols = 7;
    Size s = mat.size();
    int selected[20];
    for (int i = 0; i < 20; i++) // tochange to e.g. 3000 or 2000
    {
        int result = (rand() % 99);
        image = imread(datasetTrainPath + to_string(result) + ".png");
        cvtColor(image, gray, COLOR_BGR2GRAY);
        GaussianBlur(image, gauss, s, 2, 2);
        imwrite("testRgb/" + to_string(i) + ".png", image);
        imwrite("testGray/" + to_string(i) + ".png", gray);
        imwrite("testGauss/" + to_string(i) + ".png", gauss);
        cout << result << "-";
    }
}

/*

    Get bounding box coordinates from file generated by Python
    coordinatesFilename: the file that contains coordinates information, it is the output of python part
*/
std::map<int, std::vector<string>> getBoxCoordinates(string coordinatesFilename)
{
    string filename(coordinatesFilename);
    string file_contents;
    std::map<int, std::vector<string>> csv_contents;
    char delimiter = ',';

    file_contents = getFileData(filename);

    istringstream sstream(file_contents);
    std::vector<string> items;
    string record, filename2, detected, xmin, ymin, xmax, ymax;

    int counter = 0;
    while (std::getline(sstream, record))
    {
        istringstream line(record);
        while (std::getline(line, filename2, delimiter))
        {
            items.push_back(filename2);

            std::getline(line, detected, delimiter);
            items.push_back(detected);

            std::getline(line, detected, delimiter);
            items.push_back(detected);

            std::getline(line, xmin, delimiter);
            items.push_back(xmin);

            std::getline(line, ymin, delimiter);
            items.push_back(ymin);

            std::getline(line, xmax, delimiter);
            items.push_back(xmax);

            std::getline(line, ymax);
            items.push_back(ymax);
        }
        csv_contents[counter] = items;
        items.clear();
        counter += 1;
    }
    return csv_contents;
}

/*
    Draw a bounding box around detected banana zone
    imgStoringPath: the path that images after drawing bounding box will be stored 
*/
void drawBox(std::map<int, std::vector<string>> coordinates, string imgStoringPath)
{
    int counter = coordinates.size();
    int filename1, isDetected, xmin, ymin, xmax, ymax;
    string filename;
    std::vector<string> rowData;
    for (int i = 0; i < counter; i++)
    {
        rowData = coordinates[i];
        isDetected = stoi(rowData[2]);
        if (isDetected == 1)
        {

            filename = rowData[1];

            xmin = stoi(rowData[3]);

            ymin = stoi(rowData[4]);

            xmax = stoi(rowData[5]);

            ymax = stoi(rowData[6]);

            Rect myROI(xmin, ymin, xmax - xmin, ymax - ymin);
            Mat src = imread("testRgb/" + filename);
            rectangle(src, Point(ymin, xmin), Point(ymax, xmax), (0, 255, 0), 2); //(ymin,xmin) (ymax,xmax)
            //cout << filename1;
            imwrite(imgStoringPath + filename, src);
        }
    }
}
