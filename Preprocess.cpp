#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "bananaFunc.h"

int main()
{
    std::map<int, std::vector<string>> rawDatasetData = readDatasetData();
    createPatchesDataset(rawDatasetData);
    std::map<int, std::vector<string>> patchImages = getPatchesData();
    preparePatchDataset(patchImages, 2);
}