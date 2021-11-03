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
    std::map<int, std::vector<string>> coords = getBoxCoordinates("gray.txt");
    drawBox(coords, "grayOutput/");
}