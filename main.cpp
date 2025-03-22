#include <iostream>
#include <vector>
#include <cmath>

//eerst alles in c++ schrijven, daarna cuda versies
//dit als benchmark
using namespace std;

#define H 5
#define W 5
//file maken voor renderen, voor afbeelding uit framen... beetje deftig met klassen maken

//functie om frames uit video te halen

//per frame: functie om locatie van bepaald kleur eruit te halen

//3 matrices: naast elkaar flatten wrs dus meerdere dingen vergelijken

int main() {

    //oproepen functie om frames uit video te halen

    //itereren over alle frames en functie om coordinaat eruit te halen oproepen

    //voor alle coordinaten, punt tekenen en dus lijn die object heeft gedaan tekenen


    // 5x5 image
    int image[5][5] = {{0, 0, 0, 0, 0},
                       {0, 0, 0, 5, 5},
                       {0, 5, 2, 2, 5},
                       {0, 5, 2, 2, 5},
                       {0, 5, 5, 5, 5}};

    //vector<int> rows; // opslaan rij indices waar waarde ingevonden
    //vector<int> cols; 

    // definieer range
    int range[2] = {1, 2}; // lower bound 1, upper bound 2

    for (int i = 1; i < H; i++) {
        for (int j = 1; j < W; j++) {
            if (image[i][j] >= range[0] && image[i][j] <= range[1]) {
                //rows.push_back(i); // opslaan rij index
                //cols.push_back(j); 
                image[i][0] = 1;
                image[0][j]= 1;
            }
        }
    }

    //als in 0e rij of 0e kolom 1 staat dan die index meenemen in gemdidelde index voor coordinaten
    double meanheight = 0;
    double objectheight = 0;
    for (int h = 1; h < H; h++) {
      if(image[h][0]>0){
        meanheight += h;
        objectheight ++;
      }
    }
    meanheight /= objectheight;
    int roundedMeanHeight = round(meanheight); 

    double meanwidth = 0;
    double objectwidth = 0;
    for (int j = 1; j < W; ++j) {
      if(image[0][j]>0){
        meanwidth += j;
        objectwidth ++;
      }
    }
    meanwidth /= objectwidth;
    int roundedMeanWidth = round(meanwidth);

    // // check als iets van kleur gevonden
    // if (rows.empty() || cols.empty()) {
    //     cout << "No objects found in the specified range." << endl;
    //     return 0; // exit als geen van bepaald voorwerp gevonden
    // }

    // double meanRow = 0;
    // for (int i = 0; i < rows.size(); ++i) {
    //     meanRow += rows[i];
    // }
    // meanRow /= rows.size(); 
    // int roundedMeanRow = round(meanRow); 


    // double meanCol = 0;
    // for (int i = 0; i < cols.size(); ++i) {
    //     meanCol += cols[i];
    // }
    // meanCol /= cols.size(); 
    // int roundedMeanCol = round(meanCol); 

    cout << "Coordinates of the object: (" << roundedMeanHeight << ", " << roundedMeanWidth << ")" << endl;

    return 0;
}
