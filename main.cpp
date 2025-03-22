#include <iostream>
#include <vector>
#include <cmath>

// eerst alles in c++ schrijven, daarna cuda versies
// dit als benchmark
using namespace std;

// 5x5 matrix
#define H 5
#define W 5
// file maken voor renderen, voor afbeelding uit framen... beetje deftig met klassen maken

// functie om frames uit video te halen

// per frame: functie om locatie van bepaald kleur eruit te halen

// 3 matrices: naast elkaar flatten wrs dus meerdere dingen vergelijken

// function: krijg index in flattened array van coordinaten
int getRGBFlattenedIndex(int i, int j)
{
  return (i * W + j) * 3; // *3 omdat in RGBRGB
}

pair<int, int> getRGBMatrixCoordinates(int index)
{
  int pixelMatrixIndex = index / 3;
  int i = pixelMatrixIndex / W;
  int j = pixelMatrixIndex % W;
  return {i, j};
}

int main()
{

  // oproepen functie om frames uit video te halen

  // itereren over alle frames en functie om coordinaat eruit te halen oproepen

  // voor alle coordinaten, punt tekenen en dus lijn die object heeft gedaan tekenen

  // matrix per kleur
  int red_pixels[H][W] = {{0, 0, 0, 0, 0},
                          {0, 0, 0, 5, 5},
                          {0, 5, 2, 2, 5},
                          {0, 5, 2, 2, 5},
                          {0, 5, 5, 5, 5}};

  int green_pixels[H][W] = {{0, 0, 0, 0, 0},
                            {0, 0, 0, 3, 3},
                            {0, 3, 1, 1, 3},
                            {0, 3, 1, 1, 3},
                            {0, 3, 3, 3, 3}};

  int blue_pixels[H][W] = {{0, 0, 0, 0, 0},
                           {0, 0, 0, 4, 4},
                           {0, 4, 2, 2, 4},
                           {0, 4, 2, 2, 4},
                           {0, 4, 4, 4, 4}};

  // defineer range per kleur
  // lower bound en upper bound
  int red_range[2] = {1, 2};
  int green_range[2] = {1, 2};
  int blue_range[2] = {1, 2};

  // Flatten matrices in 1 array
  // rgbrgbrgb
  vector<int> pixels(3 * H * W);
  for (int i = 0; i < H; i++)
  {
    for (int j = 0; j < W; j++)
    {
      int index = getRGBFlattenedIndex(i, j);
      pixels[index] = red_pixels[i][j];
      pixels[index + 1] = green_pixels[i][j];
      pixels[index + 2] = blue_pixels[i][j];
    }
  }

  // pixels vinden die alle kleurranges voldoen
  //per kleur kun je met CUDA in principe ook in parallel checken
  for (int i = 1; i < H; i++)
  {
    for (int j = 1; j < W; j++)
    {
      int index = getRGBFlattenedIndex(i, j);
      int r = pixels[index];
      if (r >= red_range[0] && r <= red_range[1])
      {
        int g = pixels[index + 1];
        if (g >= green_range[0] && g <= green_range[1])
        {
          int b = pixels[index + 2];
          if (b >= blue_range[0] && b <= blue_range[1])
          {
            pixels[getRGBFlattenedIndex(i, 0)] += 1;
            pixels[getRGBFlattenedIndex(0, j)] += 1;
          }
        }
      }
    }
  }

  // als in 0e rij of 0e kolom 1 staat dan die index meenemen in gemdidelde index voor coordinaten
  double meanheight = 0;
  double objectheight = 0;
  for (int h = 1; h < H; h++)
  {
    if (pixels[getRGBFlattenedIndex(h, 0)] > 0)
    {
      meanheight += h;
      objectheight++;
    }
  }
  meanheight /= objectheight;
  int roundedMeanHeight = round(meanheight);

  double meanwidth = 0;
  double objectwidth = 0;
  for (int j = 1; j < W; ++j)
  {
    if (pixels[getRGBFlattenedIndex(0,j)] > 0)
    {
      meanwidth += j;
      objectwidth++;
    }
  }
  meanwidth /= objectwidth;
  int roundedMeanWidth = round(meanwidth);

  cout << "Coordinates of the object: (" << roundedMeanHeight << ", " << roundedMeanWidth << ")" << endl;


  return 0;
}
