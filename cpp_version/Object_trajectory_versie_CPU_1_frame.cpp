#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>

// Dit is versie op 1 frame
// Dit is versie volledig in C++: benchmark 
using namespace std;

//-------------- Hier wijzigen Hoogte, breedte, en pad naar image --------------//
#define H 3
#define W 3
#define C 3

#define image "./images/easy"

uint8_t *get_image_array_ppm(void)
{
    FILE *imageFile;
    imageFile = fopen(image, "rb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Lees header informatie (skippen) 
    char header[3];
    fscanf(imageFile, "%s", header);  // P6
    int width, height, maxval;
    fscanf(imageFile, "%d %d", &width, &height);
    fscanf(imageFile, "%d", &maxval);
    fgetc(imageFile);  

    // lees pixel data en laad pixels in flattened array 
    uint8_t *image_array = (uint8_t *)malloc(H * W * C * sizeof(uint8_t));
    fread(image_array, sizeof(uint8_t), H * W * C, imageFile);
    fclose(imageFile);

    return image_array;
}

//Functie komt uit gekregen code van Toledo
void save_image_array(uint8_t *image_array)
{
    FILE *imageFile;
    imageFile = fopen("./output_Image_test.ppm", "wb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    fprintf(imageFile, "P6\n");          // P6 filetype
    fprintf(imageFile, "%d %d\n", H, W); // dimensions
    fprintf(imageFile, "255\n");         // Max pixel

    fwrite(image_array, 1, H * W * C, imageFile);
    fclose(imageFile);
}

//van coordinaten naar index in flattened array (RGB format)
int getRGBFlattenedIndex(int i, int j)
{
  return (i * W + j) * 3; // *3 omdat in RGBRGB
}

//krijg coordinaten in image, van index in flattened array
pair<int, int> getRGBMatrixCoordinates(int index)
{
  int pixelMatrixIndex = index / 3;
  int i = pixelMatrixIndex / W;
  int j = pixelMatrixIndex % W;
  return {i, j};
}

// print matrices -> voor debugging 
void print_rgb_matrices(uint8_t *pixels)
{
    int red_matrix[H][W] = {0};
    int green_matrix[H][W] = {0};
    int blue_matrix[H][W] = {0};

    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            int index = (i * W + j) * 3; 
            red_matrix[i][j] = (int)pixels[index];
            green_matrix[i][j] = (int)pixels[index + 1];
            blue_matrix[i][j] = (int)pixels[index + 2];
        }
    }

    std::cout << "Red Matrix:" << endl;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            std::cout << red_matrix[i][j] << " ";
        }
        std::cout << endl;
    }

    std::cout << "Green Matrix:" << endl;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            std::cout << green_matrix[i][j] << " ";
        }
        std::cout << endl;
    }

    std::cout << "Blue Matrix:" << endl;
    for (int i = 0; i < H; i++)
    {
        for (int j = 0; j < W; j++)
        {
            std::cout << blue_matrix[i][j] << " ";
        }
        std::cout << endl;
    }
}

//coordinaten opslaan in txt file (output)
//eerste lijn is hoogte en breedte image
// lijnen erna zijn coordinaten waar object zich bevondt op frame 1, 2, 3, ...
void save_coordinates_to_file(const std::vector<std::pair<int, int>> &coordinates)
{
    std::ofstream outFile("coordinates.txt");
    if (outFile.is_open())
    {   
        outFile << H << " " << W << std::endl;
        for (const auto &coord : coordinates)
        {
            outFile << coord.first << " " << coord.second << std::endl;
        }
        outFile.close();
    }
    else
    {
        std::cerr << "Unable to open the file to save coordinates." << std::endl;
    }
}

int main()
{

  // defineer range per kleur
  // lower bound en upper bound
  int red_range[2] = {250, 255};
  int green_range[2] = {0, 0};
  int blue_range[2] = {0, 0};

  //array find_row wordt op index verhoogd als in de rij van die index een pixel in de kleurrange is gevonden
  //Later kan elke thread per rij gelijktijdig toegang krijgen
  int find_row[H] = {0};
  int find_column[W] = {0};

  // Flatten matrices in 1 array
  // rgbrgbrgb
  uint8_t* pixels = get_image_array_ppm();

  // cout << "Matrices: " << endl;
  // print_rgb_matrices(pixels);
  // cout << "----------------" << endl;

  // pixels vinden die alle kleurranges voldoen
  // per kleur kun je met CUDA in principe ook in parallel checken
  for (int i = 0; i < H; i++)
  {
    for (int j = 0; j < W; j++)
    {
        std::cout << i << "," << j <<endl;
      int index = getRGBFlattenedIndex(i, j);
      std::cout << "Index: " << index << endl;
      int r = (int) pixels[index ]; //+0
      if (r >= red_range[0] && r <= red_range[1])
      {
        int g = (int)pixels[index + 1];
        if (g >= green_range[0] && g <= green_range[1])
        {
          int b = (int)pixels[index +2]; //+2
          if (b >= blue_range[0] && b <= blue_range[1])
          {
            find_column[j] += 1;
            find_row[i] += 1;
          }
        }
      }
    }
  }
  //eig kun je onderbreken vanaf een rij/kolom tegenkomt na kolom/rij in range dat niet meer in range zit
  //want dan zit je voorbij object


  double meanheight = 0;
  double objectheight = 0;
  
  for (int h = 0; h < H; h++)
  {
    std::cout << "find_column " << h << ": "<< find_column[h] << endl;
    if (find_column[h] > 0)
    {
      meanheight += h;
      objectheight++;
    }
  }
  meanheight /= objectheight;
  std::cout << "mean: " << meanheight << "," << objectheight << endl;
  int roundedMeanHeight = round(meanheight);

  double meanwidth = 0;
  double objectwidth = 0;
  for (int j = 0; j < W; ++j)
  {
    if (find_row[j] > 0)
    {
      meanwidth += j;
      objectwidth++;
    }
  }
  meanwidth /= objectwidth;
  int roundedMeanWidth = round(meanwidth);

  std::cout << "Coordinates of the object: (" << roundedMeanHeight << ", " << roundedMeanWidth << ")" << endl;
  std::vector<std::pair<int, int>> object_coordinates;

  object_coordinates.push_back({roundedMeanHeight, roundedMeanWidth});

  save_coordinates_to_file(object_coordinates);

  free(pixels);
  return 0;
}
