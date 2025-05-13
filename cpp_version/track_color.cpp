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
#define H 720
#define W 1280
#define C 3

// defineer range per kleur
//------COLOR BOUNDS:------
#define R_lower 230
#define R_upper 255

#define G_lower 0
#define G_upper 15

#define B_lower 0
#define B_upper 15
//-------------------------

uint8_t *get_image_array_ppm(const char* image)
{
    FILE *imageFile;
    imageFile = fopen(image, "rb");
    if (imageFile == NULL)
    {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Lees header informatie (skippen) 
    char header[2];
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

//van coordinaten naar index in flattened array (RGB format)
int getRGBFlattenedIndex(int i, int j) //i = rij (720), j = kolom (1280)
{
  int index;
  index = i * W * C;
  index += j * C;
  return index;
}

//coordinaten opslaan in txt file (output)
// lijnen zijn coordinaten waar object zich bevondt op frame 1, 2, 3, ...
void save_coordinates_to_file(const std::vector<std::pair<int, int>> &coordinates)
{
    std::ofstream outFile("./../output/coordinates.txt");
    if (outFile.is_open())
    {   
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

bool in_range(uint8_t* pixels, int index){
    //get the red value
    int r = (int) pixels[index];

    if(r >= R_lower && r <= R_upper){
        //get the green value
        int g = (int) pixels[index + 1];
        if(g >= G_lower && g <= G_upper){

            //get the blue value
            int b = (int) pixels[index + 2];
            if(b >= B_lower && b <= B_upper){
                
                return true;

            }else{

                return false;

            }

        }else{

            return false;

        }

    }else{

        return false;

    }
}

int main(){

    std::vector<std::pair<int, int>> object_coordinates;

    for(int frameNr = 0; frameNr < 150; frameNr++){
    //---------BOUNDS:---------
    int Row_lower_bound = -1;
    int Row_upper_bound = 0;
    int Column_lower_bound = -1;
    int Column_upper_bound = 0;
    //-------------------------

    //inlezen frame
    string Prefix = "./../frames/moving_ball_frames_ppm/";
    string Nr = to_string(frameNr);
    string Suffix = ".ppm";
    string total = Prefix + Nr + Suffix;
    //cout << total << endl;
    uint8_t* pixels = get_image_array_ppm(total.c_str());

    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            // get the flattened index
            int index = getRGBFlattenedIndex(i, j);

            if(in_range(pixels, index)){
                if(i > Row_upper_bound){
                    Row_upper_bound = i;
                }else if(i < Row_lower_bound || Row_lower_bound == -1){
                    Row_lower_bound = i;
                }

                if(j > Column_upper_bound){
                    Column_upper_bound = j;
                }else if(j < Column_lower_bound || Column_lower_bound == -1){
                    Column_lower_bound = j;
                }
            }  
        }
    }

    int Middle_row = ceil(((Row_upper_bound - Row_lower_bound)/2) + Row_lower_bound); //ceiled niet correct?
    int Middle_column = ceil(((Column_upper_bound - Column_lower_bound)/2) + Column_lower_bound); //ceileid niet correct?

    object_coordinates.push_back({Middle_row + 1, Middle_column + 1}); //vandaar hier "+ 1"

    free(pixels);
  }

  save_coordinates_to_file(object_coordinates);

  
  return 0;
}