#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

//------COLOR BOUNDS:------
//--------rode bal
// #define H 720
// #define W 1280
// #define C 3

// #define R_lower 230
// #define R_upper 255

// #define G_lower 0
// #define G_upper 15

// #define B_lower 0
// #define B_upper 15

//--------oranje bal
#define H 1080
#define W 1920
#define C 3

#define R_lower 240
#define R_upper 255

#define G_lower 170
#define G_upper 230

#define B_lower 80
#define B_upper 150
//-------------------------

using namespace std;

uint8_t* get_image_array_ppm(const char* image) {
    FILE* imageFile = fopen(image, "rb");
    if (!imageFile) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    char header[2];
    fscanf(imageFile, "%s", header); // P6
    int width, height, maxval;
    fscanf(imageFile, "%d %d", &width, &height);
    fscanf(imageFile, "%d", &maxval);
    fgetc(imageFile);

    uint8_t* image_array = (uint8_t*)malloc(H * W * C * sizeof(uint8_t));
    fread(image_array, sizeof(uint8_t), H * W * C, imageFile);
    fclose(imageFile);

    return image_array;
}

int getRGBFlattenedIndex(int i, int j) {
    return i * W * C + j * C;
}

void save_coordinates_to_file(const std::vector<std::pair<int, int>>& coordinates) {
    std::ofstream outFile("./../output/coordinates.txt");
    if (outFile.is_open()) {
        for (const auto& coord : coordinates) {
            outFile << coord.first << " " << coord.second << std::endl;
        }
        outFile.close();
    } else {
        std::cerr << "Unable to open the file to save coordinates." << std::endl;
    }
}

int main() {
    std::vector<std::pair<int, int>> object_coordinates;

    std::chrono::time_point<std::chrono::system_clock> StartTime = std::chrono::system_clock::now();

    for (int frameNr = 0; frameNr < 150; frameNr++) {
        string filePath = "./../frames/moving_ball_frames_ppm/" + to_string(frameNr) + ".ppm";
        uint8_t* pixels = get_image_array_ppm(filePath.c_str());

        std::vector<int> find_row(H, 0); //initialiseert waarden op 0
        std::vector<int> find_column(W, 0);

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                int index = getRGBFlattenedIndex(i, j);
                int r = (int)pixels[index];
                if (r >= R_lower && r <= R_upper) {
                    int g = (int)pixels[index + 1];
                    if (g >= G_lower && g <= G_upper) {
                        int b = (int)pixels[index + 2];
                        if (b >= B_lower && b <= B_upper) {
                            find_column[j] += 1;
                            find_row[i] += 1;
                        }
                    }
                }
            }
        }

        double sum_rows = 0, count_rows = 0;
        for (int i = 0; i < H; i++) {
            if (find_row[i] > 0) {
                sum_rows += i;
                count_rows += 1;
            }
        }
        double mean_row = (count_rows > 0) ? sum_rows / count_rows : 0;
        int roundedMeanRow = round(mean_row);

        double sum_columns = 0, count_columns = 0;
        for (int j = 0; j < W; j++) {
            if (find_column[j] > 0) {
                sum_columns += j;
                count_columns += 1;
            }
        }
        double mean_column = (count_columns > 0) ? sum_columns / count_columns : 0;
        int roundedMeanColumn = round(mean_column);

        object_coordinates.push_back({roundedMeanRow , roundedMeanColumn}); 
        free(pixels);
    }

    std::chrono::time_point<std::chrono::system_clock> EndTime = std::chrono::system_clock::now();

    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();

    std::cout << "Time: " << milliseconds << " milliseconds.";

    

    save_coordinates_to_file(object_coordinates);
    return 0;
}
