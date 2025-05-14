#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>

#define H 720
#define W 1280
#define C 3

#define R_lower 230
#define R_upper 255
#define G_lower 0
#define G_upper 15
#define B_lower 0
#define B_upper 15

//analoog aan V0 maar in deze versie wordt asynchroon telkens de volgende image ingeladen terwijl de GPU
//de rode bol zoekt
//merk op: de CPU moet wachten op de resultaten van de GPU
//=> het berekenen van het midden van de gevonden bol kan niet asynchroon

__global__ void find_object_colour(const uint8_t* pixels, int* row_counts, int* col_counts) {
    //doordat met array werkt en er per element van elk van de arrays
    //slechts 1 thread moet aanpassingen doen is geen atomic operation nodig

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < H){
        //eerste H threads zijn voor rijen
        int count = 0;
        for (int j = 0; j < W; ++j) {
            int idx = i * W * C + j * C;
            uint8_t r = pixels[idx];
            uint8_t g = pixels[idx + 1];
            uint8_t b = pixels[idx + 2];

            if (r >= R_lower && r <= R_upper &&
                g >= G_lower && g <= G_upper &&
                b >= B_lower && b <= B_upper) {
                count++;
                
            }
        }
        row_counts[i] = count;
    }
    else if(i < (H+W)){
        int j = i -  H;
        //volende W threads zijn voor kolommen
        int count = 0;
        for (int c = 0; c < H; ++c) {
            int idx = c * W * C + j * C;
            uint8_t r = pixels[idx];
            uint8_t g = pixels[idx + 1];
            uint8_t b = pixels[idx + 2];

            if (r >= R_lower && r <= R_upper &&
                g >= G_lower && g <= G_upper &&
                b >= B_lower && b <= B_upper) {
                count++;
                
            }
        }
        col_counts[j] = count;
    }
    else{
        return;
    }
    
}

uint8_t* get_image_array_ppm(const char* image) {
    FILE* imageFile = fopen(image, "rb");
    if (!imageFile) {
        perror("ERROR: Cannot open image file");
        exit(EXIT_FAILURE);
    }

    char header[2];
    fscanf(imageFile, "%s", header);
    int width, height, maxval;
    fscanf(imageFile, "%d %d", &width, &height);
    fscanf(imageFile, "%d", &maxval);
    fgetc(imageFile);

    uint8_t* image_array = (uint8_t*)malloc(H * W * C);
    fread(image_array, sizeof(uint8_t), H * W * C, imageFile);
    fclose(imageFile);
    return image_array;
}

void save_coordinates_to_file(const std::vector<std::pair<int, int>>& coordinates) {
    std::ofstream outFile("./output/coordinates_cuda_V6.txt");
    for (const auto& p : coordinates) {
        outFile << p.first << " " << p.second << "\n";
    }
    outFile.close();
}

int main() {
    std::vector<std::pair<int, int>> object_coordinates;

    

    uint8_t* d_pixels;
    int* d_row_counts;
    int* d_col_counts;

    std::chrono::time_point<std::chrono::system_clock> StartTime = std::chrono::system_clock::now();

    uint8_t* pixels = NULL;
    
    for (int frameNr = 0; frameNr < 151; ++frameNr) { //alles loopt 1 achter dus vandaar dat we tot 150 moeten kunnen gaan
                                                      //omwille van de NULL check is de 1ste cycle eigenlijk verloren
        std::vector<int> row_counts(H, 0);
        std::vector<int> col_counts(W, 0);

        if(pixels != NULL) {
            //stream
            cudaStream_t stream0;
            cudaStreamCreate(&stream0);

            cudaMalloc(&d_pixels, H * W * C);
            //cudaMemcpy(d_pixels, pixels, H * W * C, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(d_pixels, pixels, H * W * C, cudaMemcpyHostToDevice, stream0);

            cudaMalloc(&d_row_counts, H * sizeof(int));
            cudaMalloc(&d_col_counts, W * sizeof(int));
            cudaMemset(d_row_counts, 0, H * sizeof(int));
            cudaMemset(d_col_counts, 0, W * sizeof(int));

            // kernel gegevens
            int row_block_size = 1024;
            int col_block_size = 1024;
            int grid_size = ((H + W) / row_block_size) +1;

            find_object_colour<<<grid_size, row_block_size, 0, stream0>>>(d_pixels, d_row_counts, d_col_counts);
            //cudaDeviceSynchronize();

            //cudaMemcpy(row_counts.data(), d_row_counts, H * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(col_counts.data(), d_col_counts, W * sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpyAsync(row_counts.data(), d_row_counts, H * sizeof(int), cudaMemcpyDeviceToHost, stream0);
            cudaMemcpyAsync(col_counts.data(), d_col_counts, W * sizeof(int), cudaMemcpyDeviceToHost, stream0);

            cudaStreamDestroy(stream0);
        }

        if(frameNr != 150) { // "150.ppm" bestaat niet => we mogen enkel nieuwe images inladen tot in de voorlaatste loop
            std::string path = "./CUDA_versions/frames/moving_ball_frames_ppm/" + std::to_string(frameNr) + ".ppm";
            pixels = get_image_array_ppm(path.c_str());
        }

        cudaDeviceSynchronize(); //hier syncen: we moeten wachten tot de resultaten van de GPU klaar zijn voor we verder kunnen gaan
        //Dit wnr reductie is ook op GPU doen, nu op CPU
        //kolom/rijnummer in array zetten en op een manier midden zoeken
        double row_sum = 0, row_count = 0, col_sum = 0, col_count = 0;
        for (int i = 0; i < H; ++i) {
            if (row_counts[i]) { row_sum += i; row_count += 1; }
        }
        for (int j = 0; j < W; ++j) {
            if (col_counts[j]) { col_sum += j; col_count += 1; }
        }

        int row_mean = (row_count > 0) ? round(row_sum / row_count) : 0;
        int col_mean = (col_count > 0) ? round(col_sum / col_count) : 0;
        object_coordinates.push_back({row_mean, col_mean});

        //free(pixels);
        cudaFree(d_pixels);
        cudaFree(d_row_counts);
        cudaFree(d_col_counts);
    }
    free(pixels);

    std::chrono::time_point<std::chrono::system_clock> EndTime = std::chrono::system_clock::now();

    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();

    std::cout << "Time: " << milliseconds << " milliseconds.";

    save_coordinates_to_file(object_coordinates) ;
    return 0;
}
