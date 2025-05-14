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

//er word voor iedere pixel een thread voorzien => 720 * 1280 = 921.600 threads = overklokken scenario => threads moeten in groepjes sequentieel uitvoeren 


__global__ void find_object_colour(const uint8_t* pixels, int* row_counts, int* col_counts) {
    //doordat met array werkt en er per element van elk van de arrays
    //slechts 1 thread moet aanpassingen doen is geen atomic operation nodig

    int i = blockIdx.x * blockDim.x + threadIdx.x; //thread index

    if(i < W * H){
        int idx = i * C; //index waarnaar thread i moet kijken in de flattened image array
                         //  0        1        2        => thread index (i)  |
                         // [R, G, B, R, G, B, R, G, B]                      | => i * C (= 3) = idx
                         //  0  1  2  3  4  5  6  7  8  => array index (idx) |

        //printf("idx: %d \n", idx);

        int count = 0;

        uint8_t r = pixels[idx];
        uint8_t g = pixels[idx + 1];
        uint8_t b = pixels[idx + 2];

        if (r >= R_lower && r <= R_upper &&
            g >= G_lower && g <= G_upper &&
            b >= B_lower && b <= B_upper) {

            int row_index = i / W;
            int column_index = i % W;
            
            //count++; 
            atomicAdd(&row_counts[row_index], 1); //we moeten hier een atomicAdd doen aangezien iedere thread nu één pixel vertegenwoordigt
                                                  //=> meerdere threads accessen gelijktijdig dezelfde index in de row_counts array en de
                                                  //col_counts array aangezien meerdere pixels op eenzelfde rij en kolom staan (deze arrays 
                                                  //hebben een lengte die gelijk is aan het aantal kolommen en rijen in de afbeelding)
                                                  //
                                                  //mogelijke andere manier:
                                                  //één flattened array maken met grote 921600 (voor iedere pixel een slot) en dan in preprocessing
                                                  //de index in deze flat array terugrekenen naar een matrix en zo het gemiddelde bepalen door het
                                                  //midden te tellen op de CPU te doen
            atomicAdd(&col_counts[column_index], 1);  
        }

        //int row_index = i / W; //    0  1  2
                                 // 0 [R, G, B]
                                 // 1 [X, G, B]      0  1  2    3  4  5    6  7  8
                                 // 2 [R, G, B] => [[R, G, B], [X, G, B], [R, G, B]] => X [3 = idx] (array) => 3 / 3 (= W) = 1 = row_index (matrix)
        //printf("row_index: %d \n", row_index);

        //int column_index = i % W; //rest geeft kolom weer
        //printf("column_index: %d \n", column_index);

        // row_counts[row_index] = count;
        // col_counts[column_index] = count;
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
    std::ofstream outFile("./output/coordinates_cuda_V3.txt");
    for (const auto& p : coordinates) {
        outFile << p.first << " " << p.second << "\n";
    }
    outFile.close();
}

int main() {
    std::vector<std::pair<int, int>> object_coordinates;

    std::vector<float> kernel_times;

    uint8_t* d_pixels;
    int* d_row_counts;
    int* d_col_counts;

    //std::chrono::time_point<std::chrono::system_clock> StartTime = std::chrono::system_clock::now();

    for (int frameNr = 0; frameNr < 150; ++frameNr) {
        std::string path = "./CUDA_versions/frames/moving_ball_frames_ppm/" + std::to_string(frameNr) + ".ppm";
        uint8_t* pixels = get_image_array_ppm(path.c_str());

        std::vector<int> row_counts(H, 0);
        std::vector<int> col_counts(W, 0);

        cudaMalloc(&d_pixels, H * W * C);
        cudaMemcpy(d_pixels, pixels, H * W * C, cudaMemcpyHostToDevice);

        cudaMalloc(&d_row_counts, H * sizeof(int));
        cudaMalloc(&d_col_counts, W * sizeof(int));
        cudaMemset(d_row_counts, 0, H * sizeof(int));
        cudaMemset(d_col_counts, 0, W * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // kernel gegevens
        int row_block_size = 1024;
        int col_block_size = 1024;
        int grid_size = ((H * W) / row_block_size); //levert 900 blocks op

        cudaEventRecord(start);
        find_object_colour<<<grid_size, row_block_size>>>(d_pixels, d_row_counts, d_col_counts); //wordt gecalled met 900 blocks van elk 1024 threads
        cudaEventRecord(stop);                                                     
        cudaDeviceSynchronize();

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        kernel_times.push_back(milliseconds);

        cudaMemcpy(row_counts.data(), d_row_counts, H * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_counts.data(), d_col_counts, W * sizeof(int), cudaMemcpyDeviceToHost);

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

        free(pixels);
        cudaFree(d_pixels);
        cudaFree(d_row_counts);
        cudaFree(d_col_counts);

        std::cout << milliseconds << " milliseconds" <<std::endl;
    }

    // std::chrono::time_point<std::chrono::system_clock> EndTime = std::chrono::system_clock::now();

    // int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(EndTime - StartTime).count();

    // std::cout << "Time: " << milliseconds << " milliseconds.";

    float total_time = 0.0f;
    for (float t : kernel_times) total_time += t;
    float average_time = (kernel_times.size() > 0) ? total_time / kernel_times.size() : 0.0f;

    std::cout << "Total kernel execution time: " << total_time << " ms" << std::endl;
    std::cout << "Average kernel time per frame: " << average_time << " ms" << std::endl;


    save_coordinates_to_file(object_coordinates) ;
    return 0;
}
