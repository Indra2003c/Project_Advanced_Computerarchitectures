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
#include <stdio.h>

#define H 720 //128 //
#define W 1280 //256 //
#define C 3

#define R_lower 230
#define R_upper 255
#define G_lower 0
#define G_upper 15
#define B_lower 0
#define B_upper 15

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
                //count++;
                row_counts[i] = i;
                break; 
            }
        }
        
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
                //count++;
                col_counts[j] = j;
                break;
            }
        }
        //col_counts[j] = count;
    }
    else{
        return;
    }
    
}

__global__ void min_max_index(int* row_counts, int* col_counts, int log_2_H, int log_2_W, int H_power, int W_power) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int* d_arr;
    int log_2_size;
    int N;
    for(int i = 0; i <2; i++){ //iteratie 0 is voor de rijen, iteratie 1 is voor de kolommen
        if(i == 0){
            d_arr = row_counts;
            log_2_size = log_2_H;
            N = H_power;
        }
        else{
            d_arr = col_counts;
            log_2_size = log_2_W;
            N = W_power;

        }

        for(int iteratie = 1; iteratie <= log_2_size; iteratie++){
            //enkel threads die nodig zijn gebruiken voor die iteratie
            
            //threads need to put max in val1 and min in val2
            //in the first iteration
            if(iteratie == 1 && idx < pow(2, log_2_size)/(pow(2,iteratie))){ //1

                int val1 = idx*pow(2,iteratie) ;
                int val2 = idx*pow(2,iteratie) + (pow(2,iteratie)/2);

                if(d_arr[val1] == 0 && d_arr[val2] !=0 ){
                    //if one of the two is zero
                    //then I want to keep the number as minimum
                    d_arr[val1] = d_arr[val2];
                }
                //switch them in place
                //so minimum is to the left
                //maximum is to the right
                else if(d_arr[val2] < d_arr[val1]){ //else  <
                    int tmp = d_arr[val1];
                    d_arr[val1] = d_arr[val2];
                    d_arr[val2] = tmp;
                }
            }
            //Threads needed for min
            //moving min to the left
            else if(idx < pow(2, log_2_size)/(pow(2,iteratie))){
                int val1 = idx*pow(2,iteratie) ;
                int val2 = idx*pow(2,iteratie) + (pow(2,(iteratie-1)));///2                

                if(d_arr[val1] == 0 && d_arr[val2] !=0 ){
                    //if one of the two is zero
                    //then I want to keep the number as minimum
                    d_arr[val1] = d_arr[val2];
                }
                else if(d_arr[val2] == 0 && d_arr[val1] !=0 ){
                    //if one of the two is zero
                    //then I want to keep the number as minimum
                    d_arr[val2] = d_arr[val1];
                }
                else if(d_arr[val2] > d_arr[val1]){
                    d_arr[val2] = d_arr[val1];
                }
                
            }

            //threads doing the max operation
            else if(idx >= pow(2, log_2_size)/(pow(2,iteratie)) && idx < 2* pow(2, log_2_size)/(pow(2,iteratie))){

                int new_idx =idx- pow(2, log_2_size)/(pow(2,iteratie));
                //no check if a value is zero because the max wil never be zero if a pixel was detected
                
                int val1 = N - 1 - (new_idx*pow(2,iteratie) + (pow(2,(iteratie-1)))) ; 
                int val2 = N - 1 - new_idx*pow(2,iteratie); //max index, stays to the right

                if(d_arr[val2] < d_arr[val1]){
                    d_arr[val2] = d_arr[val1];
                }
                
            }
            
            __syncthreads();

        }

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
    std::ofstream outFile("./output/multiple_coordinates/coordinates_cuda_v1_2.txt");
    for (const auto& p : coordinates) {
        outFile << p.first << " " << p.second << "\n";
    }
    outFile.close();
}

void save_times_to_file(const std::vector<int>& times, int nanoseconds) {
    std::ofstream outFile("./output/multiple_times_v1/times_cuda_v1_reductie_20.txt");
    for (const auto& p : times) {
        outFile << p << "\n";
    }
    outFile << "Total time: " << nanoseconds << " nanoseconds";
    outFile.close();
    std::cout << "Wrote to " << "./output/times_cuda_v1_reductie_1.txt\n" ;
}

int nextPowerOf(int x){
    return std::pow(2, std::ceil(std::log2(x)));
}
int main() {
    std::vector<std::pair<int, int>> object_coordinates;
    std::vector<int> frame_times;
    

    uint8_t* d_pixels;
    int* d_row_counts;
    int* d_col_counts;

    std::chrono::time_point<std::chrono::system_clock> StartTime = std::chrono::system_clock::now();

    int H_power = nextPowerOf(H);
    int W_power = nextPowerOf(W);

    for (int frameNr = 0; frameNr < 150; ++frameNr) { //150
        std::chrono::time_point<std::chrono::system_clock> framestart = std::chrono::system_clock::now();
        std::string path = "./CUDA_versions/frames/moving_ball_frames_ppm/" + std::to_string(frameNr) + ".ppm";
        uint8_t* pixels = get_image_array_ppm(path.c_str());

        
        std::vector<int> row_counts(H_power, 0);
        std::vector<int> col_counts(W_power, 0);

        cudaMalloc(&d_pixels, H * W * C* sizeof(int));
        cudaMemcpy(d_pixels, pixels, H * W * C , cudaMemcpyHostToDevice);
        
        //std::cout << H << ", " << H_power <<"\n";
        cudaMalloc(&d_row_counts, H_power * sizeof(int));
        cudaMalloc(&d_col_counts, W_power * sizeof(int));
        cudaMemset(d_row_counts, 0, H_power * sizeof(int));
        cudaMemset(d_col_counts, 0, W_power * sizeof(int));

        // kernel gegevens
        int row_block_size = 1024;
        int col_block_size = 1024; //1280
        int grid_size = ((H + W) / row_block_size) +1;

        find_object_colour<<<grid_size, row_block_size>>>(d_pixels, d_row_counts, d_col_counts);
        cudaDeviceSynchronize();
        grid_size = ((H_power + W_power)  / row_block_size) +1;
        min_max_index<<<grid_size, row_block_size>>>(d_row_counts, d_col_counts, log2(H_power), log2(W_power), H_power, W_power);
        cudaDeviceSynchronize();

        cudaMemcpy(row_counts.data(), d_row_counts, H_power * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_counts.data(), d_col_counts, W_power * sizeof(int), cudaMemcpyDeviceToHost);

        //volgende deel was wanneer min en max bepalen ook op CPU was, nu ook op GPU (min_max_index)
        //kolom/rijnummer in array zetten en op een manier midden zoeken

        // midden op (max_index + min_index) / 2
        // double row_sum = 0, row_count = 0, col_sum = 0, col_count = 0;
        // for (int i = 0; i < H; ++i) {
        //     if (row_counts[i]) { row_sum += i; row_count += 1; }
        // }
        // for (int j = 0; j < W; ++j) {
        //     if (col_counts[j]) { col_sum += j; col_count += 1; }
        // }

        //int row_mean = (row_count > 0) ? round(row_sum / row_count) : 0;
        //int col_mean = (col_count > 0) ? round(col_sum / col_count) : 0;
        

        int n = row_counts.size()-1;

        //if no object: mean is 0
        //max is element with index n, min is element with index 0
        int row_mean = round((row_counts[n]+ row_counts[0])/2);
        n = col_counts.size()-1;
        int col_mean = round((col_counts[n]+ col_counts[0])/2);
        object_coordinates.push_back({row_mean, col_mean});


        free(pixels);
        cudaFree(d_pixels);
        cudaFree(d_row_counts);
        cudaFree(d_col_counts);

        std::chrono::time_point<std::chrono::system_clock> frame_end = std::chrono::system_clock::now();

        int nanoseconds_frame = std::chrono::duration_cast<std::chrono::nanoseconds>(frame_end - framestart).count();
        frame_times.push_back(nanoseconds_frame);
    }

    std::chrono::time_point<std::chrono::system_clock> EndTime = std::chrono::system_clock::now();

    int nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count();

    std::cout << "Time: " << nanoseconds << " nanoseconds.";

    save_coordinates_to_file(object_coordinates) ;
    save_times_to_file(frame_times,nanoseconds ) ;
    return 0;
}
