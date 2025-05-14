import numpy as np
import matplotlib.pyplot as plt

# Bestanden met tijden in milliseconden
input1 = 'V2_frame_time_full_warp.txt'
input2 = 'V2_frame_time_underutilized_warp.txt'

def read_times_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter lege lijnen en zet om naar float
    times = [float(line.strip()) for line in lines if line.strip()]
    return times

# Tijden inlezen
input1_times = read_times_from_file(input1)
input2_times = read_times_from_file(input2)

# Gemiddelden berekenen
mean_v0 = np.mean(input1_times)
mean_v1 = np.mean(input2_times)

# Boxplot tekenen
plt.figure(figsize=(12, 8))
plt.boxplot([input1_times, input2_times], labels=['Full warps', 'Underutilized warps'], patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'))

plt.text(1, mean_v0 + 1, f'Gemiddelde: {mean_v0:.2f} ms', horizontalalignment='center', color='blue', fontsize=14)
plt.text(2, mean_v1 - 2, f'Gemiddelde: {mean_v1:.2f} ms', horizontalalignment='center', color='red', fontsize=14)

plt.title('Boxplot uitvoeringstijd per Frame (in Milliseconds)', fontsize=18)
plt.xlabel('Versie (v2: full warp vs underutilized warp)', fontsize=14)
plt.ylabel('Uitvoeringstijd (milliseconds)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()