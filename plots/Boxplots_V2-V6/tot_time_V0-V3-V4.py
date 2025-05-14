import numpy as np
import matplotlib.pyplot as plt

# Bestanden met tijden in milliseconden
input1 = 'V2_tot_time_full_warp.txt'
input2 = 'V3_total_time.txt'
input3 = 'V4_total_time.txt'  # <-- Extra inputbestand

def read_times_from_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines if line.strip()]

# Tijden inlezen
times1 = read_times_from_file(input1)
times2 = read_times_from_file(input2)
times3 = read_times_from_file(input3)

# Gemiddelden berekenen
mean1 = np.mean(times1)
mean2 = np.mean(times2)
mean3 = np.mean(times3)

# Alle data en labels
all_data = [times1, times2, times3]
labels = ['V0', 'V3', 'V4']

# Boxplot tekenen
plt.figure(figsize=(12, 8))
plt.boxplot(all_data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='red'))

# Y-as limieten dynamisch instellen
all_times_flat = times1 + times2 + times3
min_y = min(all_times_flat) - 50
max_y = max(all_times_flat) + 50
plt.ylim(min_y, max_y)

# Tekst met gemiddelden bij elke box
plt.text(1, mean1 - 100, f'Gemiddelde: {mean1:.2f} ms', ha='center', color='blue', fontsize=14)
plt.text(2, mean2 + 100, f'Gemiddelde: {mean2:.2f} ms', ha='center', color='red', fontsize=14)
plt.text(3, mean3 + 100, f'Gemiddelde: {mean3:.2f} ms', ha='center', color='green', fontsize=14)

# Titels en labels
plt.title('Boxplot van Totale Uitvoertijden voor V0, V3 en V4 (in Milliseconds)', fontsize=18)
plt.xlabel('Versie', fontsize=14)
plt.ylabel('Uitvoeringstijd (milliseconds)', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()