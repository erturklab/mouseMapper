import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

def isNaN(num):
    return num != num
    
# Load the data
file_path   = "/home/rami/Documents/UCHL1_HFD/pearth/Graph_Module/organ_statistics.csv"  
figure_path = "/home/rami/Documents/UCHL1_HFD/pearth/Graph_Module/comparison_figs/plots_in_um.png"
data = pd.read_csv(file_path)

# Define the column names
diet_col = 'Diet'  
place_col = 'Organ'
attribute_cols = ['Curliness Low', 'Curliness', 'Curliness High', 'Radius', 'Vertices', 'Edges', 'End nodes'] 

# Prepare the plot
fig, axs = plt.subplots(len(attribute_cols), figsize=(60, 20 * len(attribute_cols)))
pnan_attributes = {}
for i, attribute in enumerate(attribute_cols):
    # Get unique places
    places = data[place_col].unique()
    # Remove background
    places = np.delete(places, np.where(places=="background"))
    # Store differences and p-values
    differences = []
    p_values = []
    results = []
    new_places = places
    
    for place in places:
        # Filter data for the current place
        place_data = data[data[place_col] == place]
        
        # Separate the groups
        group_hfd = place_data[place_data[diet_col] == "hfd"][attribute]  
        group_chow = place_data[place_data[diet_col] == "chow"][attribute]

        if attribute == "Radius":
            group_hfd  *= 5.6 
            group_chow *= 5.6

        # Calculate the mean difference
        mean_diff = group_hfd.mean() - group_chow.mean()
        #print(f"{attribute} {place}\t\t {group_hfd.mean():.4f}\t\t {group_chow.mean():.4f}")
        if isNaN(mean_diff) or group_hfd.mean() == 0.0 or group_chow.mean() == 0.0:
            #print(f"{attribute}\t{place}\t:{group_hfd.mean()} - chow {group_chow.mean()} = {mean_diff}")
            new_places = np.delete(new_places, np.where(new_places == place))
        else:
            differences.append(mean_diff)
            
            # Perform t-test
            t_stat, p_val = ttest_ind(group_hfd, group_chow, equal_var=False, nan_policy="raise")  # Welch's t-test for unequal variances
            if isNaN(p_val):
                #print(f"{place} {attribute} {p_val} hfd {group_hfd.mean()} - chow {group_chow.mean()} ")
                #print(f"HFD \t{len(group_hfd)}")
                #print(f"chow \t{len(group_chow)}")
                if not attribute in pnan_attributes.keys():
                    pnan_attributes[attribute] = {}
                pnan_attributes[attribute][place] = f"{p_val} hfd {group_hfd.mean()} {len(group_hfd)} chow {group_chow.mean()} {len(group_chow)}"
            else:
                p_values.append(p_val)
                results.append((place, mean_diff, p_val))
            
    places = new_places
    # Calculate colors based on differences
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Unpack sorted results
    sorted_places, sorted_differences, sorted_p_values = zip(*results)
    
    intensities = np.abs(sorted_differences) / max(np.abs(sorted_differences))  # Normalize intensities
    color_map = [(*plt.cm.Blues(0.5 + 0.5 * intensity)[:-1],) if diff > 0 else (*plt.cm.Reds(0.5 + 0.5 * intensity)[:-1],)
                 for diff, intensity in zip(sorted_differences, intensities)]
    # Plotting
    axs[int(i/4)][i%4].bar(sorted_places, sorted_differences, color=color_map)
    axs[int(i/4)][i%4].set_title(f'Mean Difference in {attribute}')
    axs[int(i/4)][i%4].set_xlabel('Organ/Tissue')
    axs[int(i/4)][i%4].set_ylabel("Mean Difference")# (p-Value from Welch's t-test)")
    
    # Annotate with p-values
    
    for j, p in enumerate(sorted_p_values):
        fontweight = "normal" 
        if p < 0.05:
            fontweight = "heavy"
            print(f"Found significant p-value {p} {j}")
        p_star = ""
        if 0.05 >= p > 0.01:
            p_star = "*"
        elif 0.01 >= p > 0.001:
            p_star = "**"
        elif 0.001 > p:
            p_star = "***"
        if sorted_differences[j] > 0:
            axs[int(i/4)][i%4].text(j, sorted_differences[j], f'p={p:.3f}{p_star}', ha='center', color='black', rotation=45, fontweight=fontweight)
        else:
            axs[int(i/4)][i%4].text(j, sorted_differences[j], f'p={p:.3f}{p_star}', color='black', rotation=45, ha="right", va="top", rotation_mode="anchor", fontweight=fontweight)
       
    axs[int(i/4)][i%4].xaxis.set_tick_params(rotation=45)
    
axs[-1][-1].remove()
plt.tight_layout()
# plt.show()
plt.savefig(figure_path)
