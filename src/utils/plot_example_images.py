import matplotlib.pyplot as plt
import os

from matplotlib import cm

def plot_pie_chart(class_dis, class_names, colors_list):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(class_dis, labels=class_names, autopct='%1.1f%%', colors=colors_list, wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 20})
    plt.show()

def plot_images(root_path, class_names):
    fig, axes = plt.subplots(1, 11, figsize=(25, 5))  # Create a 1x11 grid of subplots
    n = 11
    for i in range(n):  # Iterate over all subplots
        if i < len(class_names):  # If there is an image for this subplot
            class_name = class_names[i]
            img = plt.imread(os.path.join(root_path, class_name, os.listdir(os.path.join(root_path, class_name))[0]))
            axes[i].imshow(img)
            axes[i].set_title(class_name, fontsize=20)
            axes[i].axis('off')
    plt.tight_layout()  # Adjust the padding between and around the subplots
    plt.show()

# Usage
root_path = r"C:\Users\Anwender\Desktop\Nicolas\Dokumente\FH Bielefeld\Optimierung und Simulation\2. Semester\SimulationOptischerSysteme\AI-Weather-Classification\dataset"
class_names = sorted(os.listdir(root_path))
n_classes = len(class_names)
class_dis = [len(os.listdir(os.path.join(root_path, class_name))) for class_name in class_names]
colors = cm.get_cmap('tab10', n_classes)
colors_list = colors.colors.tolist()

#plot_pie_chart(class_dis, class_names, colors_list)
plot_images(root_path, class_names)