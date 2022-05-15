import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pylab import rcParams
import seaborn as sns
from sklearn import preprocessing


rcParams['figure.figsize'] = 8, 5
plt.rc('grid', linestyle="--", color='gray')

# https://learnui.design/tools/data-color-picker.html#palette
colors = ['#33508f', '#ff5d68', '#ffa600','#af4f9b']

results = pd.read_csv(r"H:\output_dir\val_patient_stats.csv")
results.head()


print("-- Segmentation scores --\n")
# print("Min Jaccard LV: {:.4f} / Max Jaccard LV: {:.4f}".format(results["Jaccard LV"].min(), results["Jaccard LV"].max()))
# print("Min Jaccard RV: {:.4f} / Max Jaccard RV: {:.4f}".format(results["Jaccard RV"].min(), results["Jaccard RV"].max()))
# print("Min Jaccard MYO: {:.4f} / Max Jaccard MYO: {:.4f}".format(results["Jaccard MYO"].min(), results["Jaccard MYO"].max()))

print()
print("Min Dice LV: {:.4f} / Max Dice LV: {:.4f}".format(results["Dice LV"].min(), results["Dice LV"].max()))
print("Min Dice RV: {:.4f} / Max Dice RV: {:.4f}".format(results["Dice RV"].min(), results["Dice RV"].max()))
print("Min Dice MYO: {:.4f} / Max Dice MYO: {:.4f}".format(results["Dice MYO"].min(), results["Dice MYO"].max()))

print()
print("Min Hausdorff LV: {:.4f} / Max Hausdorff LV: {:.4f}".format(results["Hausdorff LV"].min(), results["Hausdorff LV"].max()))
print("Min Hausdorff RV: {:.4f} / Max Hausdorff RV: {:.4f}".format(results["Hausdorff RV"].min(), results["Hausdorff RV"].max()))
print("Min Hausdorff MYO: {:.4f} / Max Hausdorff MYO: {:.4f}".format(results["Hausdorff MYO"].min(), results["Hausdorff MYO"].max()))

print()
print("Min ASSD LV: {:.4f} / Max ASSD LV: {:.4f}".format(results["ASSD LV"].min(), results["ASSD LV"].max()))
print("Min ASSD RV: {:.4f} / Max ASSD RV: {:.4f}".format(results["ASSD RV"].min(), results["ASSD RV"].max()))
print("Min ASSD MYO: {:.4f} / Max ASSD MYO: {:.4f}".format(results["ASSD MYO"].min(), results["ASSD MYO"].max()))


plt.rcParams.update({'font.size': 16})
fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(30, 22))

# -------------------------------------------------------------- #
results.groupby("Vendor")["Dice LV"].mean().plot.bar(color=colors, ax=ax1[0])
ax1[0].set_ylabel("DICE Score")
ax1[0].set_yticks(np.arange(0, results.groupby("Vendor")["Dice LV"].mean().max()+0.05, .05))
ax1[0].set_title("Mean DICE LV")
ax1[0].grid()
ax1[0].set_xticks( [ 0,1,2,3 ] )
ax1[0].set_xticklabels( ['A','B','C', 'D'], rotation=0 )
for p in ax1[0].patches:
    ax1[0].annotate(str(p.get_height())[:5], (p.get_x() + (p.get_width()/2) - (p.get_width()/4), p.get_height() * 1.01))

# -------------------------------------------------------------- #
results.groupby("Vendor")["Dice RV"].mean().plot.bar(color=colors, ax=ax1[1])
ax1[1].set_ylabel("DICE Score")
ax1[1].set_yticks(np.arange(0, results.groupby("Vendor")["Dice RV"].mean().max()+0.05, .05))
ax1[1].set_title("Mean DICE RV")
ax1[1].grid()
ax1[1].set_xticks( [ 0,1,2,3 ] )
ax1[1].set_xticklabels( ['A','B','C', 'D'], rotation=0 )
for p in ax1[1].patches:
    ax1[1].annotate(str(p.get_height())[:5], (p.get_x() + (p.get_width()/2) - (p.get_width()/4), p.get_height() * 1.01))

# -------------------------------------------------------------- #
results.groupby("Vendor")["Dice MYO"].mean().plot.bar(color=colors, ax=ax2[0])
ax2[0].set_ylabel("DICE Score")
ax2[0].set_yticks(np.arange(0, results.groupby("Vendor")["Dice MYO"].mean().max()+0.05, .05))
ax2[0].set_title("Mean DICE MYO")
ax2[0].grid()
ax2[0].set_xticks( [ 0,1,2,3 ] )
ax2[0].set_xticklabels( ['A','B','C', 'D'], rotation=0 )
for p in ax2[0].patches:
    ax2[0].annotate(str(p.get_height())[:5], (p.get_x() + (p.get_width()/2) - (p.get_width()/4), p.get_height() * 1.01))


# -------------------------------------------------------------- #
results.groupby("Vendor")[["Dice LV", "Dice RV", "Dice MYO"]].mean().mean(axis=1).plot.bar(color=colors, ax=ax2[1])
ax2[1].set_ylabel("DICE Score")
ax2[1].set_yticks(np.arange(0, results.groupby("Vendor")[["Dice LV", "Dice RV", "Dice MYO"]].mean().mean(axis=1).max()+0.05, .05))
ax2[1].set_title("Mean DICE Global")
ax2[1].grid()
ax2[1].set_xticks( [ 0,1,2,3 ] )
ax2[1].set_xticklabels( ['A','B','C', 'D'], rotation=0 )
for p in ax2[1].patches:
    ax2[1].annotate(str(p.get_height())[:5], (p.get_x() + (p.get_width()/2) - (p.get_width()/4), p.get_height() * 1.01))

plt.savefig('H:\dice_vendor.png', bbox_inches='tight', dpi=160)