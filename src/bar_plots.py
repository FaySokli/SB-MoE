import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes
plt.rcParams.update({
    'font.size': 18,         # Base font size
    'axes.titlesize': 20,    # Title
    'axes.labelsize': 20,    # Axis labels
    'xtick.labelsize': 16,   # X-axis tick labels
    'ytick.labelsize': 16,   # Y-axis tick labels
    'legend.fontsize': 18,   # Legend
    'figure.titlesize': 20   # Figure title
})

# Data
categories = ['MSMARCO', 'TREC DL 19', 'TREC DL 20', 'NQ', 'HotpotQA', 'PS', 'CS']
num_groups = len(categories)
num_bars = 5

# Recall@100: shape (5 methods × 7 datasets)
values = [
    [0.682, 0.421, 0.521, 0.722, 0.465, 0.282, 0.327],  # FT
    [0.693, 0.426, 0.518, 0.732, 0.477, 0.294, 0.333],  # 3
    [0.688, 0.422, 0.516, 0.725, 0.477, 0.291, 0.332],  # 6
    [0.692, 0.413, 0.520, 0.724, 0.481, 0.291, 0.331],  # 9
    [0.690, 0.429, 0.514, 0.729, 0.476, 0.290, 0.328]   # 12
]

# NDCG@10: shape (5 methods × 7 datasets)
# values = [
#     [0.244, 0.439, 0.466, 0.261, 0.243, 0.145, 0.172],  # FT
#     [0.251, 0.462, 0.470, 0.268, 0.251, 0.150, 0.174],  # 3
#     [0.246, 0.438, 0.468, 0.271, 0.252, 0.149, 0.175],  # 6
#     [0.247, 0.458, 0.467, 0.273, 0.257, 0.150, 0.175],  # 9
#     [0.251, 0.455, 0.475, 0.273, 0.251, 0.148, 0.173]   # 12
# ]

bar_width = 0.15
x = np.arange(num_groups)
colors = ['#12436D', '#28A197', '#801650', '#F46A25', '#A285D1']
hatches = ['/', '\\', 'x', '-', '||']

plt.figure(figsize=(12, 6))
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(num_bars):
    plt.bar(x + i * bar_width, values[i], width=bar_width, label=f'experts {i*3}' if i > 0 else 'fine-tuned', color=colors[i], hatch=hatches[i], edgecolor='white', alpha=0.85)

# Labels and title
# plt.xlabel('Dataset')
plt.ylabel('Recall@100')
plt.xticks(x + bar_width * (num_bars - 1) / 2, categories, rotation=45)
plt.legend()
plt.tight_layout()

# Save
plt.savefig('../src/recall.png', dpi=900, format='png', bbox_inches='tight')