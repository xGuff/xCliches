import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import numpy as np

# Matplotlib settings
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[cm]{sfmath}\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'cm',
    'font.size': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})
plt.style.use('tableau-colorblind10')

# Paths
input_path = "data/processed/favourite_cliches.csv"
output_path = "data/outputs/wordcloud.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(input_path)

# Aggregate total usage across all clubs
overall_freq = df.groupby("cliche")["count"].sum().to_dict()

# Normalize and get colormap
max_freq = max(overall_freq.values())
cmap = plt.get_cmap("plasma")

# Define color function
def frequency_color_func(word, font_size, position, orientation, font_path, random_state):
    freq = overall_freq.get(word, 0)
    normalized = freq / max_freq if max_freq > 0 else 0
    r, g, b, _ = [int(255 * v) for v in cmap(1 - normalized)]  # Flip for high freq = dark
    return f"rgb({r},{g},{b})"

# Generate word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    color_func=frequency_color_func,
    prefer_horizontal=1.0
).generate_from_frequencies(overall_freq)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.1)

# Show wordcloud
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")

# Color bar with matching height
norm = Normalize(vmin=1, vmax=max_freq)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="plasma_r"), cax=cax)
cb.set_label("Cliché Count")
cb.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(output_path, dpi=500)
plt.close()

print(f"✅ Word cloud with full-height color bar saved to {output_path}")
