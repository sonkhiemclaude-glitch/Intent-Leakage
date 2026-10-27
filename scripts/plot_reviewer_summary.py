"""Plot released aggregate statistics; no fitting or private inputs are used.

The capacity plot shows class maxima, not the private per-exchange ECDF.
Individual assurance timelines cannot be reconstructed from these aggregates.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=ROOT / 'data/reviewer_summary.json')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
data = json.loads(args.input.read_text())
blue, teal, coral = '#167AAF', '#198D77', '#C95D55'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.edgecolor': '#BFD4E1', 'text.color': '#183348',
    'axes.labelcolor': '#183348', 'xtick.color': '#647B8D', 'ytick.color': '#647B8D',
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(args.output / (name + '.' + ext), bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.9), sharey=True)
for ax, target in zip(axes, ('class', 'scope')):
    rows = [r for r in data['tree_results'] if r['target'] == target]
    for y, row in enumerate(rows):
        color = blue if row['cohort'] == 'mcp' else teal
        score, ref = 100 * row['balanced_accuracy'], 100 * row['archived_full_rf_ba_same_split']
        ax.plot([score, ref], [y, y], color=color, alpha=.3, linewidth=2)
        ax.scatter(score, y, marker='s', s=35, color=color, zorder=3)
        ax.scatter(ref, y, marker='o', s=35, facecolor='white', edgecolor=color, zorder=3)
        ax.text(score, y-.23, f'{score:.1f}', ha='center', fontsize=8, color=color)
        ax.text(ref, y+.32, f'{ref:.1f}', ha='center', fontsize=8, color=color)
    ax.axvline(100/3, color='#8D9CA7', linestyle='--', linewidth=.8)
    ax.axhline(2.5, color='#D8E3E9', linewidth=.8)
    ax.set(title='Operation class' if target == 'class' else 'Target-count scope',
           xlim=(28, 105), ylim=(5.65, -.65), xlabel='Balanced accuracy (%)')
axes[0].set_yticks(range(6), ['MCP / tool', 'MCP / assurance', 'MCP / combined',
                            'Native / tool', 'Native / assurance', 'Native / combined'])
fig.legend(handles=[Line2D([], [], marker=m, color='#647B8D', markerfacecolor=f,
                          linestyle='none', label=l) for m, f, l in
                    [('s', '#647B8D', 'Count-only tree (smoke)'), ('o', 'white', 'Archived full-feature forest')]],
           loc='lower center', ncol=2, frameon=False, fontsize=8)
fig.subplots_adjust(left=.21, right=.98, top=.91, bottom=.21, wspace=.19)
save(fig, 'smoke_count_baseline')

cmap = LinearSegmentedColormap.from_list('leakage', ['#F3F6F8', '#ADD8DD', blue, '#183348'])
fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.75))
for ax, target in zip(axes, ('class', 'scope')):
    z = np.array([[next(r['balanced_accuracy'] for r in data['transfer_results']
                       if (r['cohort'], r['view'], r['target']) == (c, v, target)) * 100
                   for v in ('ct', 'ca', 'ctca')] for c in ('mcp', 'native')])
    ax.imshow(z, vmin=100/3, vmax=100, cmap=cmap, aspect='auto')
    for y in range(2):
        for x in range(3):
            ax.text(x, y, f'{z[y,x]:.2f}%', ha='center', va='center', fontsize=11,
                    color='white' if z[y,x] > 78 else '#183348', fontweight='bold')
    ax.set_xticks(range(3), ['Tool', 'Assurance', 'Combined'])
    ax.set_yticks([0, 1], ['MCP-trained', 'Native-trained'])
    ax.set_title('Operation class' if target == 'class' else 'Target-count scope', pad=12)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)
axes[1].tick_params(labelleft=False)
fig.text(.55, .03, 'Same 36 archived direct-driver runs; stored models; no refitting', ha='center', fontsize=8)
fig.subplots_adjust(left=.18, right=.99, top=.78, bottom=.22, wspace=.16)
save(fig, 'archive_direct_transfer')

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout='constrained')
body = data['body_summary']
values = [r['max_response_bytes'] / 1024 for r in body]
axes[0].barh(['Quarantine', 'Release', 'Audit'], values, color=[blue, coral, teal], height=.45)
for y, value in enumerate(values):
    axes[0].text(value - 1 if value > 10 else value + 1, y, f'{value:.2f}',
                 va='center', ha='right' if value > 10 else 'left', fontsize=8,
                 color='white' if value > 10 else '#183348')
axes[0].axvline(64, linestyle='--', color=coral, linewidth=1)
axes[0].set(xlim=(0, 85), xlabel='Maximum response body (KiB)', title='Aggregate capacity maxima')
axes[0].invert_yaxis()
for y, row in enumerate(data['scheduler_summary']):
    color = coral if row['loaded'] else blue
    axes[1].plot([row['median_ms'], row['max_ms']], [y, y], color=color, alpha=.3)
    for key, marker in [('median_ms', 'o'), ('p95_ms', 's'), ('max_ms', '|')]:
        axes[1].scatter(row[key], y, marker=marker, s=35, color=color)
axes[1].axvline(10, linestyle='--', color='#647B8D', linewidth=.8)
axes[1].set_yticks(range(4), ['Thread / idle', 'Thread / load', 'Process / idle', 'Process / load'])
axes[1].set(xlim=(0, 60), ylim=(3.5, -.5), xlabel='Sender-write lateness (ms)', title='Local synthetic-handler stress')
axes[1].legend(handles=[Line2D([], [], color='#647B8D', marker=m, linestyle='none', label=l)
                        for m, l in [('o', 'Median'), ('s', 'p95'), ('|', 'Max')]],
               loc='upper center', bbox_to_anchor=(.5, -.23), ncol=3, frameon=False, fontsize=8)
save(fig, 'archive_engineering_aggregates')
