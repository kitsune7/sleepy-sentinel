import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/11/10.csv')
df['time_ms'] = df.index * 15

exclude_cols = ['frame_idx', 'time_ms', 't_ms', 'bright_mean', 'face', 'warmth']
plot_cols = [c for c in df.columns if c not in exclude_cols]

stds = df[plot_cols].std()
constant_cols = stds[stds == 0].index.tolist()
if constant_cols:
  plot_cols = [c for c in df.columns if c not in constant_cols]
df_norm = (df[plot_cols] - df[plot_cols].mean()) / df[plot_cols].std()


fig, ax = plt.subplots(figsize=(14,6))
for col in plot_cols:
    ax.plot(df['time_ms'], df_norm[col], label=col, linewidth=0.8)

ax.set_xlabel('Time (ms)')
ax.set_ylabel('Value')
ax.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
