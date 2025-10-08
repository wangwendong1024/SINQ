import matplotlib
matplotlib.use('TkAgg')  # Windows GUI 后端
import matplotlib.pyplot as plt

# ==============================
# 手动记录实验结果
# 格式： (nbits, perplexity, memory_GB)
# ==============================
results = [
    (16, 16.5, 5.6),
    (8, 16.9, 2.8),
    (3, 22.39, 1.28),
    (4, 17.14, 1.42),
    (5, 17.20, 1.65)
]

# ==============================
# 按 nbits 排序
# ==============================
results.sort(key=lambda x: x[0])  # nbits 从小到大

nbits = [r[0] for r in results]
ppl = [r[1] for r in results]
mem = [r[2] for r in results]

# ==============================
# 绘制双轴图
# ==============================
fig, ax1 = plt.subplots(figsize=(8, 5))

# 左轴: Perplexity
ax1.plot(nbits, ppl, marker='o', color='tab:blue', label='Perplexity', linewidth=2)
for x, y in zip(nbits, ppl):
    ax1.text(x, y+0.2, f"{y:.2f}", ha='center', color='tab:blue')
ax1.set_xlabel('Quantization Bits (nbits)')
ax1.set_ylabel('Perplexity', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.invert_xaxis()  # 位数越低越靠右

# 右轴: GPU Memory
ax2 = ax1.twinx()
ax2.plot(nbits, mem, marker='s', linestyle='--', color='tab:red', label='Memory', linewidth=2)
for x, y in zip(nbits, mem):
    ax2.text(x, y+0.05, f"{y:.2f}", ha='center', color='tab:red')
ax2.set_ylabel('GPU Memory (GB)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# ==============================
# 美化
# ==============================
plt.title('Quantization vs Performance (Qwen3-1.7B)')
plt.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
plt.show()
