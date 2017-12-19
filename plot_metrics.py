import pickle
import matplotlib.pyplot as plt

dev_metrics_path = "models/transfer1/dev_metrics/"

with open(dev_metrics_path + "auc", "rb") as f:
    auc = pickle.load(f)

plt.plot(auc)
plt.show()

# with open(dev_metrics_path + "map", "rb") as f:
#     maps = pickle.load(f)

# with open(dev_metrics_path + "mrr", "rb") as f:
#     mrr = pickle.load(f)

# with open(dev_metrics_path + "p1", "rb") as f:
#     p1 = pickle.load(f)

# with open(dev_metrics_path + "p5", "rb") as f:
#     p5 = pickle.load(f)  


# plt.plot(maps)
# plt.plot(mrr)
# plt.plot(p1)
# plt.plot(p5)

# plt.show()
