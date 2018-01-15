from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mdp = pd.read_csv("mdp.csv")
mmdp = pd.read_csv("mmdp.csv")

all = pd.concat([mdp.iloc[:,1]+1,mmdp.iloc[:,1]], axis=1)
all.columns = ["mdp", "mmdp"]
all = pd.concat([pd.DataFrame([{"mdp":1, "mmdp":1}]), all])
print(all)
all = pd.rolling_apply(all,30,stats.gmean,min_periods=1)
all = all.iloc[:-200:30,:]
all.index = all.index*100
print(all)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 11}

mpl.rc('font', **font)
plt.subplot()
plt.plot(all.mmdp, label="Reduced MMDP",
             linestyle="-", marker="o", color="black")
# plt.fill_between(df_m.index, df_m["min"], df_m["max"],
#                 alpha=0.3, facecolor='#089FFF',
#                 linewidth=0)
plt.plot(all.mdp, label="MDP",
             linestyle="--", marker="v", color="black", mfc='none', markersize=10)
# plt.fill_between(df_a.index, df_a["min"], df_a["max"],
#                 alpha=0.3, facecolor='#FF9848',
#                 linewidth=0)
plt.xlabel("epochs", font)
plt.ylabel("final capital", font)
plt.tight_layout()
plt.legend(loc="best")
plt.show()
