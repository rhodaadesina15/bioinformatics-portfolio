from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
B=Path(__file__).resolve().parent; R=B/"results"; R.mkdir(exist_ok=True)
d=pd.read_csv(B/"data/simulated_eeg_summary.csv")
d["alpha_change"]=d.task_alpha_power-d.rest_alpha_power
d["beta_change"]=d.task_beta_power-d.rest_beta_power
s=pd.DataFrame({
"measure":["Alpha power","Beta power"],
"rest_mean":[d.rest_alpha_power.mean(),d.rest_beta_power.mean()],
"task_mean":[d.task_alpha_power.mean(),d.task_beta_power.mean()],
"mean_change":[d.alpha_change.mean(),d.beta_change.mean()],
"paired_t_pvalue":[ttest_rel(d.task_alpha_power,d.rest_alpha_power).pvalue,
ttest_rel(d.task_beta_power,d.rest_beta_power).pvalue]})
s.to_csv(R/"eeg_summary_results.csv",index=False)
s.set_index("measure")[["rest_mean","task_mean"]].plot(kind="bar")
plt.ylabel("Mean power (arbitrary units)"); plt.title("Simulated EEG: rest vs task")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(R/"eeg_rest_vs_task.png",dpi=200); plt.close()
print(s.round(4))
