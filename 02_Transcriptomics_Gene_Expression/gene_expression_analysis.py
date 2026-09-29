from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
B=Path(__file__).resolve().parent; R=B/"results"; R.mkdir(exist_ok=True)
d=pd.read_csv(B/"data/simulated_gene_expression.csv")
d["fold_change"]=(d.case_expression+1e-9)/(d.control_expression+1e-9)
d["log2_fold_change"]=np.log2(d.fold_change)
d["abs_log2_fold_change"]=d.log2_fold_change.abs()
d["large_expression_change"]=d.abs_log2_fold_change>=1
ranked=d.sort_values("abs_log2_fold_change",ascending=False)
ranked.to_csv(R/"ranked_gene_expression_results.csv",index=False)
top=ranked.head(15).sort_values("log2_fold_change")
plt.figure(figsize=(8,6)); plt.barh(top.gene,top.log2_fold_change)
plt.xlabel("log2 fold change"); plt.title("Top simulated gene-expression changes")
plt.tight_layout(); plt.savefig(R/"top_gene_changes.png",dpi=200); plt.close()
print(ranked[["gene","log2_fold_change","large_expression_change"]].head(10))
