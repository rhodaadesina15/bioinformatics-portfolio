from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
B=Path(__file__).resolve().parent; R=B/"results"; R.mkdir(exist_ok=True)
d=pd.read_csv(B/"data/simulated_biomedical_classification.csv"); X=d.drop(columns="disease_status"); y=d.disease_status
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
models={"Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000))]),
"Random Forest":RandomForestClassifier(n_estimators=250,random_state=42)}
rows=[]
for name,m in models.items():
 m.fit(Xt,yt); pred=m.predict(Xv); proba=m.predict_proba(Xv)[:,1]
 rows.append({"model":name,"accuracy":accuracy_score(yv,pred),"precision":precision_score(yv,pred,zero_division=0),
 "recall":recall_score(yv,pred,zero_division=0),"f1":f1_score(yv,pred,zero_division=0),"roc_auc":roc_auc_score(yv,proba)})
pd.DataFrame(rows).to_csv(R/"model_metrics.csv",index=False)
rf=models["Random Forest"]; imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values()
plt.figure(figsize=(7,4)); plt.barh(imp.index,imp.values); plt.xlabel("Feature importance")
plt.title("Random Forest feature importance — simulated data"); plt.tight_layout()
plt.savefig(R/"random_forest_feature_importance.png",dpi=200); plt.close()
print(pd.DataFrame(rows).round(3))
