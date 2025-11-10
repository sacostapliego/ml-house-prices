# quick evaluation snippet
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
preds = model.predict(X_val)
plt.scatter(y_val, preds, alpha=0.4)
plt.xlabel("True price"); plt.ylabel("Predicted price")
plt.title("True vs Pred")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')
plt.show()
print("R2:", r2_score(y_val, preds))
