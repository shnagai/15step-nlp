from sklearn.metrics import accuracy_score

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print(accuracy_score(y_true, y_pred))
