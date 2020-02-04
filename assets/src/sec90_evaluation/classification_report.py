from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print(classification_report(y_true, y_pred))
