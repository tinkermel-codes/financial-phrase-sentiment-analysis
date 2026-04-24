from sklearn.metrics import classification_report

def evaluate_model(pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    report_dict = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))
    return report_dict