from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=pipeline.named_steps["clf"].classes_, columns=pipeline.named_steps["clf"].classes_)
    return report_dict, cm_df