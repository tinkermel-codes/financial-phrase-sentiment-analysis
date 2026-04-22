from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)


    report = classification_report(
        y_test,
        preds
    )

    print(report)


    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()

    return report