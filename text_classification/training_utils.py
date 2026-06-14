from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
    # scikit-learn >= 1.5 requires `classes` to be a numpy ndarray, not a list.
    classes = np.array(sorted(df['label'].unique().tolist()))
    class_weights = compute_class_weight("balanced",
                         classes=classes,
                         y=df['label'].tolist()
                         )
    return class_weights