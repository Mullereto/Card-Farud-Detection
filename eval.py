from prepare_data import *
from data_saver import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve


def eval_with_thershold(model, x, thershold=0.5):
    """evaluate the model using Default thershold

    Args:
        model (model): your trained model
        x (DataFrame): yout Test Data
        thershold (float, optional): the thershold you want to use. Defaults to 0.5.

    Returns:
        nparray: the prdicted data
    """
    y_pred_propa = model.predict_proba(x)
    y_pred = (y_pred_propa[:,1]>= thershold).astype(int) #for class 1
    return y_pred

def eval_with_optimal(model, x, y):
    """evaluate the model using best thershold

    Args:
        model (model): the model you trained
        x (the target): the tagert variabl
    """
    probs = model.predict_proba(x)[:,1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
    
def eval_with_Confusion_matrix_classification_report(y_pred, y_true, title=""):
    """copute the confusion matrix and the the image

    Args:
        y_pred (nparray): the predectid values
        y_true (nparray): the ground truth
        title (str, optional): name oof the model the has been used. Defaults to "".
    """
    print(f'{title} Classification Report')
    print(classification_report(y_pred=y_pred, y_true=y_true))   
    stateOFmodel = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
    
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for "+title)
    save_img("Confusion Matrix for "+title)
    plt.show()

    return stateOFmodel

def eval_with_PR_AUC(modle, X, y_true, title=""):
    """calculate the PR-AUC and draw it

    Args:
        modle (model): the model you tranied
        X (DataFrame): thedataset
        y_true (ndarray): the ground truth
        title (str, optional): the name of the model. Defaults to "".
    """
    probs = modle.predict_proba(X)[:,1]
    precision, recall, _ = precision_recall_curve(y_true, probs)
    PR_AUC = auc(recall, precision)
    print(f"PR-AUC: {PR_AUC:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', label=f'PR Curve (AUC = {PR_AUC:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve for ' + title)
    plt.legend()
    plt.grid()
    save_img('Precision-Recall (PR) Curve for ' + title)
    plt.show()
    return PR_AUC
    
def eval_with_ROC_curv(modle, x, y_true, title=""):
    probs = modle.predict_proba(x)[:, 1]
    roc_auc = roc_auc_score(y_true, probs)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    precision, recall, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for '+title)
    plt.legend()
    plt.grid()
    save_img('Receiver Operating Characteristic (ROC) Curve for '+title)
    plt.show()
    
    return roc_auc

def eval_precision_recall_for_different_threshold(model, x, y_true, title=""):
    probs = model.predict_proba(x)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label='Precision', marker='.')
    plt.plot(thresholds, recall[:-1], label='Recall', marker='.')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall for different Thresholds for '+title)
    plt.legend('Precision and Recall for different Thresholds for '+title)
    save_img('Precision and Recall for different Thresholds for '+title)
    plt.show()



# def eval_train(model_b, x_train, y_train, models_dict_compare:dict, model_name=""):
#     y_pred = model_b.predict(x_train)
#     model_state = eval_with_Confusion_matrix_classification_report(y_pred, y_train)
#     eval_with_PR_AUC(model_b, x_train, y_train, model_name + " Train")
#     eval_precision_recall_for_different_threshold(model_b, x_train, y_train, model_name + " Train")
#     return model_state

def eval_valdtion(model_b, x_val, y_val, models_dict_compare:dict, model_name="", dataset_title=""):
    y_pred = model_b.predict(x_val)
    print("With default thershold to 0.5 ")
    model_state = eval_with_Confusion_matrix_classification_report(y_pred, y_val , model_name+dataset_title)
    pr_auc = eval_with_PR_AUC(model_b, x_val, y_val, model_name + dataset_title)
    roc = eval_with_ROC_curv(model_b, x_val, y_val, model_name + dataset_title)
    print(f"model PR-AUC = {pr_auc} and ROC = {roc} ")
    optimal_threshold = eval_with_optimal(model_b, x_val, y_val)
    print(f"With optimal thershold to {optimal_threshold}")
    y_pred = eval_with_thershold(model_b, x_val, optimal_threshold)
    model_state = eval_with_Confusion_matrix_classification_report(y_pred, y_val,model_name+ dataset_title)
    eval_precision_recall_for_different_threshold(model_b, x_val, y_val, model_name + dataset_title)
    return model_state, pr_auc, roc

# def eval_test(model_b, x_test, y_test, models_dict_compare:dict, model_name=""):
#     y_pred = model_b.predict(x_test)
#     print("With default thershold to 0.5 ")
#     model_state = eval_with_Confusion_matrix_classification_report(y_pred, y_test)
#     eval_with_PR_AUC(model_b, x_test, y_test, model_name + " Testing")
    
#     optimal_threshold = eval_with_optimal(model_b, x_test, y_test)
#     print(f"With optimal thershold to {optimal_threshold}")
#     y_pred = eval_with_thershold(model_b, x_test, optimal_threshold)
#     model_state = eval_with_Confusion_matrix_classification_report(y_pred, y_test,model_name+ " Testing")
#     eval_precision_recall_for_different_threshold(model_b, x_test, y_test, model_name + " Testing")
#     return model_state