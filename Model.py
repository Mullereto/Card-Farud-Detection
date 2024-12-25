from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from eval import *
from data_saver import *


saver = {}

def do_logistic(x, y, dataset_title=""):
    model = LogisticRegression(penalty='elasticnet',C=1.0,class_weight={0: 1, 1: 10},solver='saga', max_iter=1000, l1_ratio=0.5,n_jobs=-1)
    model.fit(x,y)
    model_report, pr_auc, roc = eval_valdtion(model, x, y, saver, "LogisticRegression ", "Train")
    save_model_to_pkl(model, "LogisticRegression")
    return model ,model_report, pr_auc, roc

def do_RFC(x,y,data_title=""):#class_weight={0:0.5,1:5},n_jobs=-1
    model = RandomForestClassifier(criterion='log_loss', max_depth=100, max_features='sqrt', class_weight={0:0.5,1:5},n_jobs=-1)
    model.fit(x,y)
    model_report, pr_auc, roc = eval_valdtion(model, x, y, saver, "RandomForestClassifier ", "Train")
    save_model_to_pkl(model, "RandomForestClassifier")
    return model ,model_report, pr_auc, roc


if __name__ == '__main__':
    x_trin, y_train, x_val, y_val = load_data("split/")
    x_trin_scaled, x_val_scaled = do_the_scale(x_trin, x_val, sclaer="power")
    x_trin_scaled, y_train = solve_imbalance(x_trin_scaled, y_train, technique="smotetomek")
    
    
    model,_,_,_ = do_RFC(x_trin_scaled, y_train, "Train")
    #x_val_scaled, y_val = solve_imbalance(x_val_scaled, y_val, technique="smotetomek")
        
    model_report, pr_auc, roc = eval_valdtion(model, x_val, y_val, saver, "RandomForestClassifier ", "validation")