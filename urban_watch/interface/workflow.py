# import os
# import requests
# from prefect import task, flow
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score

# from urban_watch.interface.main import full_preproc_pipeline
# from urban_watch.ml_logic.registry import save_model, mlflow_transition_model



# def preprocess_data():
#     X_preproc, y_preproc = full_preproc_pipeline()
#     return X_preproc, y_preproc

# def train_model(X_train, y_train):
#     model = LogisticRegression(
#         C=1.0,
#         solver='lbfgs',
#         max_iter=1000,
#         random_state=42
#     )
#     model.fit(X_train, y_train)
#     return model


# def evaluate_model(model, X_test, y_test):
#     y_pred=model.predict(X_test)
#     f1 = f1_score(y_test,y_pred)
#     try:
#         roc= roc_auc_score(y_test,model.predict_proba(X_test))
#     except:
#         roc=None
#     return f1, roc


# def transition_model_if_better(model_f1, baseline_f1):
#     if model_f1 > baseline_f1:
#         mlflow_transition_model(current_stage="Staging", new_stage="Production")
#     else:
#         print("notbetter")

# def save_model_task(model):
#     save_model(model, model_name="logistic_reg")


# def notify(f1_score_new, f1_score_old):
#     content = f"old{f1_score_old}, new{f1_score_new}"

# def urban_watch_flow():
#     X, y = preprocess_data()


#     split_ratio = 0.8
#     idx_split = int(X.shape[0] * split_ratio)
#     X_train, X_test = X[:idx_split], X[idx_split:]
#     y_train, y_test = y[:idx_split], y[idx_split:]

#     model_task = train_model(X_train, y_train)
#     model = model_task()

#     f1, roc = evaluate_model(model, X_test, y_test)

#     baseline_f1 = 0.8

#     transition_model_if_better(f1,baseline_f1)

#     save_model_task(model)

#     notify(f1, baseline_f1)

# if __name__ == "__main__":
#     urban_watch_flow()
