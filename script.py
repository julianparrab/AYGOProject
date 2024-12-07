
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
  clf = joblib.load(os.path.join(model_dir, "model.joblib"))
  return clf

if __name__ == "__main__":
  print("[INFO] Extracting arguments")
  parser = argparse.ArgumentParser()

  # Hyperparameters
  parser.add_argument("--n_estimators", type=int, default=100)
  parser.add_argument("--random_state", type=int, default=0)

  # Directorios
  parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
  parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
  parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
  parser.add_argument("--train-file", type=str, default="train-V-1.csv")
  parser.add_argument("--test-file", type=str, default="test-V-1.csv")
  args, _ = parser.parse_known_args()

  #Versiones de scikit-learn y joblib
  print("Scikit-learn Version:", sklearn.__version__)
  print("Joblib Version:", joblib.__version__)

  # Carga los datos de entrenamiento y prueba desde los archivos CSV especificados
  print("[INFO] Reading data")
  train_df = pd.read_csv(os.path.join(args.train, args.train_file))
  test_df = pd.read_csv(os.path.join(args.test, args.test_file))

  # Obtiene las columnas
  features = list(train_df.columns)
  label = features.pop(-1)

  # Conjuntos de entrenamiento y prueba
  print("Building training and testing datasets")
  X_train = train_df[features]
  X_test = test_df[features]
  y_train = train_df[label]
  y_test = test_df[label]

  print("Training Data")
  print(X_train.shape)
  print(y_train.shape)
  print()

  print("Testing Data")
  print(X_test.shape)
  print(y_test.shape)
  print()

  print("Training RandomForest Model")
  model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
  model.fit(X_train, y_train)


  model_path = os.path.join(args.model_dir, "model.joblib")
  joblib.dump(model, model_path)


  y_pred_test = model.predict(X_test)
  test_acc = accuracy_score(y_test, y_pred_test)
  test_rep = classification_report(y_test, y_pred_test)


  print("RESULTS FOR TESTING DATA")
  print("Total Rows are: ", X_test.shape[0])
  print("[TESTING] Model Accuracy is: ", test_acc)
  print("[TESTING] Testing Report:")
  print(test_rep)
