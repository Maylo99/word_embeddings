from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import export_graphviz
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
from sklearn import metrics
class DecisionTree:

    def __init__(self,dataset,test_dataset,accounts,document_type):
        self.dataset=dataset
        self.test_dataset=test_dataset
        self.accounts=accounts
        self.document_type=document_type


    def _tree_graph_to_png(self,tree, feature_names, png_file_to_save):
        tree_str = export_graphviz(tree, feature_names=feature_names,
                                   filled=True, out_file=None)
        graph = pydotplus.graph_from_dot_data(tree_str)
        graph.write_png(png_file_to_save)

    def _search_missing_accounts(self,input, missing_hash, predicated_account):
        if predicated_account in input:
            input.remove(predicated_account)
        else:
            missing_hash[str(predicated_account)] = predicated_account

    def _search_for_replacement(self,input, new_hash, missing_accounts):
        for value_array in input:
            closest_key = None
            min_difference = float('inf')
            # Find the closest key in the dictionary
            for key, value in missing_accounts.items():
                difference = abs(value_array - value)
                if difference < min_difference and value in missing_accounts.values():
                    closest_key = key
                    min_difference = difference
            # Swap values
            if closest_key is not None:
                del missing_accounts[closest_key]
                new_hash[closest_key] = value_array

    def _json_response_format(self,d,md):
        md_output = {str(value): "md" for value in md}
        d_output = {str(value): "d" for value in d}
        combined_output = {**md_output, **d_output}
        output_json = json.dumps(combined_output)
        return output_json

    def _update_outputs(self,md, d, hash):
        for key, value in hash.items():
            if int(key) in md:
                # Find the index of the key in the array
                index = md.index(int(key))
                # Update the array with the corresponding value
                md[index] = value
            if int(key) in d:
                # Find the index of the key in the array
                index = d.index(int(key))
                # Update the array with the corresponding value
                d[index] = value
            # Create a dictionary containing both arrays
            return self._json_response_format(d,md)

    def _test_tree(self,model,output_column, X_test,y_test):
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Get unique classes present in the test data
        unique_classes = np.unique(np.concatenate((y_test, y_pred)))

        # Create a mapping of class indices to class names
        class_mapping = {i: class_name for i, class_name in enumerate(unique_classes)}
        accuracy_md = accuracy_score(y_test, y_pred)
        print("Presnosť Rozhodovacieho stromu:"+output_column, accuracy_md)

        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.colorbar()

        # Add annotations
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

        # Set tick labels to actual class values using the mapping
        plt.xticks(np.arange(len(conf_matrix)), labels=[class_mapping[i] for i in range(len(conf_matrix))])
        plt.yticks(np.arange(len(conf_matrix)), labels=[class_mapping[i] for i in range(len(conf_matrix))])

        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title("Confusion Matrix - Rozhodovací strom"+ output_column)
        plt.show()

    def train_model(self):
        data = pd.read_csv(self.dataset, sep=';')
        label_encoder = LabelEncoder()
        data['document_type'] = label_encoder.fit_transform(data['document_type'])
        df = pd.DataFrame(data)
        X = df[['document_type', 'account1', 'account2', 'account3']]
        y_md = df['MD']
        y_d = df['D']
        X_train_md, X_test_md, y_train_md, y_test_md = train_test_split(X, y_md, test_size=0.15, random_state=42)
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_d, test_size=0.15, random_state=42)

        # Decision tree for predicting MD
        clf_md = DecisionTreeClassifier()
        clf_md.fit(X_train_md, y_train_md)
        joblib.dump(clf_md, 'decision_tree_model_md.pkl')
        plt.figure(figsize=(20, 10))
        plot_tree(clf_md, feature_names=list(X.columns), rounded=True,filled=False, fontsize=12,max_depth=2)
        plt.title("Rozhodovací strom pre MD")
        plt.show()
        #Confusion matrix
        self._test_tree(clf_md,"MD",X_test_md,y_test_md)

        # Decision tree for predicting MD
        clf_d = DecisionTreeClassifier()
        clf_d.fit(X_train_d, y_train_d)
        joblib.dump(clf_d, 'decision_tree_model_d.pkl')
        plt.figure(figsize=(20, 10))
        plot_tree(clf_d, feature_names=list(X.columns), rounded=True,filled=False, fontsize=11, max_depth=3)
        plt.title("Rozhodovací strom pre D")
        plt.show()
        #Confusion matrix
        self._test_tree(clf_d, "D",X_test_d,y_test_d)

        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        my_dict_converted = {str(key): int(value) for key, value in label_mapping.items()}
        with open("label_mapping.json", "w") as json_file:
            json.dump(my_dict_converted, json_file)

    def make_prediction(self):
        clf_d = joblib.load('decision_tree_model_d.pkl')
        clf_md = joblib.load('decision_tree_model_md.pkl')
        with open("label_mapping.json", "r") as json_file:
            label_mapping_loaded = json.load(json_file)
        document_type = label_mapping_loaded.get(self.document_type)
        input=self.accounts
        new_data = {
            'document_type': document_type,
            'account1': self.accounts[0],
            'account2': self.accounts[1],
            'account3': self.accounts[2]
        }

        # Convert input data into DataFrame
        new_df = pd.DataFrame(new_data, index=[0])

        # Making prediction
        prediction_d = clf_d.predict(new_df)
        prediction_md = clf_md.predict(new_df)
        string_md_prediction = str(prediction_md[0])
        string_d_prediction = str(prediction_d[0])
        missing_accounts = {}

        if len(string_md_prediction) >= 4:
            md_account1 = int(string_md_prediction[:3])
            md_account2 = int(string_md_prediction[-3:])
            self._search_missing_accounts(input, missing_accounts, md_account1)
            self._search_missing_accounts(input, missing_accounts, md_account2)
            md_output = [md_account1, md_account2]
        else:
            self._search_missing_accounts(input, missing_accounts, prediction_md[0])
            md_output = prediction_md.tolist()

        if len(string_d_prediction) >= 4:
            d_account1 = int(string_d_prediction[:3])
            d_account2 = int(string_d_prediction[-3:])
            self._search_missing_accounts(input, missing_accounts, d_account1)
            self._search_missing_accounts(input, missing_accounts, d_account2)
            d_output = [d_account1, d_account2]
        else:
            self._search_missing_accounts(input, missing_accounts, prediction_d[0])
            d_output = prediction_d.tolist()
        new_hash = {}
        self._search_for_replacement(input, new_hash, missing_accounts)
        if len(new_hash)!=0:
            return self._update_outputs(md_output, d_output, new_hash)
        else:
            return self._json_response_format(d_output,md_output)