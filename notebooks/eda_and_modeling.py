{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "de63ffce-2a2c-431d-8c5b-0a4cc0a065a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.51      0.54      0.52       925\n",
      "           1       0.51      0.48      0.50       934\n",
      "\n",
      "    accuracy                           0.51      1859\n",
      "   macro avg       0.51      0.51      0.51      1859\n",
      "weighted avg       0.51      0.51      0.51      1859\n",
      "\n",
      "Accuracy: 0.5099515868746638\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['attrition_model.pkl']"
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "import joblib\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"C:/Users/sathi/Downloads/HREmployee_data.csv\")\n",
    "\n",
    "# Encode categorical variables\n",
    "le = LabelEncoder()\n",
    "for col in df.select_dtypes(include='object').columns:\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "\n",
    "# Feature and target selection\n",
    "X = df.drop('Attrition', axis=1)  # Assuming 'Attrition' is the target column\n",
    "y = df['Attrition']\n",
    "\n",
    "# Train/test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train classifier\n",
    "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate\n",
    "y_pred = clf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "\n",
    "# Save model\n",
    "joblib.dump(clf, 'attrition_model.pkl')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53044f31-2857-4514-8cf4-7793578bab5b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
