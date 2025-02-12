import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('Djerba_Tourist_Activities_Large.csv')


df_encoded = pd.get_dummies(df, columns=['Tourist_Type', 'Interest_Category', 'Budget', 'Season', 'Duration_of_Stay', 'Accessibility'])


X = df_encoded.drop(columns='Recommended_Activity')
y = df_encoded['Recommended_Activity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


st.title('Tourist Activity Recommendation System for Djerba Using Machine Learning')
st.header("Get a Place to Visit or Activity Recommendation Based on Your Preferences")


Tourist_Type = st.selectbox("Select your type of group", ["Solo", "Couple", "Family", "Group"])
Interest_Category = st.selectbox('Select your category', ['Relaxation', 'Adventure', 'Culture'])
Budget = st.selectbox('Select your budget', ['Low', 'Medium', 'High'])
Season = st.selectbox('Select your season', ['Spring', 'Fall', 'Summer', 'Winter'])
Duration_of_Stay = st.selectbox('Select your duration of stay', ['Short', 'Medium', 'Long'])
Accessibility = st.selectbox('Select your accessibility', ['Walking', 'Public Transport', 'Car Rental'])


user_input = {
    'Tourist_Type': Tourist_Type,
    'Interest_Category': Interest_Category,
    'Budget': Budget,
    'Season': Season,
}


user_input_encoded = pd.DataFrame([user_input])
user_input_encoded = pd.get_dummies(user_input_encoded)


user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)


st.write(f"Model Accuracy: {accuracy:.2%}")


if st.button('Get Recommendation'):
    probabilities = best_model.predict_proba(user_input_encoded)[0]
    class_labels = best_model.classes_
    recommendations = pd.DataFrame({'Activity': class_labels, 'Probability': probabilities})
    recommendations = recommendations.sort_values(by='Probability', ascending=False)

    st.subheader("Top Recommendations:")
    for i, row in recommendations.head(3).iterrows():
        st.write(f"- {row['Activity']} (Confidence: {row['Probability']:.2%})")