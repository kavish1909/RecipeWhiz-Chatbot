from flask import Flask, request, render_template, Response, jsonify, redirect

import json
import re
import recipeWhiz


# set up chatagent
chatbotAgent = recipeWhiz.RecipeWhiz()

#greetings and ending chat responses.
def chat(input):
    # get input from somewhere
    if "Hi" in input or "hi" in input or "hey" in input:
        return "Hi There! How can I help you today?"
    if "bye" in input:
        return "Bye! See you again."
    if "thank" in input:
        return "Most welcome!"
    reply = chatbotAgent.respond(input)

    # send reply somewhere output # text
    return reply


app = Flask(__name__, template_folder='templates')

app.static_folder = 'static'

## load the transformers zero shot classficiation model from hugging_face
## sentence_transformers

from transformers import pipeline

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


## Text classfication using zero-shot-classification

def classify_text(text):
    sequence_to_classify = text
    candidate_labels = ['food related', 'casual text']
    output = classifier(sequence_to_classify, candidate_labels)
    if output['labels'][0] == "food related":
        return True
    else:
        return False


#recipie recommendation chatbot logic starts here

import pandas as pd
import joblib
import ast
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Downloading stopwords if not already present
nltk.download('stopwords')

# Loading the clustering model
model = joblib.load('clustering_model.pkl')
print('\nModel loaded\n')

# Loading the dataset
df = pd.read_csv("cleaned.csv")  # Replace path of cleaned dataset
print('\nDataset read\n')

# Creating a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Evaluating rows of columns to their types
df['Features'] = df['Features'].apply(ast.literal_eval)
df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
df['directions'] = df['directions'].apply(ast.literal_eval)
print('\nLiteral eval done\n')

# Concatenating all strings in the 'Features' column into a single list
all_features = [feature for features in df['Features'] for feature in features]

# Removing stopwords from all_features
nltk_stopwords = set(stopwords.words('english'))
more_stops = ['hello', 'hi', 'hey', 'good', 'morning', 'evening']  # Add desired stop words here
stopwords = nltk_stopwords.union(more_stops)

all_features = [feature for feature in all_features if feature.lower() not in stopwords]
print('\nStopwords removed from all_features\n')

# Initializing an empty list to store the documents
documents = df['RecipeCategory'].tolist()

# Creating the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
print('\nTF-IDF matrix created\n')


recommendations = pd.DataFrame({})


def recommend_food(text):
    global recommendations
    food_text = ""

    user_input = text.replace(" ", "")

    if user_input.lower() == 'quit':
        pass

    # Getting recipe recommendations based on user input
    recommendations = recommend_recipes(user_input)

    # Checking if the user input contains multiple words without a comma
    if len(user_input.split(',')) > 1 and ',' not in user_input:
        return 'Error: Please separate multiple words with a comma.\n'

    # Filtering rows where any of the strings in 'Features' column contain the user input
    matching_keywords = []
    for word in user_input.split(','):
        if any(word.strip().lower() in feature.lower() for feature in all_features):
            matching_keywords.append(word)

    if not matching_keywords or any(word.strip().lower() in stopwords for word in user_input.split(',')):
        return 'Error: No recipe found. Try again.\n'

    # Printing the recommendations
    print('\nRecommendations:')
    food_text = "Recommendations:<br><br>"
    if recommendations is not None:
        for i, recipe in enumerate(recommendations['title']):
            food_text += f'{i + 1}. {recipe}'
            food_text += "<br>"
            print(f'{i + 1}. {recipe}')
    else:
        food_text = 'No recipes found.\n'
        print('No recipes found.\n')

    return food_text  # returns the text output of recommendation list.


def show_process(recommendations):
    # Get the recipe names for chat history display
    recipe_names = recommendations['title'].tolist()
    print(recommendations["directions"])
    return recommendations


def recommend_recipes(user_input):
    # Perform clustering on user input
    user_tfidf = vectorizer.transform([user_input])
    label = model.predict(user_tfidf)

    # Filter recipes based on the predicted cluster label
    cluster_labels = model.predict(tfidf_matrix)
    cluster_recipes = df[cluster_labels == label[0]]

    # Check if there are any recipes in the cluster
    if cluster_recipes.empty:
        return None

    # Calculate confidence scores for each recipe based on keyword matching
    recipe_scores = {}
    user_keywords = set(user_input.lower().split(','))

    for index, recipe in cluster_recipes.iterrows():
        features = set(recipe['Features'])
        matching_keywords = user_keywords.intersection(features)
        confidence_score = len(matching_keywords) / len(user_keywords)
        recipe_scores[index] = confidence_score

    # Sort recipes based on confidence scores in descending order
    sorted_recipes = sorted(recipe_scores, key=recipe_scores.get, reverse=True)

    # Select top 10 recipes with the highest confidence scores
    recommended_recipes = cluster_recipes.loc[sorted_recipes[:10]]

    return recommended_recipes  # returns the recommended recipie dataframe.


@app.route('/show_recommendations')
def show_recommendations():
    global recommendations
    df = show_process(recommendations)
    recommendations_html = df.to_html()
    response = app.make_response(
        render_template('recommendations.html', recommendations=recommendations, chat_history=[]))
    return response


@app.route('/')
def index():
    return render_template('base.html')


#     The /predict route for the frontend.
#     This is the main flask chat api which sends response to frontend


flag_food = True


@app.route('/predict', methods=['POST'])
def chat_post():
    global flag_food
    text = request.get_json('text')  # Use 'text' instead of 'form['text']'

    text = text['text']
    id = classify_text(text)
    if flag_food == "NEXT":
        if "sho" in text.lower():
            global recommendations
            # df = show_process(recommendations)

            return redirect('/show_recommendations')
        flag_food = True

    if "," in text:
        response = recommend_food(text)
        return jsonify(response=response)
    elif flag_food == False:
        flag_food = "NEXT"
    
    if id and flag_food:
        response = "Please share the list of ingredients you have which are ", " seperated, I will provide the recommendations based on this."  # return_recipie(text)
        flag_food = False
    else:
        response = chat(text)

    return jsonify(response=response)


if __name__ == '__main__':
    app.run("0.0.0.0")
