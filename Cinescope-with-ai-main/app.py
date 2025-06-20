# app.py
from flask import Flask, render_template,request
from model.recommender import MovieRecommender


app = Flask(__name__)
recommender = MovieRecommender()



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations, posters = recommender.recommend(title)
    return render_template('index.html', recommendations=recommendations, posters=posters, zip=zip)

    
if __name__ == '__main__':
    recommender.load_data('C:/Users/sreel/OneDrive/Desktop/project/data/dataset.csv')  # Provide the correct path to your dataset
    recommender.train_model()
    app.run(debug=True)
    
