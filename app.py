from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# df = pickle.load(open('temp_df.pkl', 'rb'))

df = pd.read_pickle('temp_df.pkl')
# books =pickle.load(open('books.pkl'), 'rb')
dff = pd.read_pickle('books.pkl')
# similarity_score = pickle.load(open('similarity_scores.pkl', 'rb'))

similarity_score = pd.read_pickle('similarity_scores.pkl')

# collaborative pickle files
pv = pd.read_pickle('pv2.pkl')
similarity_scores = pd.read_pickle('col_similarity_scores2.pkl')
col_df = pd.read_pickle('col_Df2.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html',
                           book_name = list(df['title'].values),
                           author_name = list(df['author_name'].values),
                           book_img = list(df['image_url'].values)
                           )

@app.route('/recommendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/colabrecommendations')
def col_recommendations():
    return render_template('collaborative_recommendations.html')

@app.route('/recommend', methods=["post"])
def recommend_book():
    book_name = request.form.get('book_name')

    book_index = dff[dff['title']==book_name].index[0]
    similarity_array = similarity_score[book_index ]
    recommended_books = sorted(list(enumerate(similarity_array)), reverse=True, key=lambda x: x[1])[1:10]
    
    books = []
    for i in recommended_books:
        items = []
        temp_df = (dff.iloc[i[0]])
        items.append(temp_df['title'])
        items.append(temp_df['author_name'])
        items.append(temp_df['image_url'])
       

        books.append(items)
    print(books)
    return render_template('recommendations.html', data=books)

@app.route('/col_recommend', methods=["post"])
def col_recommend():
    book_name = request.form.get('book')
    print(book_name)
    index = np.where(pv.index==book_name)[0][0]
    print(index)
    distances = similarity_scores[index]
    suggestions = sorted(list(enumerate(similarity_scores[index])), key=lambda x:x[1], reverse=True)[1:6]
    books = []
    for i in suggestions:
        # print(i)
        items = []
        # print(pv.index[i[0]])
        temp_df = col_df[col_df['title'] == pv.index[i[0]]]
        items.extend(list(temp_df.drop_duplicates('title')['title'].values))
        items.extend(list(temp_df.drop_duplicates('title')['image_url'].values))
        items.extend(list(temp_df.drop_duplicates('title')['author_name'].values))


        books.append(items)
    return render_template('collaborative_recommendations.html', data=books)



if __name__ == 'main':
    app.run(debug=True)