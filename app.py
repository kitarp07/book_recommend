from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
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


#neural_network_pickle_files
enumerate_bookids = pd.read_pickle('nn_pkl_files/enumerate_bookids.pkl')
enumerate_userids = pd.read_pickle('nn_pkl_files/enumerate_userids.pkl')
enumerate_bookids_reverse = pd.read_pickle('nn_pkl_files/enumerate_bookids_reverse.pkl')
model = tf.keras.models.load_model('nn_pkl_files/model_3.h5')
nn_books = pd.read_pickle('nn_pkl_files/nn_books.pkl')
nn_books_2 = pd.read_pickle('nn_pkl_files/nn_books_2.pkl')
nn_ratings = pd.read_pickle('nn_pkl_files/nn_ratings.pkl')

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

@app.route('/nn_recommendations')
def nn_recommendations():
    return render_template('neural_net_preds.html')


def get_books_rated_by_user(user_id):
  results=[]
  user_ratings_data = nn_ratings[nn_ratings["user_id"]==user_id]
 
  books_read_by_user = nn_books_2[nn_books_2["book_id"].isin(user_ratings_data.book_id.values)]
  print(books_read_by_user)
  for index, row in books_read_by_user.iterrows():
    values = [row['title'], row["description"], row['image_url'], row['author_name'], row["genres"]]
    results.append(values)
  return results

@app.route('/nn_recommend', methods=["post"])
def recommend_book_nn():
  user_id = request.form.get('id')
  user_id = int(user_id)
  results = []
  books_readby_user = nn_ratings[nn_ratings.user_id == user_id]
  
  books_not_read = nn_books_2[~nn_books_2["book_id"].isin(books_readby_user.book_id.values)]["book_id"]
  filtered_books_not_read = list(set(books_not_read).intersection(set(enumerate_bookids.keys())))
  books_not_read_by_user = [[enumerate_bookids.get(x)] for x in filtered_books_not_read]
  user_idx = enumerate_userids.get(user_id)
  user_books = np.hstack(([[user_idx]] * len(books_not_read_by_user), books_not_read_by_user ))
  ratings = model.predict([user_books[:, 0], user_books[:, 1]]).flatten()
  rating_idxs = np.argsort(ratings)[-10:][::-1]
  recommendations_idx = [enumerate_bookids_reverse.get(books_not_read_by_user[x][0]) for x in rating_idxs]

  #get books rated by user
  books_rated_by_user = get_books_rated_by_user(user_id)
  print(books_rated_by_user)

  books = nn_books_2[nn_books_2["book_id"].isin(recommendations_idx)]

  for index, row in books.iterrows():
    values = [row['title'], row["description"], row['image_url'], row['author_name'], row["genres"]]
    results.append(values)
  return render_template('neural_net_preds.html', data=results, user_books_data = books_rated_by_user)


if __name__ == 'main':
    app.run(debug=True)