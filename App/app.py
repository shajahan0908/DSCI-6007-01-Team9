import flask
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = flask.Flask(__name__, template_folder='templates')

# Load Data
df2 = pd.read_csv('./model/tmdb.csv')

# Check if models are already pickled
try:
    with open('./model/tfidf_model.pkl', 'rb') as tfidf_file:
        tfidf = pickle.load(tfidf_file)

    with open('./model/cosine_sim.pkl', 'rb') as cosine_file:
        cosine_sim = pickle.load(cosine_file)

    print("Loaded pickled models.")

except FileNotFoundError:
    print("Pickle files not found. Re-training models...")

    # Train and pickle the models if not already done
    tfidf = TfidfVectorizer(stop_words='english', analyzer='word')
    tfidf_matrix = tfidf.fit_transform(df2['soup'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    with open('./model/tfidf_model.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf, tfidf_file)

    with open('./model/cosine_sim.pkl', 'wb') as cosine_file:
        pickle.dump(cosine_sim, cosine_file)

    print("Models have been trained and pickled.")

# Reset index and setup indices
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
all_titles = [df2['title'][i] for i in range(len(df2['title']))]

# Recommendation function
def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Print similarity scores (for debugging)
    print("\n movieId      score")
    for i in sim_scores:
        print(i)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return list of similar movies
    return_df = pd.DataFrame(columns=['Title', 'Homepage', 'ReleaseDate'])
    return_df['Title'] = df2['title'].iloc[movie_indices]
    return_df['Homepage'] = df2['homepage'].iloc[movie_indices]
    return_df['ReleaseDate'] = df2['release_date'].iloc[movie_indices]
    return return_df

# Main route for the web app
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('index.html')

    if flask.request.method == 'POST':
        m_name = " ".join(flask.request.form['movie_name'].title().split())
        if m_name not in all_titles:
            return flask.render_template('notFound.html', name=m_name)
        else:
            result_final = get_recommendations(m_name)
            names = []
            homepage = []
            releaseDate = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                releaseDate.append(result_final.iloc[i][2])
                if len(str(result_final.iloc[i][1])) > 3:
                    homepage.append(result_final.iloc[i][1])
                else:
                    homepage.append("#")

            return flask.render_template('found.html', movie_names=names, movie_homepage=homepage, search_name=m_name, movie_releaseDate=releaseDate)

# Run the app
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
