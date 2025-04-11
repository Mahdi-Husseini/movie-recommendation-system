import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components
import requests

data = pd.read_csv('combined_shit3.csv')

@st.cache_resource
def load_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['everything'])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

similarity_matrix = load_similarity_matrix(data)


import requests

def fetch_poster(movie_id):
    # Fetch movie details
    details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=3769b99b59867949cfd4a65ae3a7c656&language=en-US"
    details_response = requests.get(details_url)
    details_data = details_response.json()
    poster_path = details_data.get('poster_path')
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    # Fetch videos
    videos_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=3769b99b59867949cfd4a65ae3a7c656"
    videos_response = requests.get(videos_url)
    videos_data = videos_response.json()
    trailer_url = None
    for video in videos_data.get('results', []):
        if video.get('type') == 'Trailer' and video.get('site') == 'YouTube':
            trailer_url = f"https://www.youtube.com/watch?v={video.get('key')}"
            break

    return poster_url, trailer_url



st.image('msba.png')

st.title('Movie Recommendation System')

st.header('MSBA 315 - Group 7')

st.info('This is a movie recommendation system that suggests movies based on your preferences. You can filter movies by genre, popularity, and vote average in the sidebar.')
with st.container():
    st.markdown("""
    <div style="background-color:#1f1f1f;padding:20px;border-radius:10px">
        <h3 style="color:white;">Participants:</h3>
        <h4 style="color:white;">• Motorcycle Dude</h4>
        <h4 style="color:white;">• Cat Woman</h4>
        <h4 style="color:white;">• Alex's Mom</h4>
        <h4 style="color:white;">• Sergi Lavrov</h4>
        <h4 style="color:white;">• Sleepless Man</h4>
    </div>
    """, unsafe_allow_html=True)

st.write('---')

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")


# Initial handpicked movie IDs
default_ids = [
    299536, 559, 17455, 2830, 429422, 157336, 118340,
    240, 155, 19995, 250546, 255709, 38757
]

# Set up session state to preserve posters across reruns
if 'imageUrls' not in st.session_state:
    st.session_state.imageUrls = [fetch_poster(mid) for mid in default_ids]

# Title and carousel
st.title("🎬 Movie Poster Carousel")

# Show current posters
imageCarouselComponent(imageUrls=st.session_state.imageUrls, height=200)

# Suggestion button
st.markdown("### Want movie suggestions?")
if st.button("🎲 Suggest Movies"):
    random_ids = np.random.choice(data['id'].dropna().astype(int).unique(), size=13, replace=False)
    st.session_state.imageUrls = [fetch_poster(mid) for mid in random_ids]

unique_genres = set()

for genres in data['genre']:
    if isinstance(genres, str):  # only split if it's a string
        for genre in genres.split(','):
            unique_genres.add(genre.strip())

unique_genres = sorted(list(unique_genres))

with st.sidebar:
    st.title('Filter')

    st.subheader('Filter by Genre')
    unselected_genres = st.multiselect('Filter out:', unique_genres, default=[])
    # Change default to [] for "must have" filter
    selected_genres = st.multiselect('Must have those genres:', unique_genres, default=[])

    st.subheader('Popularity Filter')
    min_pop, max_pop = float(data['popularity'].min()), float(data['popularity'].max())
    pop_range = st.slider('Select popularity range:', min_pop, max_pop, (min_pop, max_pop))

    st.subheader('Vote Average Filter')
    min_vote, max_vote = float(data['vote_average'].min()), float(data['vote_average'].max())
    vote_range = st.slider('Select vote average range:', min_vote, max_vote, (min_vote, max_vote))




def recommend_movie(title, data, similarity_matrix, unselected_genres=None, selected_genres=None,
                    min_pop=None, max_pop=None, min_vote=None, max_vote=None, top_k=10):
    
    try:
        idx = data[data['original_title'].str.lower() == title.lower()].index[0]
    except IndexError:
        raise ValueError(f"Movie title '{title}' not found in dataset.")
    
    sim_scores = similarity_matrix[idx]
    similar_indices = sim_scores.argsort()[::-1][1:]  # Skip the movie itself
    
    def is_valid(movie):
        row = data.iloc[movie]
        
        # Genre filter
        if isinstance(row['genre'], str):
            genre_list = [g.strip() for g in row['genre'].split(',')]
        else:
            genre_list = []
        
        if selected_genres and not all(g in genre_list for g in selected_genres):
            return False
        if unselected_genres and any(g in genre_list for g in unselected_genres):
            return False
        
        # Popularity & vote filters
        if min_pop is not None and row['popularity'] < min_pop:
            return False
        if max_pop is not None and row['popularity'] > max_pop:
            return False
        if min_vote is not None and row['vote_average'] < min_vote:
            return False
        if max_vote is not None and row['vote_average'] > max_vote:
            return False
        
        return True

    # Collect results that pass filters until we get top_k
    recommended = []
    for i in similar_indices:
        if is_valid(i):
            row = data.iloc[i]
            recommended.append({
                'original_title': row['original_title'],
                'everything': row['everything'],
                'id': row['id']
            })
        if len(recommended) == top_k:
            break

    return pd.DataFrame(recommended)



movie =  st.selectbox('Select the Movie:', data['original_title'])

butt = st.button('Recommend')

if butt:
    pred_movies = recommend_movie(
        movie,
        data,
        similarity_matrix,
        unselected_genres=unselected_genres,
        selected_genres=selected_genres,
        min_pop=pop_range[0],
        max_pop=pop_range[1],
        min_vote=vote_range[0],
        max_vote=vote_range[1]
    )
    st.subheader(f'The recommended movies similar to {movie} are:')
    for _, row in pred_movies.iterrows():
        with st.container():
            # Try to fetch both poster and trailer
            try:
                poster_url, trailer_url = fetch_poster(row['id'])
            except Exception:
                poster_url, trailer_url = None, None

            # Always show poster if available
            st.markdown(f"""
            <div style="background-color:#2b2b2b;padding:15px;margin-bottom:10px;border-radius:8px;">
                <h4 style="color:#f63366;">🎬 {row['original_title']}</h4>
                <img alt='poster' src='{poster_url}' height='200'>
            </div>
            """, unsafe_allow_html=True)

            # Show trailer only if available
            if trailer_url:
                st.video(trailer_url)
            st.write('***')
