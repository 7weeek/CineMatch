import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random
import sqlite3
import bcrypt
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from functools import partial





st.set_page_config(
    page_title="CineMatch 🎬",
    page_icon="🍿",
    layout="wide"
)



st.markdown("""
    <style>
        /* Header styling */
        .header {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #ff4d4d, #f9cb28);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Movie card hover effect */
        .movie-card {
            transition: transform 0.3s ease;
            border-radius: 10px;
            overflow: hidden;
        }
        .movie-card:hover {
            transform: scale(1.03);
        }

        /* Footer styling */
        .footer {
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            color: #666;
            border-top: 1px solid #eee;
        }
    </style>
""", unsafe_allow_html=True)






st.markdown("""
    <div class="header">
        <h1>🎬 CineMatch</h1>
        <p>Your personal movie recommendation engine</p>
    </div>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")







# Initialize session state for search persistence
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "Title"
if 'watchlist_states' not in st.session_state:
    st.session_state.watchlist_states = {}
if 'like_states' not in st.session_state:
    st.session_state.like_states = {}







# Load movie data
try:
    with open("movie_data_new_sfw.pkl", "rb") as f:
        movie_dict = pickle.load(f)
        movie = pd.DataFrame(movie_dict)
except Exception as e:
    st.error(f"Error loading data: {e}")







# Database setup
conn = sqlite3.connect("user_data.db", check_same_thread=False)
cursor = conn.cursor()

# Create tables if not exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_tags (
    user_id INTEGER PRIMARY KEY,
    tag_vector TEXT NOT NULL,
    raw_tags TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_likes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id TEXT NOT NULL,
    liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, movie_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, movie_id)
)
""")
conn.commit()









# Helper functions
def start_onboarding_quiz():
    st.session_state["onboarding_stage"] = 0
    st.session_state["picked_titles"] = []
    st.session_state["movie_indices"] = random.sample(range(len(vectorized_movie)), 14)
    st.rerun()


def run_onboarding_quiz():
    if "onboarding_stage" not in st.session_state or "movie_indices" not in st.session_state:
        return

    if st.session_state["onboarding_stage"] >= 7:
        if "picked_titles" in st.session_state and "user_id" in st.session_state:
            picked_movies = movie[movie["title"].isin(st.session_state["picked_titles"])]
            combined_tags = " ".join(picked_movies["tag"].dropna().tolist())

            # Add all quiz selections to liked movies
            for _, movie_row in picked_movies.iterrows():
                add_like(st.session_state["user_id"], movie_row['imdb_id'])

            # Create and store the user's preference vector
            user_vector = cv.transform([combined_tags]).toarray()[0]
            final_vector = pickle.dumps(user_vector)

            # Update database
            cursor.execute(
                "UPDATE user_tags SET tag_vector = ?, raw_tags = ? WHERE user_id = ?",
                (final_vector, combined_tags, st.session_state["user_id"])
            )
            conn.commit()

            st.success("🎯 Taste profile saved! Check your personalized recommendations.")
            st.balloons()

        # Clear quiz state
        for key in ["onboarding_stage", "picked_titles", "movie_indices"]:
            st.session_state.pop(key, None)
        st.rerun()
        return

    idx1 = st.session_state["movie_indices"][st.session_state["onboarding_stage"] * 2]
    idx2 = st.session_state["movie_indices"][st.session_state["onboarding_stage"] * 2 + 1]

    movie1 = vectorized_movie.iloc[idx1]
    movie2 = vectorized_movie.iloc[idx2]

    st.subheader(f"🎥 Round {st.session_state['onboarding_stage'] + 1}/7: Pick your favorite")

    col1, col2, col3 = st.columns([1, 1, 0.3])

    with col1:
        if pd.notna(movie1.poster_path):
            st.image(base_url + movie1.poster_path, width=250)
        if st.button(f"👍 {movie1.title}", key=f"pick_m1_{idx1}"):
            st.session_state["picked_titles"].append(movie1.title)
            st.session_state["onboarding_stage"] += 1
            st.rerun()

    with col2:
        if pd.notna(movie2.poster_path):
            st.image(base_url + movie2.poster_path, width=250)
        if st.button(f"👍 {movie2.title}", key=f"pick_m2_{idx2}"):
            st.session_state["picked_titles"].append(movie2.title)
            st.session_state["onboarding_stage"] += 1
            st.rerun()

    with col3:
        st.write("")
        st.write("")
        if st.button("♻️ Reroll", key=f"reroll_{st.session_state['onboarding_stage']}"):
            available_indices = [i for i in range(len(vectorized_movie))
                                 if i not in st.session_state["movie_indices"]]
            if len(available_indices) >= 2:
                new_idx1, new_idx2 = random.sample(available_indices, 2)
                current_round = st.session_state["onboarding_stage"]
                st.session_state["movie_indices"][current_round * 2] = new_idx1
                st.session_state["movie_indices"][current_round * 2 + 1] = new_idx2
                st.rerun()
            else:
                st.warning("No more unique movies available")



def add_to_watchlist(user_id, movie_id):
    try:
        cursor.execute(
            "INSERT INTO user_watchlist (user_id, movie_id) VALUES (?, ?)",
            (user_id, movie_id)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False





def remove_from_watchlist(user_id, movie_id):
    cursor.execute(
        "DELETE FROM user_watchlist WHERE user_id = ? AND movie_id = ?",
        (user_id, movie_id)
    )
    conn.commit()
    return cursor.rowcount > 0




def add_like(user_id, movie_id):
    try:
        cursor.execute(
            "INSERT INTO user_likes (user_id, movie_id) VALUES (?, ?)",
            (user_id, movie_id)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False





def remove_like(user_id, movie_id):
    cursor.execute(
        "DELETE FROM user_likes WHERE user_id = ? AND movie_id = ?",
        (user_id, movie_id)
    )
    conn.commit()
    return cursor.rowcount > 0




def is_in_watchlist(user_id, movie_id):
    cursor.execute(
        "SELECT 1 FROM user_watchlist WHERE user_id = ? AND movie_id = ?",
        (user_id, movie_id)
    )
    return cursor.fetchone() is not None




def is_liked(user_id, movie_id):
    cursor.execute(
        "SELECT 1 FROM user_likes WHERE user_id = ? AND movie_id = ?",
        (user_id, movie_id)
    )
    return cursor.fetchone() is not None




def get_liked_movies(user_id):
    cursor.execute("SELECT movie_id FROM user_likes WHERE user_id = ?", (user_id,))
    return [x[0] for x in cursor.fetchall()]




def get_enhanced_recommendations(user_id, n_recommendations=10):
    cursor.execute("SELECT tag_vector FROM user_tags WHERE user_id = ?", (user_id,))
    tag_result = cursor.fetchone()
    if not tag_result:
        return None

    liked_ids = get_liked_movies(user_id)
    if not liked_ids:
        return None

    user_vector = pickle.loads(tag_result[0])
    liked_movies = movie[movie['imdb_id'].isin(liked_ids)]

    if not liked_movies.empty:
        liked_tags = " ".join(liked_movies['tag'].dropna().tolist())
        liked_vector = cv.transform([liked_tags]).toarray()[0]
        combined_vector = user_vector + liked_vector * 1.5
        combined_vector = normalize(combined_vector.reshape(1, -1))[0]
    else:
        combined_vector = user_vector

    distances, indices = nn.kneighbors([combined_vector], n_neighbors=n_recommendations + len(liked_ids))

    recommendations = []
    for i in indices[0]:
        movie_id = vectorized_movie.iloc[i]['imdb_id']
        if movie_id not in liked_ids:
            recommendations.append(vectorized_movie.iloc[i])
            if len(recommendations) >= n_recommendations:
                break

    return recommendations




# Callback functions
def watchlist_callback(movie_id, add=True):
    if "user_id" not in st.session_state:
        return

    if add:
        add_to_watchlist(st.session_state["user_id"], movie_id)
    else:
        remove_from_watchlist(st.session_state["user_id"], movie_id)

    st.session_state.watchlist_states[movie_id] = add


def like_callback(movie_id, add=True):
    if "user_id" not in st.session_state:
        return

    if add:
        add_like(st.session_state["user_id"], movie_id)
    else:
        remove_like(st.session_state["user_id"], movie_id)

    st.session_state.like_states[movie_id] = add




def display_movie_with_actions(row, cols, i, search_mode=None, user_input=None):
    poster_url = base_url + row.poster_path if pd.notna(row.poster_path) else None
    imdb_link = f"https://www.imdb.com/title/{row.imdb_id}/" if pd.notna(row.imdb_id) else None

    # Clean up genres display
    if isinstance(row.genres, str):
        # Remove brackets and quotes if they exist
        genres_display = row.genres.replace("[", "").replace("]", "").replace("'", "")
    else:
        genres_display = str(row.genres)

    with cols[i]:
        st.markdown(f"""
            <div class="movie-card">
                <a href="{imdb_link}" target="_blank">
                    <img src="{poster_url if poster_url else 'https://via.placeholder.com/300x450?text=No+Poster'}" 
                         width="100%" 
                         style="border-radius: 10px 10px 0 0;">
                </a>
                <div style="padding: 10px;">
                    <h4>{row.title}</h4>
                    <p style="color: #666; font-size: 14px;">
                        {genres_display}<br>
                        ⭐ {row.vote_average} | 🔥 {row.popularity}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if "user_id" in st.session_state:
            movie_id = row.imdb_id
            user_id = st.session_state["user_id"]

            # Get current states
            watchlist_state = st.session_state.watchlist_states.get(movie_id, is_in_watchlist(user_id, movie_id))
            like_state = st.session_state.like_states.get(movie_id, is_liked(user_id, movie_id))

            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if watchlist_state:
                    st.button(
                        "✅ Watchlist",
                        key=f"watch_{movie_id}_{i}",
                        on_click=partial(watchlist_callback, movie_id, False)
                    )
                else:
                    st.button(
                        "➕ Watchlist",
                        key=f"watch_{movie_id}_{i}",
                        on_click=partial(watchlist_callback, movie_id, True)
                    )

            with btn_col2:
                if like_state:
                    st.button(
                        "❤️ Liked",
                        key=f"like_{movie_id}_{i}",
                        on_click=partial(like_callback, movie_id, False)
                    )
                else:
                    st.button(
                        "🤍 Like",
                        key=f"like_{movie_id}_{i}",
                        on_click=partial(like_callback, movie_id, True)
                    )




def filter_by_genre(genre, top_n=20):
    genre = genre.lower()
    genre_str = movie['genres'].astype(str).str.lower()
    filtered = movie[genre_str.str.contains(genre)]
    results = filtered.sort_values(by='popularity', ascending=False).head(top_n)
    return results[['title', 'genres', 'popularity', 'vote_average', "poster_path", "imdb_id"]]


def hybrid_recommender(movier, total_recs=20):
    movier = movier.lower()
    title_matches = vectorized_movie[vectorized_movie['title'].str.lower().str.contains(movier)]
    title_list = title_matches['title'].tolist()

    if not vectorized_movie['title'].str.lower().eq(movier).any():
        return ["Movie not found."]

    movie_index = vectorized_movie[vectorized_movie['title'].str.lower() == movier].index[0]
    distances, indices = nn.kneighbors(vectors[movie_index].reshape(1, -1), n_neighbors=total_recs + 1)
    tag_matches = [vectorized_movie.iloc[i]["title"] for i in indices[0][1:] if
                   vectorized_movie.iloc[i]["title"] not in title_list]

    return (title_list[:8] + tag_matches)[:total_recs]


def title_based(movier):
    movier = movier.lower()
    matches = movie[movie['title'].str.lower().str.contains(movier)]
    if not matches.empty:
        return matches['title'].tolist()
    return []








# Initialize vectorizer and model
cv = TfidfVectorizer(max_features=5000, stop_words="english")
movie = movie.reset_index(drop=True)
vectors = cv.fit_transform(movie['tag']).toarray()
nn = NearestNeighbors(metric='cosine', algorithm='auto').fit(vectors)
vectorized_movie = movie.iloc[:len(vectors)].reset_index(drop=True)
base_url = "https://media.themoviedb.org/t/p/w300_and_h450_bestv2"





# Main tabs navigation
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔍 Search", "👤 Profile", "🔑 Auth"])




# Home Tab
with tab1:
    st.subheader("🔥 Top Popular Movies")
    top_popular = movie.sort_values(by="popularity", ascending=False).head(10)

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j, row in enumerate(top_popular.iloc[i:i + 5].itertuples()):
            display_movie_with_actions(row, cols, j, None, None)

    st.subheader("🎯 Top Rated Movies (IMDb)")
    top_rated = movie.sort_values(by="vote_average", ascending=False).head(10)

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j, row in enumerate(top_rated.iloc[i:i + 5].itertuples()):
            display_movie_with_actions(row, cols, j, None, None)

    if "user_id" in st.session_state:
        st.subheader("✨ Recommended For You")
        try:
            recommendations = get_enhanced_recommendations(st.session_state["user_id"])

            if recommendations:
                for chunk_start in range(0, len(recommendations), 5):
                    cols = st.columns(5)
                    for i, row in enumerate(recommendations[chunk_start:chunk_start + 5]):
                        with cols[i]:
                            poster_url = base_url + row.poster_path if pd.notna(row.poster_path) else None
                            imdb_link = f"https://www.imdb.com/title/{row.imdb_id}" if pd.notna(row.imdb_id) else None

                            # Updated to include the movie-card class and consistent styling
                            st.markdown(f"""
                                <div class="movie-card">
                                    <a href="{imdb_link}" target="_blank">
                                        <img src="{poster_url if poster_url else 'https://via.placeholder.com/300x450?text=No+Poster'}" 
                                             style="width: 100%; border-radius: 10px 10px 0 0;">
                                    </a>
                                    <div style="padding: 10px;">
                                        <h4>{row.title}</h4>
                                        <p style="color: #666; font-size: 14px;">
                                            {row.genres}<br>
                                            ⭐ {row.vote_average} | 🔥 {row.popularity if hasattr(row, 'popularity') else ''}
                                        </p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            # Add like/watchlist buttons
                            movie_id = row.imdb_id
                            user_id = st.session_state["user_id"]

                            # Get current states
                            watchlist_state = st.session_state.watchlist_states.get(movie_id,
                                                                                    is_in_watchlist(user_id, movie_id))
                            like_state = st.session_state.like_states.get(movie_id, is_liked(user_id, movie_id))

                            btn_col1, btn_col2 = st.columns(2)

                            with btn_col1:
                                if watchlist_state:
                                    st.button(
                                        "✅ Watchlist",
                                        key=f"watch_{movie_id}_{i}_home",
                                        on_click=partial(watchlist_callback, movie_id, False)
                                    )
                                else:
                                    st.button(
                                        "➕ Watchlist",
                                        key=f"watch_{movie_id}_{i}_home",
                                        on_click=partial(watchlist_callback, movie_id, True)
                                    )

                            with btn_col2:
                                if like_state:
                                    st.button(
                                        "❤️ Liked",
                                        key=f"like_{movie_id}_{i}_home",
                                        on_click=partial(like_callback, movie_id, False)
                                    )
                                else:
                                    st.button(
                                        "🤍 Like",
                                        key=f"like_{movie_id}_{i}_home",
                                        on_click=partial(like_callback, movie_id, True)
                                    )
            else:
                st.info("🎬 Complete the onboarding quiz to get personalized recommendations!")
                if st.button("Take Quiz Now", key="take_quiz_home"):
                    start_onboarding_quiz()

        except Exception as e:
            st.error(f"⚠️ Error fetching recommendations: {e}")

        st.subheader("📝 Your Watchlist")
        cursor.execute("""
                    SELECT movie_id FROM user_watchlist 
                    WHERE user_id = ? 
                    ORDER BY added_at DESC
                    LIMIT 10
                    """, (st.session_state["user_id"],))

        watchlist_ids = [x[0] for x in cursor.fetchall()]
        watchlist_movies = movie[movie['imdb_id'].isin(watchlist_ids)]

        if not watchlist_movies.empty:
            for i in range(0, len(watchlist_movies), 5):
                cols = st.columns(5)
                for j, row in enumerate(watchlist_movies.iloc[i:i + 5].itertuples()):
                    display_movie_with_actions(row, cols, j)
        else:
            st.info("Your watchlist is empty. Add movies to your watchlist to see them here!")
# Search Tab
with tab2:
    left_col, right_col = st.columns([3, 1])

    with right_col:
        search_mode = st.selectbox(
            "🔍 Search Type",
            ["Title", "Genre", "Similar Movies"],
            key="search_type_select"
        )

    with left_col:
        if search_mode == "Title":
            search_query = st.text_input("Enter movie title", key="title_search_input")
        elif search_mode == "Genre":
            genre_list = sorted(set(
                genre.strip()
                for genres in movie['genres'].dropna().astype(str)
                for genre in genres.replace("[", "").replace("]", "").replace("'", "").split(",")
                if genre.strip()
            ))
            search_query = st.selectbox("Choose Genre", genre_list, key="genre_select")
        else:
            search_query = st.selectbox("Choose a Movie", movie['title'].values, key="movie_select")

    if st.button("Find Movies 🎥", key="find_movies_button"):
        if search_mode == "Title":
            results = title_based(search_query)
            st.session_state.search_results = movie[movie['title'].isin(results)].copy()
        elif search_mode == "Genre":
            st.session_state.search_results = filter_by_genre(search_query)
        else:
            recommended_titles = hybrid_recommender(search_query)
            st.session_state.search_results = movie[movie['title'].isin(recommended_titles)].copy()

        st.session_state.search_query = search_query
        st.session_state.search_mode = search_mode

    if st.session_state.search_results is not None:
        st.markdown(f"### 🎬 Results for: {st.session_state.search_query}")

        results_df = st.session_state.search_results
        for chunk_start in range(0, len(results_df), 5):
            cols = st.columns(5)
            for i, row in enumerate(results_df.iloc[chunk_start:chunk_start + 5].itertuples()):
                display_movie_with_actions(row, cols, i, st.session_state.search_mode, st.session_state.search_query)





# Profile Tab
with tab3:
    if "user_id" in st.session_state:
        cursor.execute("SELECT username, email, full_name, password_hash FROM users WHERE id = ?",
                       (st.session_state["user_id"],))
        user_data = cursor.fetchone()

        if user_data:
            st.subheader("👤 Manage Account Settings")

            with st.form("profile_form"):
                username = st.text_input("Username", value=user_data[0])
                email = st.text_input("Email", value=user_data[1])
                full_name = st.text_input("Full Name", value=user_data[2])

                # Password change section
                st.markdown("### Change Password")
                old_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm New Password", type="password")

                if st.form_submit_button("Update Profile"):
                    try:
                        # First validate password change if fields are filled
                        password_updated = False
                        if old_password or new_password or confirm_password:
                            if not (old_password and new_password and confirm_password):
                                st.error("Please fill all password fields to change password")
                            elif not bcrypt.checkpw(old_password.encode(), user_data[3].encode()):
                                st.error("Current password is incorrect")
                            elif new_password != confirm_password:
                                st.error("New passwords don't match")
                            else:
                                # Hash the new password
                                new_password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                                password_updated = True

                        # Update the database
                        if password_updated:
                            cursor.execute(
                                """UPDATE users 
                                SET username = ?, email = ?, full_name = ?, password_hash = ?
                                WHERE id = ?""",
                                (username, email, full_name, new_password_hash, st.session_state["user_id"])
                            )
                        else:
                            cursor.execute(
                                """UPDATE users 
                                SET username = ?, email = ?, full_name = ?
                                WHERE id = ?""",
                                (username, email, full_name, st.session_state["user_id"])
                            )

                        conn.commit()
                        st.success("Profile updated successfully!")
                        st.rerun()

                    except sqlite3.IntegrityError:
                        st.error("Username or email already exists")

            st.subheader("🎬 Your Collections")
            tab_watchlist, tab_likes = st.tabs(["📝 Watchlist", "❤️ Likes"])

            with tab_watchlist:
                cursor.execute("""
                SELECT movie_id FROM user_watchlist 
                WHERE user_id = ? 
                ORDER BY added_at DESC
                """, (st.session_state["user_id"],))

                watchlist_ids = [x[0] for x in cursor.fetchall()]
                watchlist_movies = movie[movie['imdb_id'].isin(watchlist_ids)]

                if not watchlist_movies.empty:
                    for i in range(0, len(watchlist_movies), 5):
                        cols = st.columns(5)
                        for j, row in enumerate(watchlist_movies.iloc[i:i + 5].itertuples()):
                            with cols[j]:
                                if pd.notna(row.poster_path):
                                    st.image(base_url + row.poster_path, width=150)
                                st.write(row.title)
                                if st.button("Remove", key=f"remove_watch_{row.imdb_id}_{i}_{j}"):
                                    remove_from_watchlist(st.session_state["user_id"], row.imdb_id)
                                    st.rerun()
                else:
                    st.info("Your watchlist is empty")

            with tab_likes:
                cursor.execute("""
                SELECT movie_id FROM user_likes 
                WHERE user_id = ? 
                ORDER BY liked_at DESC
                """, (st.session_state["user_id"],))

                liked_ids = [x[0] for x in cursor.fetchall()]
                liked_movies = movie[movie['imdb_id'].isin(liked_ids)]

                if not liked_movies.empty:
                    for i in range(0, len(liked_movies), 5):
                        cols = st.columns(5)
                        for j, row in enumerate(liked_movies.iloc[i:i + 5].itertuples()):
                            with cols[j]:
                                if pd.notna(row.poster_path):
                                    st.image(base_url + row.poster_path, width=150)
                                st.write(row.title)
                                if st.button("Unlike", key=f"unlike_{row.imdb_id}_{i}_{j}"):
                                    remove_like(st.session_state["user_id"], row.imdb_id)
                                    st.rerun()
                else:
                    st.info("You haven't liked any movies yet")
        if st.button("🚪 Logout", key="logout_button"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.info("🔐 Please log in to manage your account")

# Auth Tab
with tab4:
    # Determine which view to show
    show_auth = True

    if "user_id" in st.session_state:
        cursor.execute("SELECT raw_tags FROM user_tags WHERE user_id = ?", (st.session_state["user_id"],))
        result = cursor.fetchone()
        if result and result[0]:
            show_auth = False

    if show_auth:
        st.subheader("🔑 Authentication")
        auth_mode = st.radio("Choose Mode", ["Login", "Signup"], key="auth_mode_radio")

        if auth_mode == "Login":
            st.subheader("🔓 Login")
            login_email = st.text_input("Email", key="login_email_input")
            login_password = st.text_input("Password", type="password", key="login_password_input")

            if st.button("Login", key="login_button"):
                cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (login_email,))
                user = cursor.fetchone()
                if user and bcrypt.checkpw(login_password.encode(), user[1].encode()):
                    st.session_state["user_id"] = user[0]
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials.")

        elif auth_mode == "Signup":
            st.subheader("🆕 Create Account")
            signup_username = st.text_input("New Username", key="signup_username_input")
            signup_email = st.text_input("Email", key="signup_email_input")
            signup_name = st.text_input("Full Name", key="signup_name_input")
            signup_password = st.text_input("New Password", type="password", key="signup_password_input")

            if st.button("Sign Up", key="signup_button"):
                if signup_username and signup_password:
                    hashed_pw = bcrypt.hashpw(signup_password.encode(), bcrypt.gensalt()).decode()
                    try:
                        cursor.execute(
                            "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
                            (signup_username, signup_email, hashed_pw, signup_name)
                        )
                        user_id = cursor.lastrowid
                        cursor.execute(
                            "INSERT INTO user_tags (user_id, tag_vector, raw_tags) VALUES (?, ?, ?)",
                            (user_id, pickle.dumps(np.zeros(vectors.shape[1])), "")
                        )
                        conn.commit()
                        st.session_state["user_id"] = user_id
                        st.success("✅ Account created! Let's personalize your experience...")
                        start_onboarding_quiz()
                    except sqlite3.IntegrityError:
                        st.error("❌ Username or email already exists.")
                else:
                    st.warning("Please fill out all required fields.")
    else:
        st.success("You're already logged in!")
        if st.button("Go to Profile"):
            st.switch_page("streamlit_app.py#profile")

    if "onboarding_stage" in st.session_state:
        run_onboarding_quiz()

st.markdown("""
    <div class="footer">
        <p>© 2025 CineMatch | Made with ❤️ for movie lovers</p>
        <p style="font-size: 12px;">Data provided by TMDB</p>
    </div>
""", unsafe_allow_html=True)
