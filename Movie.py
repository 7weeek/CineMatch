import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random

st.title("Movie Recommender")


try:
    with open("movie_data_new_sfw.pkl", "rb") as f:
        movie_dict = pickle.load(f)
        movie = pd.DataFrame(movie_dict)
except Exception as e:
    st.error(f"Error loading data: {e}")

import sqlite3
import bcrypt

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

conn.commit()
def start_onboarding_quiz():
    st.session_state["onboarding_stage"] = 0
    st.session_state["picked_titles"] = []
    st.session_state["movie_indices"] = random.sample(range(len(vectorized_movie)), 14)  # 7 rounds × 2 movies
    st.rerun()


def run_onboarding_quiz():
    if "onboarding_stage" not in st.session_state or "movie_indices" not in st.session_state:
        return

    if st.session_state["onboarding_stage"] >= 7:
        if "picked_titles" in st.session_state and "user_id" in st.session_state:
            picked_tags = movie[movie["title"].isin(st.session_state["picked_titles"])]["tag"].dropna()
            combined_tags = " ".join(picked_tags)
            user_vector = cv.transform([combined_tags]).toarray()[0]
            final_vector = pickle.dumps(user_vector)

            cursor.execute(
                "UPDATE user_tags SET tag_vector = ?, raw_tags = ? WHERE user_id = ?",
                (final_vector, combined_tags, st.session_state["user_id"])
            )
            conn.commit()
            st.success("🎯 Taste profile saved!")

        # Clear quiz state
        for key in ["onboarding_stage", "picked_titles", "movie_indices"]:
            st.session_state.pop(key, None)
        return

    idx1 = st.session_state["movie_indices"][st.session_state["onboarding_stage"] * 2]
    idx2 = st.session_state["movie_indices"][st.session_state["onboarding_stage"] * 2 + 1]

    movie1 = vectorized_movie.iloc[idx1]
    movie2 = vectorized_movie.iloc[idx2]

    st.subheader(f"🎥 Round {st.session_state['onboarding_stage'] + 1}/7: Pick your favorite")

    col1, col2 = st.columns(2)

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








#https://media.themoviedb.org/t/p/w300_and_h450_bestv2

def filter_by_genre(genre, top_n=20):
    genre = genre.lower()
    genre_str = movie['genres'].astype(str).str.lower()
    filtered = movie[genre_str.str.contains(genre)]
    results = filtered.sort_values(by='popularity', ascending=False).head(top_n)
    return results[['title', 'genres', 'popularity', 'vote_average',"poster_path","imdb_id"]]


from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(max_features=5000, stop_words="english")  # finds top 5000 most relevant words
movie = movie.reset_index(drop=True)
vectors = cv.fit_transform(movie['tag']).toarray()


# Initialize this ONCE at startup (global scope)
nn = NearestNeighbors(metric='cosine', algorithm='auto').fit(vectors)  # or metric='euclidean' etc.
# Keep your original function exactly as-is

vectorized_movie = movie.iloc[:len(vectors)].reset_index(drop=True)

#movie indexes are more than 5000 movies obviously, we vectorized it to make it accessible
#vectorized movie is only used for hybrid
def hybrid_recommender(movier, total_recs=20):
    movier = movier.lower()
    title_matches = vectorized_movie[vectorized_movie['title'].str.lower().str.contains(movier)]
    title_list = title_matches['title'].tolist()

    if not vectorized_movie['title'].str.lower().eq(movier).any():
        return ["Movie not found."]

    movie_index = vectorized_movie[vectorized_movie['title'].str.lower() == movier].index[0]
    distances, indices = nn.kneighbors(vectors[movie_index].reshape(1, -1), n_neighbors=total_recs + 1)
    tag_matches = [vectorized_movie.iloc[i]["title"] for i in indices[0][1:] if vectorized_movie.iloc[i]["title"] not in title_list]

    return (title_list[:8] + tag_matches)[:total_recs]



def title_based(movier):
    movier = movier.lower()
    matches = movie[movie['title'].str.lower().str.contains(movier)]
    if not matches.empty:
        return matches['title'].tolist()
    return []



base_url = "https://media.themoviedb.org/t/p/w300_and_h450_bestv2"


with st.sidebar:
    st.header("🎞️ Navigation")
    if "user_id" in st.session_state:
        cursor.execute("SELECT username FROM users WHERE id = ?", (st.session_state["user_id"],))
        user = cursor.fetchone()
        if user:
            with st.container():
                st.markdown("### 👤 Logged in as:")
                st.success(user[0])  # Displays the username in a green box

    menu_pages = ["Home", "Search", "User Options","Auth"]
    page = st.radio("Go to", menu_pages)

    if "user_id" in st.session_state:
        cursor.execute("SELECT username, full_name FROM users WHERE id = ?", (st.session_state["user_id"],))
        user_data = cursor.fetchone()

        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()











if(page=="Home"):
    st.subheader("🔥 Top Popular Movies")
    top_popular = movie.sort_values(by="popularity", ascending=False).head(10)

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j, row in enumerate(top_popular.iloc[i:i + 5].itertuples()):
            poster_url = base_url + row.poster_path if pd.notna(row.poster_path) and row.poster_path else None
            imdb_link = f"https://www.imdb.com/title/{row.imdb_id}/" if hasattr(row, 'imdb_id') and pd.notna(
                row.imdb_id) else None

            with cols[j]:
                if poster_url:
                    st.markdown(f"[![{row.title}]({poster_url})]({imdb_link})", unsafe_allow_html=True)
                st.markdown(f"**{row.title}**", unsafe_allow_html=True)
                st.caption(f"⭐ {row.vote_average} | 🔥 {row.popularity}")


    st.subheader("🎯 Top Rated Movies (IMDb)")
    top_rated = movie.sort_values(by="vote_average", ascending=False).head(10)

    for i in range(0, 10, 5):
        cols = st.columns(5)
        for j, row in enumerate(top_rated.iloc[i:i + 5].itertuples()):
            poster_url = base_url + row.poster_path if pd.notna(row.poster_path) and row.poster_path else None
            imdb_link = f"https://www.imdb.com/title/{row.imdb_id}/" if hasattr(row, 'imdb_id') and pd.notna(
                row.imdb_id) else None

            with cols[j]:
                if poster_url:
                    st.markdown(f"[![{row.title}]({poster_url})]({imdb_link})", unsafe_allow_html=True)
                st.markdown(f"**{row.title}**", unsafe_allow_html=True)
                st.caption(f"⭐ {row.vote_average} | 🔥 {row.popularity}")
    if "user_id" in st.session_state:
        st.subheader("✨ Recommended For You")

        try:
            cursor.execute("SELECT tag_vector FROM user_tags WHERE user_id = ?", (st.session_state["user_id"],))
            result = cursor.fetchone()
            if result:
                user_vector = pickle.loads(result[0])
                distances, indices = nn.kneighbors(user_vector.reshape(1, -1), n_neighbors=10)
                recommended_df = vectorized_movie.iloc[indices[0]]

                for chunk_start in range(0, len(recommended_df), 5):
                    cols = st.columns(5)
                    for i, row in enumerate(recommended_df.iloc[chunk_start:chunk_start+5].itertuples()):
                        poster_url = base_url + row.poster_path if pd.notna(row.poster_path) else None
                        imdb_link = f"https://www.imdb.com/title/{row.imdb_id}/" if pd.notna(row.imdb_id) else None

                        with cols[i]:
                            if poster_url:
                                st.markdown(f"[![{row.title}]({poster_url})]({imdb_link})", unsafe_allow_html=True)
                            else:
                                st.write("❌ No poster available")
                            st.markdown(f"**{row.title}**", unsafe_allow_html=True)
                            st.caption(f"{row.genres}  \n⭐ {row.vote_average} | 🔥 {row.popularity}")
            else:
                st.info("🔍 No taste profile found. Try signing up or taking the quiz!")
        except Exception as e:
            st.error(f"⚠️ Error fetching recommendations: {e}")






elif(page=="Search"):
    left_col, right_col = st.columns([3, 1])

    with right_col:
        search_mode = st.selectbox("🔍 Search Type", ["Title", "Genre", "Similar Movies"])

    with left_col:
        if search_mode == "Title":
            user_input = st.text_input("Enter movie title")
        elif search_mode == "Genre":
            genre_list = sorted(set( # sorted and removed duplicates
                genre.strip()
                for genres in movie['genres'].dropna().astype(str)
                for genre in genres.replace("[", "").replace("]", "").replace("'", "").split(",")
                if genre.strip()  # removes empty strings
            ))

            user_input = st.selectbox("Choose Genre", genre_list)
        else:
            user_input = st.selectbox("Choose a Movie", movie['title'].values)






    if st.button("Find Movies 🎥"):
        if search_mode == "Title":
            results = title_based(user_input)
            results_df = movie[movie['title'].isin(results)].copy()
        elif search_mode == "Genre":
            results_df = filter_by_genre(user_input)
        else:
            recommended_titles = hybrid_recommender(user_input)
            results_df = movie[movie['title'].isin(recommended_titles)].copy()

        st.markdown("### 🎬 Recommendations:")

        # 🍿 Horizontal rows of 5
        for chunk_start in range(0, len(results_df), 5):
            cols = st.columns(5)
            for i, row in enumerate(results_df.iloc[chunk_start:chunk_start+5].itertuples()): #itertuples convert each row to a tupple
                poster_url = base_url + row.poster_path
                imdb_link = f"https://www.imdb.com/title/{row.imdb_id}/" if pd.notna(row.imdb_id) else None

                with cols[i]:
                    if poster_url:
                        st.markdown(f"[![{row.title}]({poster_url})]({imdb_link})", unsafe_allow_html=True)
                    else:
                        st.write("❌ No poster available")
                    st.markdown(f"**{row.title}**", unsafe_allow_html=True)
                    st.caption(f"{row.genres}  \n⭐ {row.vote_average} | 🔥 {row.popularity}")







elif page == "User Options":
    st.subheader("👤 Manage Account Settings")

    if "user_id" in st.session_state:
        cursor.execute("SELECT username, email, full_name, password_hash FROM users WHERE id = ?", (st.session_state["user_id"],))
        user_data = cursor.fetchone()
        if user_data:
            current_email = user_data[1]
            current_name = user_data[2]
            current_hash = user_data[3]

            # 🔁 Update Full Name
            new_name = st.text_input("📝 Change Full Name", value=current_name or "")
            if st.button("Update Name"):
                cursor.execute("UPDATE users SET full_name = ? WHERE id = ?", (new_name, st.session_state["user_id"]))
                conn.commit()
                st.success("✅ Full name updated!")
                st.rerun()
            # 🧑 Update Username
            new_username = st.text_input("👤 Change Username", value=user_data[0])
            if st.button("Update Username"):
                try:
                    cursor.execute("UPDATE users SET username = ? WHERE id = ?",
                                   (new_username, st.session_state["user_id"]))
                    conn.commit()
                    st.success("✅ Username updated!")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("❌ Username already taken.")

            # 📧 Update Email
            new_email = st.text_input("📮 Change Email", value=current_email or "")
            if st.button("Update Email"):
                cursor.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, st.session_state["user_id"]))
                conn.commit()
                st.success("✅ Email updated!")
                st.rerun()

            # 🔐 Update Password
            st.markdown("### 🔒 Change Password")
            old_pw = st.text_input("Old Password", type="password")
            new_pw = st.text_input("New Password", type="password")
            if st.button("Update Password"):
                if bcrypt.checkpw(old_pw.encode(), current_hash.encode()):
                    hashed_new_pw = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
                    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hashed_new_pw, st.session_state["user_id"]))
                    conn.commit()
                    st.success("✅ Password changed!")
                    st.rerun()
                else:
                    st.error("❌ Old password is incorrect.")
        else:
            st.warning("User not found.")
    else:
        st.info("🔐 Please log in to manage your account.")




if page == "Auth":
    st.subheader("🔑 Authentication")
    auth_mode = st.radio("Choose Mode", ["Login", "Signup"])

    if auth_mode == "Login":
        st.subheader("🔓 Login")
        login_email = st.text_input("Email")
        login_password = st.text_input("Password", type="password")

        if st.button("Login"):
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
        signup_username = st.text_input("New Username")
        signup_email = st.text_input("Email")
        signup_name = st.text_input("Full Name")
        signup_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
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
                    start_onboarding_quiz()
                except sqlite3.IntegrityError:
                    st.error("❌ Username or email already exists.")
            else:
                st.warning("Please fill out all required fields.")

if "onboarding_stage" in st.session_state:
    run_onboarding_quiz()
