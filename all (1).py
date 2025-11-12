#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Data Loading Function
def load_yield_data(filepath):
    df = pd.read_csv(filepath)
    # Select relevant columns
    df = df[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'hg/ha_yield']]
    df = df.dropna()

    X = df[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
    y = df['hg/ha_yield']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Step 2: Image Data Generation (for CNN)
def generate_image_data(num_images=100, img_shape=(64, 64, 3)):
    X_images = np.random.rand(num_images, *img_shape)
    y_labels = np.random.randint(2, size=num_images)
    return X_images, y_labels

# Step 3: CNN Model for Disease Detection
def build_cnn_model(img_shape=(64, 64, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Random Forest Model for Yield Prediction
def build_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 5: Recommendation Logic
def recommend(disease_pred, yield_pred):
    if disease_pred >= 0.5:
        return "Disease detected! Recommended action: Apply pesticide."
    elif yield_pred < 50000:
        return "Low yield predicted! Recommended action: Improve irrigation and soil quality."
    else:
        return "Crop is healthy and yield prediction is optimal."

# Step 6: Main Menu
def main_menu():
    cnn_model, rf_model = None, None
    X_train_images, y_train_images = None, None
    X_env, y_env, scaler = None, None, None

    while True:
        print("\nCrop Disease & Yield System - Menu")
        print("1. Train Disease Detection Model (CNN)")
        print("2. Train Yield Prediction Model (Random Forest using Dataset)")
        print("3. Test Disease & Yield Prediction and get Recommendation")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            X_train_images, y_train_images = generate_image_data()
            cnn_model = build_cnn_model()
            cnn_model.fit(X_train_images, y_train_images, epochs=5, batch_size=10, verbose=1)
            print(" Disease Detection Model trained successfully.")

        elif choice == '2':
            # Using your specified dataset path
            file_path = "RS-A1_yield.csv"
            X_env, y_env, scaler = load_yield_data(file_path)
            X_train, X_test, y_train, y_test = train_test_split(X_env, y_env, test_size=0.2, random_state=42)
            rf_model = build_rf_model(X_train, y_train)
            print(" Yield Prediction Model trained successfully using dataset.")

        elif choice == '3':
            if cnn_model is None or rf_model is None:
                print(" Please train both models first!")
                continue

            # Random test image & environmental data
            test_image = np.random.rand(1, 64, 64, 3)
            disease_pred = cnn_model.predict(test_image, verbose=0)[0][0]

            test_env_data = np.random.rand(1, 3)
            test_env_scaled = (test_env_data - test_env_data.mean()) / test_env_data.std()
            yield_pred = rf_model.predict(test_env_scaled)[0]

            recommendation = recommend(disease_pred, yield_pred)

            print(f"Disease Prediction: {disease_pred:.4f} (0: Healthy, 1: Diseased)")
            print(f"Yield Prediction: {yield_pred:.2f} hg/ha")
            print(f"Recommendation: {recommendation}")

        elif choice == '4':
            print(" Exiting program. Goodbye!")
            break

        else:
            print(" Invalid input. Please select from 1-4.")

# Run the system
if __name__ == "__main__":
    main_menu()


# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Step 1: Load Movie Data
def load_movie_data(file_path):
    movies_df = pd.read_csv(file_path)
    print(f"\n✅ Dataset loaded successfully with {len(movies_df)} records.")
    print(f"Available columns: {list(movies_df.columns)}\n")
    return movies_df

# Step 2: Preprocess Data
def preprocess_data(movies_df):
    possible_genre_cols = ['genres', 'genre', 'Category']
    possible_desc_cols = ['description', 'overview', 'summary', 'plot', 'storyline']

    genre_col = next((c for c in possible_genre_cols if c in movies_df.columns), None)
    desc_col = next((c for c in possible_desc_cols if c in movies_df.columns), None)

    if genre_col and desc_col:
        movies_df['content'] = movies_df[genre_col].fillna('') + ' ' + movies_df[desc_col].fillna('')
    elif genre_col:
        movies_df['content'] = movies_df[genre_col].fillna('')
    elif desc_col:
        movies_df['content'] = movies_df[desc_col].fillna('')
    else:
        raise KeyError("❌ No suitable text column (genre/description) found in dataset!")

    return movies_df

# Step 3: Create TF-IDF Matrix
def create_tfidf_matrix(movies_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    print(f"✅ TF-IDF matrix created with shape {tfidf_matrix.shape}")
    return tfidf_matrix

# Step 4: Fit Nearest Neighbors Model
def build_knn_model(tfidf_matrix):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(tfidf_matrix)
    return model

# Step 5: Get Recommendations
def get_recommendations(title, movies_df, tfidf_matrix, model, top_n=3):
    if title not in movies_df['title'].values:
        print(f"❌ Movie '{title}' not found in dataset.")
        return pd.Series(dtype=str)

    idx = movies_df.index[movies_df['title'] == title][0]
    movie_vec = tfidf_matrix[idx]
    distances, indices = model.kneighbors(movie_vec, n_neighbors=top_n+1)
    rec_indices = indices.flatten()[1:]  # skip itself
    return movies_df['title'].iloc[rec_indices]

# Step 6: Menu
def main_menu():
    file_path = "RS-A2_A3_movie.csv"  # ✅ your dataset
    movies_df = None
    tfidf_matrix = None
    model = None

    while True:
        print("\n🎥 Movie Recommendation System")
        print("1. Load Movie Dataset")
        print("2. Preprocess and Build Model")
        print("3. Get Movie Recommendations")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == '1':
            movies_df = load_movie_data(file_path)
            if 'title' not in movies_df.columns:
                raise KeyError("❌ Dataset must have a 'title' column.")
            print("Movies loaded successfully.")

        elif choice == '2':
            if movies_df is None:
                print("⚠️ Please load the dataset first (option 1).")
                continue
            movies_df = preprocess_data(movies_df)
            tfidf_matrix = create_tfidf_matrix(movies_df)
            model = build_knn_model(tfidf_matrix)
            print("✅ Model built and ready for recommendations.")

        elif choice == '3':
            if movies_df is None or model is None:
                print("⚠️ Please load data and build model first (options 1 and 2).")
                continue
            movie_title = input("Enter movie title from dataset: ")
            recs = get_recommendations(movie_title, movies_df, tfidf_matrix, model, top_n=3)
            if not recs.empty:
                print(f"\n🎬 Top recommendations for '{movie_title}':")
                for r in recs:
                    print(f" - {r}")
            else:
                print("No recommendations found.")

        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1–4.")

# Step 7: Run
if __name__ == "__main__":
    main_menu()


# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# ----------------------------
# Step 1: Load Data
# ----------------------------
def load_data():
    file_path = "RS-A2_A3_Filtered_Ratings.csv"
    ratings_df = pd.read_csv(file_path)

    # Create dummy movie dataset with movieId, title, and genres
    unique_movies = ratings_df['movieId'].unique()[:10]  # take first 10 for simplicity
    movies_data = {
        'movieId': unique_movies,
        'title': [
            f"Movie {i}" for i in range(1, len(unique_movies) + 1)
        ],
        'genres': [
            'Action, Adventure', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi',
            'Horror', 'Romance', 'Crime', 'Fantasy', 'Animation'
        ][:len(unique_movies)]
    }
    movies_df = pd.DataFrame(movies_data)

    return movies_df, ratings_df

# ----------------------------
# Step 2: Preprocess (Content-Based)
# ----------------------------
def preprocess_content_based(movies_df):
    movies_df['content'] = movies_df['genres']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# ----------------------------
# Step 3: Preprocess (Collaborative)
# ----------------------------
def preprocess_collaborative(ratings_df, movies_df):
    merged_df = pd.merge(ratings_df, movies_df, on='movieId')
    user_item_matrix = merged_df.pivot(index='userId', columns='title', values='rating').fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix.values)
    svd = TruncatedSVD(n_components=2)
    latent_matrix = svd.fit_transform(user_item_sparse)
    return latent_matrix, user_item_matrix

# ----------------------------
# Step 4: Recommendation Functions
# ----------------------------
def get_content_based_recommendations(title, movies_df, cosine_sim, top_n=3):
    if title not in movies_df['title'].values:
        print(f"Movie '{title}' not found in dataset.")
        return []
    idx = movies_df.index[movies_df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    return [movies_df['title'].iloc[i[0]] for i in sim_scores]

def get_collaborative_recommendations(user_id, ratings_df, latent_matrix, user_item_matrix, top_n=3):
    if user_id not in user_item_matrix.index:
        print(f"User ID {user_id} not found.")
        return []
    user_idx = user_item_matrix.index.get_loc(user_id)
    distances = pairwise_distances(latent_matrix[user_idx].reshape(1, -1), latent_matrix, metric='cosine')[0]
    similar_users = distances.argsort()[1:top_n+1]
    recommended_movies = []
    for idx in similar_users:
        uid = user_item_matrix.index[idx]
        movies = ratings_df[ratings_df['userId'] == uid]['movieId'].tolist()
        recommended_movies.extend(movies)
    return list(set(recommended_movies))

def hybrid_recommendations(user_id, movie_title, movies_df, ratings_df, cosine_sim, latent_matrix, user_item_matrix):
    content_recs = get_content_based_recommendations(movie_title, movies_df, cosine_sim)
    collab_recs = get_collaborative_recommendations(user_id, ratings_df, latent_matrix, user_item_matrix)
    combined = list(set(content_recs + [f"Movie ID {m}" for m in collab_recs]))
    return combined

# ----------------------------
# Step 5: Main Menu
# ----------------------------
def main_menu():
    movies_df, ratings_df = None, None
    cosine_sim = None
    latent_matrix, user_item_matrix = None, None

    while True:
        print("\n🎥 Movie Recommendation System")
        print("1. Load Dataset")
        print("2. Preprocess and Compute Similarity")
        print("3. Get Movie Recommendations")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == '1':
            movies_df, ratings_df = load_data()
            print("✅ Dataset loaded successfully!")
            print(f"Movies available: {list(movies_df['title'])}")
            print(f"Users available: {ratings_df['userId'].unique()[:5]} ...")

        elif choice == '2':
            if movies_df is None or ratings_df is None:
                print("Please load dataset first.")
                continue
            cosine_sim = preprocess_content_based(movies_df)
            latent_matrix, user_item_matrix = preprocess_collaborative(ratings_df, movies_df)
            print("✅ Data preprocessed and similarity computed successfully.")

        elif choice == '3':
            if cosine_sim is None or latent_matrix is None:
                print("Please preprocess data first (option 2).")
                continue
            try:
                user_id = int(input("Enter user ID: "))
                movie_title = input("Enter movie title (e.g., Movie 1): ")
                recs = hybrid_recommendations(user_id, movie_title, movies_df, ratings_df, cosine_sim, latent_matrix, user_item_matrix)
                print(f"\n🎬 Hybrid Recommendations for user {user_id} and movie '{movie_title}':")
                for r in recs:
                    print(f" - {r}")
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '4':
            print("👋 Goodbye!")
            break

        else:
            print("Invalid choice. Please select 1-4.")

# ----------------------------
# Step 6: Run the Program
# ----------------------------
if __name__ == "__main__":
    main_menu()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# Step 1: Load and preprocess dataset
# ---------------------------
def load_dataset():
    file_path = "RS-A4_SEER Breast Cancer Dataset .csv"
    df = pd.read_csv(file_path)

    print("✅ Dataset loaded successfully!")
    print("Columns available:\n", df.columns.tolist()[:15], "...\n")

    # Drop unnamed or empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()

    # Encode string/object columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Detect target variable
    possible_targets = ['Survival_Status', 'Status', 'Outcome']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        raise ValueError("❌ No suitable target column found. Ensure 'Survival_Status' or similar exists.")

    print(f"Target variable assumed as: {target_col}\n")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, df.columns

# ---------------------------
# Step 2: Split data
# ---------------------------
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------
# Step 3: Train Model
# ---------------------------
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Step 4: Evaluate Model
# ---------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return accuracy

# ---------------------------
# Step 5: Prognosis prediction (interactive)
# ---------------------------
def prognosis_prediction(model, X):
    print("\nEnter patient details for prognosis prediction (or type 'exit' to quit):")
    while True:
        proceed = input("Proceed? (yes/exit): ").strip().lower()
        if proceed == "exit":
            print("👋 Exiting program.")
            break
        elif proceed == "yes":
            print("Please enter patient features:")
            input_data = []
            for col in X.columns:
                value = input(f"{col}: ")
                try:
                    input_data.append(float(value))
                except:
                    input_data.append(0)  # fallback for missing or invalid entries
            pred = model.predict([input_data])[0]
            if pred == 0:
                print("Prognosis Recommendation: ✅ Benign results. Routine monitoring suggested, but follow up with a healthcare provider.\n")
            else:
                print("Prognosis Recommendation: ⚠️ Malignant risk detected. Immediate medical attention recommended.\n")
        else:
            print("Please type 'yes' to continue or 'exit' to quit.")

# ---------------------------
# Step 6: Main function
# ---------------------------
def main():
    X, y, columns = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    prognosis_prediction(model, X)

# ---------------------------
# Run program
# ---------------------------
if __name__ == "__main__":
    main()


# In[5]:


import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from math import sqrt

# ------------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------------
def load_real_data():
    """
    Load the real movie rating dataset (Filtered Ratings CSV)
    Columns: userId, movieId, rating
    """
    file_path = "RS-A2_A3_Filtered_Ratings.csv"
    df = pd.read_csv(file_path)

    print(" Dataset loaded successfully!\n")
    print("Columns available:\n", df.columns.tolist(), "\n")
    print(df.head())

    df = df[['userId', 'movieId', 'rating']].dropna()

    # Reduce dataset size if it's huge (optional safety)
    if len(df) > 200000:
        df = df.sample(200000, random_state=42)

    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)

    return df

# ------------------------------------------------------------
# Step 2: Create User-Item Matrix
# ------------------------------------------------------------
def create_user_item_matrix(df):
    user_item = df.pivot_table(index='userId', columns='movieId', values='rating')
    print("\nUser-Item Interaction Matrix (sample):")
    print(user_item.head())
    return user_item

# ------------------------------------------------------------
# Step 3: Train SVD-based Matrix Factorization
# ------------------------------------------------------------
def train_svd(user_item_matrix, k=10):
    """
    Perform memory-optimized SVD using sparse representation.
    """
    matrix_filled = user_item_matrix.fillna(0).values
    user_ratings_mean = np.mean(matrix_filled, axis=1)
    matrix_demeaned = matrix_filled - user_ratings_mean.reshape(-1, 1)

    # Perform truncated SVD (low-rank approximation)
    U, sigma, Vt = svds(matrix_demeaned, k=k)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings,
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)
    return preds_df

# ------------------------------------------------------------
# Step 4: Evaluate Model
# ------------------------------------------------------------
def evaluate_model(preds_df, original_df):
    y_true = []
    y_pred = []

    # Only evaluate on known ratings
    for _, row in original_df.iterrows():
        user = row['userId']
        item = row['movieId']
        actual = row['rating']
        try:
            predicted = preds_df.loc[user, item]
            y_true.append(actual)
            y_pred.append(predicted)
        except KeyError:
            continue

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n Root Mean Squared Error (RMSE) on known ratings: {rmse:.4f}")

# ------------------------------------------------------------
# Step 5: Recommend Items
# ------------------------------------------------------------
def recommend_items(preds_df, user_id, original_df, num_recommendations=5):
    user_row = preds_df.loc[user_id]
    rated_items = original_df[original_df['userId'] == user_id]['movieId'].tolist()
    recommendations = user_row.drop(labels=rated_items, errors='ignore').sort_values(ascending=False).head(num_recommendations)
    return recommendations

# ------------------------------------------------------------
# Step 6: Main Program
# ------------------------------------------------------------
def main():
    print(" Loading movie ratings dataset...")
    df = load_real_data()

    user_item_matrix = create_user_item_matrix(df)
    print("\n🔧 Training Matrix Factorization model using SVD (optimized)...")
    preds_df = train_svd(user_item_matrix, k=10)  # reduced k for less memory

    print("\n Predicted Ratings Matrix (sample):")
    print(preds_df.round(2).head())

    evaluate_model(preds_df, df)

    while True:
        try:
            user_id = int(input("\nEnter a User ID to get recommendations (or 0 to exit): "))
            if user_id == 0:
                print(" Exiting Recommendation System.")
                break
            if user_id not in user_item_matrix.index:
                print(" User ID not found. Try again.")
                continue

            recs = recommend_items(preds_df, user_id, df, num_recommendations=5)
            print(f"\n Top Recommendations for User {user_id}:")
            for movie, rating in recs.items():
                print(f"Movie ID: {movie} | Predicted Rating: {rating:.2f}")
        except ValueError:
            print(" Invalid input. Please enter a valid integer User ID.")

# ------------------------------------------------------------
# Run the Program
# ------------------------------------------------------------
if __name__ == "__main__":
    main()


# In[6]:


# ================================================================
#  CROP-WISE YIELD PREDICTION + DISEASE DETECTION + RECOMMENDATION
# Dataset: RS-A1_yield.csv
# ================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
file_path = "RS-A1_yield.csv"
df = pd.read_csv(file_path)

print("\n Dataset Loaded Successfully!")
print("Columns:", df.columns.tolist())
print(df.head(), "\n")

# Select necessary columns
df = df[['Area', 'Item', 'Year', 'hg/ha_yield',
         'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']].dropna()

# ---------------------------------------------------------------
# CNN MODEL (SIMULATED CROP DISEASE DETECTION)
# ---------------------------------------------------------------
X_train_images = np.random.rand(100, 64, 64, 3)
y_train_images = np.random.randint(2, size=100)

cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_images, y_train_images, epochs=2, batch_size=10, verbose=1)

# ---------------------------------------------------------------
# TRAIN RANDOM FOREST MODEL FOR EACH CROP TYPE
# ---------------------------------------------------------------
crop_models = {}
for crop, data in df.groupby('Item'):
    X = data[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
    y = data['hg/ha_yield']
    if len(X) > 3:  # only train if there are enough data points
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        crop_models[crop] = model

print(f"\n Trained models for {len(crop_models)} crops successfully!")

# ---------------------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------------------
def recommend(disease_prob, yield_pred, crop_name):
    if disease_prob >= 0.5:
        return f" {crop_name}: Disease detected! Apply pesticide immediately."
    elif yield_pred < 0.5 * np.median(df['hg/ha_yield']):
        return f" {crop_name}: Low yield predicted. Improve irrigation and soil nutrients."
    else:
        return f" {crop_name}: Crop is healthy and yield is optimal."

# ---------------------------------------------------------------
# MENU-DRIVEN INTERFACE
# ---------------------------------------------------------------
while True:
    print("\n======  CROP RECOMMENDER MENU ======")
    print("1. View Sample Data")
    print("2. Predict Yield for All Crops")
    print("3. Full Recommendation for All Crops")
    print("4. Exit")
    choice = input("Enter choice: ")

    if choice == '1':
        print(df.head())

    elif choice == '2':
        print("\n Predicted Yields for Each Crop:")
        for crop, model in crop_models.items():
            sample = df[df['Item'] == crop].sample(1)
            X_sample = sample[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
            yield_pred = model.predict(X_sample)[0]
            print(f"{crop:25s} --> {yield_pred:.2f} hg/ha")

    elif choice == '3':
        print("\n Crop-wise Recommendations:")
        for crop, model in crop_models.items():
            sample = df[df['Item'] == crop].sample(1)
            X_sample = sample[['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]
            yield_pred = model.predict(X_sample)[0]
            disease_prob = cnn.predict(np.random.rand(1, 64, 64, 3))[0][0]
            print(recommend(disease_prob, yield_pred, crop))

    elif choice == '4':
        print(" Exiting... Stay sustainable and grow smart!")
        break

    else:
        print(" Invalid choice. Try again.")


# In[7]:


#RS ASSIGNMENT-2
#MENU-DRIVEN CONTENT-BASED MOVIE RECOMMENDER SYSTEM (by Movie ID)

# 1. Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. Load dataset (ensure the file is in the same folder)
file_path = "RS-A2_A3_Filtered_Ratings.csv"
df = pd.read_csv(file_path)

# 3. Preprocess dataset
df.dropna(inplace=True)

# Combine all textual features to form "content" for similarity
text_cols = df.select_dtypes('object').columns
if len(text_cols) == 0:
    print("No text columns found for content-based filtering.")
    df['content'] = df.apply(lambda x: ' '.join(x.astype(str)), axis=1)
else:
    df['content'] = df[text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# 4. TF-IDF + Similarity computation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['content'])
cosine_sim = cosine_similarity(tfidf_matrix)

# 5. Recommendation Function (by movieId)
def recommend_by_id(movie_id, n=5):
    if 'movieId' not in df.columns:
        print("'movieId' column not found in dataset.")
        return
    if movie_id not in df['movieId'].values:
        print("Movie ID not found. Please try again.")
        return

    idx = df.index[df['movieId'] == movie_id][0]
    scores = list(enumerate(cosine_sim[idx]))
    similar = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    print(f"\nTop {n} movies similar to Movie ID {movie_id}:\n")
    for i, s in similar:
        print(f"Movie ID: {df['movieId'].iloc[i]}  (Similarity: {s:.2f})")

# 6. Menu System
while True:
    print("\n===== MOVIE RECOMMENDER MENU =====")
    print("1. View first 5 rows of dataset")
    print("2. Recommend similar movies (enter Movie ID)")
    print("3. View average similarity score")
    print("4. Exit")

    choice = input("\nEnter your choice (1-4): ")

    if choice == '1':
        print("\nFirst 5 rows of dataset:\n")
        print(df.head())

    elif choice == '2':
        try:
            movie_id = int(input("\nEnter Movie ID: "))
            recommend_by_id(movie_id)
        except ValueError:
            print("Please enter a valid numeric Movie ID.")

    elif choice == '3':
        print("\nAverage Similarity Between Movies:", cosine_sim.mean().round(4))

    elif choice == '4':
        print("\nExiting the program. Thank you!")
        break

    else:
        print("Invalid choice. Please enter a number between 1 and 4.") 


# In[8]:


# RS ASSIGNMENT -3
#  SIMPLE MENU-DRIVEN HYBRID MOVIE RECOMMENDER
# Dataset: RS-A2_A3_movie.csv
# (Now works with or without year in movie title)
# ===================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# --- Load Dataset ---
data = pd.read_csv("RS-A2_A3_movie.csv")
data.fillna('', inplace=True)

# Basic column setup
if 'title' not in data.columns:
    data.rename(columns={data.columns[0]: 'title'}, inplace=True)
if 'genres' not in data.columns:
    data['genres'] = ''

# Remove year from title for easy matching
data['title_clean'] = data['title'].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()

# --- Content-Based Filtering ---
data['content'] = data['genres'].astype(str) + ' ' + data.get('description', '')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# --- Collaborative Filtering (Matrix Factorization using SVD) ---
np.random.seed(42)
num_users = 5
ratings = pd.DataFrame({
    'user_id': np.repeat(range(1, num_users + 1), 5),
    'movie_id': np.random.randint(0, len(data), num_users * 5),
    'rating': np.random.randint(3, 6, num_users * 5)
})
pivot = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
svd = TruncatedSVD(n_components=3, random_state=42)
latent = svd.fit_transform(csr_matrix(pivot))

# --- Recommendation Functions ---
def content_recommend(title):
    """Recommend movies based on content similarity."""
    title = title.strip()
    if title not in data['title_clean'].values:
        print(" Movie not found.")
        return []
    idx = data[data['title_clean'] == title].index[0]
    sim_scores = cosine_sim[idx]
    top = sim_scores.argsort()[-6:-1][::-1]
    return data.iloc[top]['title'].tolist()

def collaborative_recommend(user_id):
    """Recommend movies based on similar users."""
    if user_id not in pivot.index:
        print(" User not found.")
        return []
    idx = list(pivot.index).index(user_id)
    sims = pairwise_distances(latent[idx].reshape(1, -1), latent, metric='cosine')[0]
    top_users = sims.argsort()[1:4]
    recs = ratings[ratings['user_id'].isin(top_users)]['movie_id'].unique()[:5]
    return data.iloc[recs]['title'].tolist()

def hybrid_recommend(user_id, title):
    """Combine content-based and collaborative recommendations."""
    cb = content_recommend(title)
    cf = collaborative_recommend(user_id)
    return list(set(cb + cf))[:10]

# --- Menu ---
while True:
    print("\n====== MOVIE RECOMMENDER MENU ======")
    print("1. View Sample Movies")
    print("2. Content-Based Recommendation")
    print("3. Collaborative Recommendation")
    print("4. Hybrid Recommendation")
    print("5. Exit")
    ch = input("Enter choice: ")

    if ch == '1':
        print(data[['title', 'genres']].head())

    elif ch == '2':
        t = input("Enter movie title: ")
        recs = content_recommend(t)
        print(" Recommendations:", recs)

    elif ch == '3':
        u = int(input("Enter user ID (1-5): "))
        recs = collaborative_recommend(u)
        print(" Recommendations:", recs)

    elif ch == '4':
        u = int(input("Enter user ID (1-5): "))
        t = input("Enter movie title: ")
        recs = hybrid_recommend(u, t)
        print(" Hybrid Recommendations:", recs)

    elif ch == '5':
        print(" Goodbye!")
        break

    else:
        print(" Invalid choice. Try again.")


# In[9]:


#RS ASSIGNMENT-4
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("RS-A4_SEER Breast Cancer Dataset .csv", sep=None, engine='python')
print(df)
# 2️⃣ Clean data
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.ffill().bfill()

# 3️⃣ Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# 4️⃣ Split data
target_col = "Status"
X = df.drop(columns=[target_col])
y = df[target_col]

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Evaluate model
y_pred = model.predict(X_test)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Dead', 'Alive'], yticklabels=['Dead', 'Alive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 9️⃣ Recommendation Section
print("\n🔍 Enter patient details to predict prognosis:")
sample = {}

# 🔹 Instead of 'Race', ask for 'Tumor Size'
for col in X.columns:
    if col == 'Race':
        value = float(input(f"Enter Tumor Size value: "))
    else:
        value = float(input(f"Enter value for {col}: "))
    sample[col] = value

new_data = pd.DataFrame([sample])
prediction = model.predict(new_data)[0]

if prediction == 1:
    print("\n💡 Recommendation: The patient is likely to SURVIVE (Alive).")
else:
    print("\n⚠️ Recommendation: The patient is likely to have a POOR PROGNOSIS (Dead).")


# In[10]:


#RS ASSIGNMENT-5
# ADVANCED RECOMMENDATION SYSTEM
# Using Matrix Factorization (SVD)
# Output Format Similar to Given Example
# ============================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

# Step 2: Load Dataset
data_path = "RS-A5_amazon_products_sales_data_cleaned.csv"
products = pd.read_csv(data_path)

# Step 3: Select Important Columns
products = products[['product_title', 'product_category', 'product_rating', 'discount_percentage']]

# Step 4: Simulate User-Item Ratings
np.random.seed(42)
num_users = 4       # keep small for easy-to-read output
num_items = 4

# randomly pick 4 products
selected_products = products.sample(n=num_items, random_state=42).reset_index(drop=True)

# create artificial user ratings
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'item_id': [1, 2, 3, 2, 3, 1, 4, 2, 4],
    'rating':  [5, 4, 3, 4, 5, 2, 4, 3, 5]
}
df = pd.DataFrame(data)

# Step 5: Create User-Item Matrix
user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Step 6: Apply Matrix Factorization (SVD)
svd = TruncatedSVD(n_components=2, random_state=42)
latent_matrix = svd.fit_transform(user_item_matrix)

# Step 7: Reconstruct Predicted Ratings
predicted_ratings = np.dot(latent_matrix, svd.components_)
pred_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Step 8: Evaluate RMSE
rmse = np.sqrt(mean_squared_error(user_item_matrix.values.flatten(), pred_df.values.flatten()))

# Step 9: Recommend Items
def recommend_items(user_id, n=2):
    user_ratings = pred_df.loc[user_id]
    rated_items = df[df['user_id'] == user_id]['item_id'].values
    recommendations = user_ratings.drop(rated_items).sort_values(ascending=False).head(n)
    return recommendations

# Step 10: Display Outputs (same format as your sample)
print("Predicted Rating Matrix:\n", pred_df.round(2))
print("\nRMSE:", round(rmse, 3))
print("\nTop Recommendations for User 1:")
print(recommend_items(1)) 


# In[1]:


# To check versions after installation
import numpy as np, pandas as pd, sklearn, keras, tensorflow, matplotlib, seaborn, scipy

print("✅ All libraries imported successfully!")


# In[ ]:


# Install all required libraries
get_ipython().system('pip install numpy pandas scikit-learn matplotlib seaborn keras tensorflow scipy')

