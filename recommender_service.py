import pandas as pd
from flask import Flask, jsonify, request
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import psycopg2

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
movie_mapper = {} 
movie_inv_mapper = {}
movie_user_matrix = None
n_samples_fit = 0 

def load_data_and_train():
   
    global model_knn, movie_mapper, movie_inv_mapper, movie_user_matrix, n_samples_fit
    
    print("Đang kết nối CSDL (localhost: movie_ai_app)...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="movie_ai_app",
            user="postgres",
            password="0000"
        )
        
        df = pd.read_sql('SELECT account_id, movie_id FROM "LikeMovie"', conn)
        conn.close()

        if df.empty:
            print("Không có dữ liệu 'LikeMovie' để huấn luyện.")
            return

        print(f"Tìm thấy {len(df)} lượt thích...")


        movie_ids = df['movie_id'].unique()
        user_ids = df['account_id'].unique()
        
        movie_mapper = {movie_id: i for i, movie_id in enumerate(movie_ids)}
        movie_inv_mapper = {i: movie_id for i, movie_id in enumerate(movie_ids)}
        
        user_mapper = {user_id: i for i, user_id in enumerate(user_ids)}
        
        movie_indices = df['movie_id'].map(movie_mapper)
        user_indices = df['account_id'].map(user_mapper)
        
        movie_user_matrix = csr_matrix((
            [1] * len(df), 
            (movie_indices, user_indices)
        ), shape=(len(movie_ids), len(user_ids)))

    
        n_samples_fit = movie_user_matrix.shape[0]
  
        print("Đang huấn luyện model KNN...")
        model_knn.fit(movie_user_matrix)
        print(f"Model đã sẵn sàng! (Đã huấn luyện trên {n_samples_fit} phim)")
        
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")


app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_id = request.args.get('movie_id', type=int)

    if not movie_id:
        return jsonify({"error": "Cần 'movie_id'"}), 400
        
    if movie_id not in movie_mapper:
        return jsonify({"movie_ids": []}) 

    movie_index = movie_mapper[movie_id]
    movie_vector = movie_user_matrix[movie_index]


    k_neighbors = min(11, n_samples_fit)


    if k_neighbors <= 1:
        return jsonify({"movie_ids": []}) # Không có hàng xóm

    distances, indices = model_knn.kneighbors(movie_vector, n_neighbors=k_neighbors)
   
    
    neighbor_indices = indices[0][1:]
    
    recommended_movie_ids_raw = [movie_inv_mapper[i] for i in neighbor_indices]

    recommended_movie_ids = [int(movie_id) for movie_id in recommended_movie_ids_raw]
    
    return jsonify({"movie_ids": recommended_movie_ids})

if __name__ == '__main__':
    load_data_and_train()
    app.run(port=5001, debug=True)