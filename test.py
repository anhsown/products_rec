import streamlit as st
import pandas as pd
from gensim.corpora import Dictionary
import pickle
import re

# Tải dữ liệu cần thiết
@st.cache_resource
def load_data():
    # Tải DataFrame
    df = pd.read_csv('San_pham.csv')  # Thay bằng đường dẫn thật của file
    data = pd.read_csv('data_gem.csv')  # Dữ liệu content_gem_re
    # Tải dictionary
    dictionary = Dictionary.load('dictionary.dict')
    # Tải tfidf và index
    with open('tfidf_matrix.pkl', 'rb') as tfidf_file, open('similarity_index.pkl', 'rb') as index_file:
        tfidf = pickle.load(tfidf_file)
        index = pickle.load(index_file)
    return df, data, dictionary, tfidf, index

# Hàm gợi ý sản phẩm
def get_recommendations(product_id, data, df, dictionary, index, tfidf):
    # Tìm index của sản phẩm dựa trên mã sản phẩm
    product_index = df[df['ma_san_pham'] == product_id].index[0]

    # Lấy nội dung của sản phẩm đầu vào
    view_content = data['content_gem_re'][product_index]

    # Tách từ nếu view_content là chuỗi
    if isinstance(view_content, str):
        view_content = re.sub(r'[^\w\s]', '', view_content).split()  # Hoặc dùng tokenizer khác

    # Chuyển nội dung thành vector BoW
    kw_vector = dictionary.doc2bow(view_content)

    # Tính toán mức độ tương tự giữa sản phẩm đầu vào và các sản phẩm khác
    sim = index[tfidf[kw_vector]]

    # Sắp xếp mức độ tương tự giảm dần
    sim_sort = sorted(enumerate(sim), key=lambda item: -item[1])

    # Khởi tạo danh sách các sản phẩm được gợi ý
    recommended_products = []

    # Lặp qua các sản phẩm tương tự
    for idx, sim_score in sim_sort[1:]:
        # Kiểm tra nếu điểm trung bình của sản phẩm này >= 3
        if df.iloc[idx]['diem_trung_binh'] >= 3:
            recommended_products.append(idx)
        # Nếu đã có đủ 5 sản phẩm, thoát vòng lặp
        if len(recommended_products) == 6:
            break

    # Trả về thông tin của 6 sản phẩm được gợi ý từ DataFrame gốc
    return df.iloc[recommended_products]

# Hàm hiển thị các sản phẩm gợi ý
def display_recommended_products(recommended_products, cols=3):
    # Chia giao diện thành các cột
    for i in range(0, len(recommended_products), cols):
        cols_layout = st.columns(cols)
        for j, col in enumerate(cols_layout):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:
                    # Hiển thị thông tin sản phẩm
                    # st.image("product_image.png", use_container_width=True)  # Thêm ảnh sản phẩm nếu có
                    st.write(f"**Mã sản phẩm:** {product['ma_san_pham']}")
                    st.write(f"**Tên sản phẩm:** {product['ten_san_pham']}")
                    st.write(f"**Điểm trung bình:** {product['diem_trung_binh']}")
                    st.write(f"**Giá sản phẩm:** {product['gia_ban']:,.0f} VND")   # Hiển thị giá sản phẩm

                    # Mô tả sản phẩm trong expander
                    with st.expander("Xem mô tả sản phẩm"):
                        st.write(product['mo_ta'][:200] + '...')  # Hiển thị mô tả sản phẩm, có thể thay đổi chiều dài


# Load dữ liệu
df, data, dictionary, tfidf, index = load_data()

###### Giao diện Streamlit ######
st.image('hasaki_banner.jpg', use_container_width=True)

st.title("Hệ thống gợi ý sản phẩm")
st.subheader("Nhập mã sản phẩm để nhận gợi ý tương tự")


# Nhập ID sản phẩm
product_id_input = st.text_input("Mã sản phẩm", "")

if st.button("Gợi ý sản phẩm"):
    try:
        # Chuyển đổi input thành số
        product_id = int(product_id_input)
        
        # Kiểm tra nếu ID sản phẩm tồn tại
        if product_id in df['ma_san_pham'].values:
            # Hiển thị thông tin sản phẩm nhập vào
            st.subheader("Thông tin sản phẩm nhập vào:")
            product_info = df[df['ma_san_pham'] == product_id].iloc[0]

            # Hiển thị thông tin từng dòng
            st.write(f"**Mã sản phẩm:** {product_info['ma_san_pham']}")
            st.write(f"**Tên sản phẩm:** {product_info['ten_san_pham']}")
            st.write(f"**Điểm trung bình:** {product_info['diem_trung_binh']}")
            st.write(f"**Mô tả sản phẩm:** {product_info['mo_ta'][:500]} ...")

            # Lấy danh sách sản phẩm được gợi ý
            recommendations = get_recommendations(product_id, data, df, dictionary, index, tfidf)
            st.subheader("Sản phẩm được gợi ý:")

            # Hiển thị sản phẩm gợi ý theo cách đẹp mắt
            display_recommended_products(recommendations)
        else:
            st.error("Mã sản phẩm không tồn tại trong hệ thống!")
    except ValueError:
        st.error("Mã sản phẩm phải là số!")