from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp {
    background: #ffffff !important;
    color: #182338;
}

/* Keep Streamlit's header in the layout so the page doesn't jump or clip */
header[data-testid="stHeader"] {
    background: #ffffff !important;
    height: 3rem !important;
}
[data-testid="stToolbar"] {
    background: transparent !important;
}

/* Main content spacing */
.stApp .main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f5f8fc !important;
    border-right: 1px solid #d5dfeb;
}

/* Dashboard heading */
.topbar {
    box-sizing: border-box;
    width: 100%;
    min-height: 60px;
    padding: 16px 20px;
    margin: 0 0 22px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    background: #101a2e !important;
    color: #ffffff !important;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.5;
    border: 0 !important;
    outline: 0 !important;
    box-shadow: none !important;
    overflow-wrap: anywhere;
}
.topbar small {
    color: #aab7ce !important;
    font-size: 11px;
    font-weight: 400;
}

/* Section headings */
.section-title {
    text-align: center;
    color: #24334d;
    font-size: 23px;
    font-weight: 750;
    line-height: 1.35;
    margin: 10px 0 3px;
}
.section-subtitle {
    text-align: center;
    color: #78869b;
    font-size: 13px;
    margin-bottom: 22px;
}

/* Book information */
.book-title {
    color: #182338;
    font-weight: 750;
    font-size: 12px;
    line-height: 1.35;
    min-height: 32px;
    text-align: center;
    overflow-wrap: anywhere;
}
.book-author {
    color: #6e7c91;
    font-size: 11px;
    text-align: center;
    margin-top: 6px;
}
.book-stats {
    color: #1765c0;
    font-weight: 700;
    font-size: 10px;
    text-align: center;
    margin-top: 10px;
}

/* Sidebar navigation */
.sidebar-label {
    color: #202d43;
    font-weight: 750;
    font-size: 14px;
    margin: 8px 0 12px;
}
/* Change navigation text only */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #000000 !important;
}

/* Sidebar information box */
.sidebar-note {
    background: #eef3f9;
    border: 1px solid #d5dfeb;
    border-radius: 4px;
    padding: 10px;
    color: #52627a;
    font-size: 11px;
    line-height: 1.65;
    margin-top: 22px;
}

/* Recommend button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #2864e8;
    border: 1px solid #2864e8;
    color: #ffffff;
    border-radius: 4px;
    font-weight: 700;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data():
    names = ["popular.pkl", "pt.pkl", "books.pkl", "similarity_scores.pkl"]
    missing = [name for name in names if not (BASE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing: " + ", ".join(missing) +
            ". Keep these files beside app.py."
        )

    data = []
    for name in names:
        with open(BASE_DIR / name, "rb") as file:
            data.append(pickle.load(file))
    return data


def val(value, fallback=""):
    try:
        return fallback if pd.isna(value) else str(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else fallback


def render_book(book, stats=False):
    title = val(book.get("Book-Title", book.get("title", "Unknown title")), "Unknown title")
    author = val(book.get("Book-Author", book.get("author", "Unknown author")), "Unknown author")
    image = book.get("Image-URL-M", book.get("image", ""))

    with st.container(border=True):
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            st.image(image, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="height:150px;background:#e7edf5;border:1px solid #b9c8dc;
                display:flex;align-items:center;justify-content:center;color:#718096;
                font-size:12px;">Book Cover Thumbnail</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="book-author">{author}</div>', unsafe_allow_html=True)

        if stats:
            bits = []
            try:
                rating = book.get("avg_rating")
                if pd.notna(rating):
                    bits.append(f"{float(rating):.2f} ★")
            except (TypeError, ValueError):
                pass

            try:
                votes = book.get("num_ratings")
                if pd.notna(votes):
                    bits.append(f"{int(votes):,} votes")
            except (TypeError, ValueError):
                pass

            if bits:
                st.markdown(
                    f'<div class="book-stats">{" | ".join(bits)}</div>',
                    unsafe_allow_html=True,
                )


def get_recommendations(title, pt, books, similarity_scores, limit=5):
    found = (pt.index == title).nonzero()[0]
    if not len(found):
        return []

    scores = similarity_scores[int(found[0])]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[1:limit + 1]
    results = []

    for idx, _score in ranked:
        recommended_title = pt.index[idx]
        matches = books.loc[books["Book-Title"] == recommended_title]
        if matches.empty:
            continue

        book = matches.drop_duplicates("Book-Title").iloc[0]
        results.append({
            "Book-Title": book.get("Book-Title", "Unknown title"),
            "Book-Author": book.get("Book-Author", "Unknown author"),
            "Image-URL-M": book.get("Image-URL-M", ""),
        })

    return results


def main():
    st.markdown(
        """
        <div class="topbar">
            <span>📚 Book Recommendation System &nbsp; | &nbsp; Streamlit Dashboard</span>
            <small>localhost:8501</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        popular, pt, books, similarity_scores = load_data()
    except Exception as error:
        st.error(f"Could not load the recommendation data: {error}")
        st.stop()

    st.sidebar.markdown(
        '<div class="sidebar-label">Navigation Menu</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        ["Top 50 Books", "Recommend Book"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        f'<div class="sidebar-note">'
        f'• Dataset: Book-Crossing<br>'
        f'• Popular books: {len(popular):,}<br>'
        f'• Book titles: {len(pt.index):,}<br>'
        f'• Recommendation engine: Ready'
        f'</div>',
        unsafe_allow_html=True,
    )

    if page == "Top 50 Books":
        st.markdown(
            '<div class="section-title">Top 50 Most Popular Books in Repository</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Ranked by average rating across titles with at least 50 user ratings</div>',
            unsafe_allow_html=True,
        )

        if popular.empty:
            st.warning("No popular books found.")
            return

        for start in range(0, min(50, len(popular)), 4):
            columns = st.columns(4, gap="medium")
            for column, (_, book) in zip(columns, popular.iloc[start:start + 4].iterrows()):
                with column:
                    render_book(book, stats=True)

    else:
        st.markdown(
            '<div class="section-title">Collaborative Filtering Recommendation Engine</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-subtitle">Select a book and discover titles with similar rating patterns</div>',
            unsafe_allow_html=True,
        )

        titles = sorted(str(title) for title in pt.index if pd.notna(title))
        if not titles:
            st.warning("No titles available in pt.pkl.")
            return

        left, right = st.columns([4, 1], vertical_alignment="bottom")
        default = titles.index("The Da Vinci Code") if "The Da Vinci Code" in titles else 0

        with left:
            selected = st.selectbox("Selected book", titles, index=default)
        with right:
            clicked = st.button("Recommend", type="primary", use_container_width=True)

        if "rec_title" not in st.session_state:
            st.session_state.rec_title = None
        if clicked:
            st.session_state.rec_title = selected

        chosen = st.session_state.rec_title
        if chosen:
            results = get_recommendations(chosen, pt, books, similarity_scores)
            st.markdown(
                f'<div style="text-align:center;font-weight:750;color:#27364e;'
                f'margin:10px 0 18px">Top 5 Books Similar to "{chosen}"</div>',
                unsafe_allow_html=True,
            )
            if results:
                columns = st.columns(5, gap="small")
                for column, book in zip(columns, results):
                    with column:
                        render_book(book)
            else:
                st.info("No recommendations found for this title.")
        else:
            st.info("Choose a title and click Recommend to see similar books.")


if __name__ == "__main__":
    main()

