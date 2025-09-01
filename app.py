import streamlit as st
import pandas as pd
import camelot
import tempfile

st.set_page_config(page_title="Camelot PDF Table Extractor", layout="wide")
st.title("📄 Camelot PDF Table Extractor (Configurable)")

st.sidebar.header("🛠️ Camelot Parameters")

# File upload
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

# Flavor selector
flavor = st.sidebar.selectbox("Flavor", options=["stream", "lattice"])

# Shared parameters
pages = st.sidebar.text_input("Pages", value="all")
strip_text = st.sidebar.text_input("strip_text", value="\n")

# Initialize kwargs
stream_kwargs = {}
lattice_kwargs = {}

# Parameters for flavor=stream
if flavor == "stream":
    edge_tol = st.sidebar.slider("edge_tol", min_value=0, max_value=100, value=50)
    row_tol = st.sidebar.slider("row_tol", min_value=0, max_value=50, value=10)
    detect_vertical = st.sidebar.checkbox("detect_vertical (layout_kwargs)", value=True)

    stream_kwargs = {
        "edge_tol": edge_tol,
        "row_tol": row_tol,
        "layout_kwargs": {"detect_vertical": detect_vertical},
    }

# Parameters for flavor=lattice
if flavor == "lattice":
    line_scale = st.sidebar.slider("line_scale", min_value=1, max_value=100, value=40)
    copy_text = st.sidebar.checkbox("copy_text", value=True)
    shift_text_raw = st.sidebar.text_input("shift_text (comma-separated)", value="")

    # Clean and validate shift_text
    shift_text_list = [s.strip() for s in shift_text_raw.split(",") if s.strip()]
    shift_text_valid = shift_text_list if shift_text_list else None

    lattice_kwargs = {
        "line_scale": line_scale,
        "copy_text": copy_text,
    }
    if shift_text_valid:
        lattice_kwargs["shift_text"] = shift_text_valid

# Main execution
if pdf_file and st.button("📥 Extract Tables"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    st.info(f"Using flavor: `{flavor}`")

    try:
        base_kwargs = {
            "flavor": flavor,
            "pages": pages,
            "strip_text": strip_text,
        }

        if flavor == "stream":
            base_kwargs.update(stream_kwargs)
        elif flavor == "lattice":
            base_kwargs.update(lattice_kwargs)

        with st.spinner("Parsing PDF with Camelot..."):
            tables = camelot.read_pdf(tmp_path, **base_kwargs)

        if tables.n == 0:
            st.warning("No tables detected. Try adjusting parameters.")
        else:
            st.success(f"✅ {tables.n} table(s) extracted.")
            for i, table in enumerate(tables):
                st.subheader(f"📊 Table {i+1}")
                df = table.df
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download Table {i+1}", csv, file_name=f"table_{i+1}.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
