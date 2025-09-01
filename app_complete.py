# All code, variables and comments are in English (as requested).

import streamlit as st
import pandas as pd
import tempfile
from typing import Dict, Any, List, Optional

# ---- Robust import for camelot-py ----
try:
    import camelot  # camelot-py usually exposes read_pdf at top level
    if not hasattr(camelot, "read_pdf"):
        import camelot.io as camelot  # fallback
except Exception as e:
    st.error(f"Failed to import camelot-py: {e}")
    st.stop()

st.set_page_config(page_title="Camelot PDF Table Extractor (All Flavors)", layout="wide")
st.title("📄 Camelot PDF Table Extractor — Stream, Lattice, Network, Hybrid")

with st.sidebar:
    st.header("🛠️ Global Parameters")

# ---------- File upload ----------
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

# ---------- Global read_pdf-level params ----------
with st.sidebar:
    pages = st.text_input("pages", value="all", help="Comma/range string like '1,3-5,10' or 'all'.")
    password = st.text_input("password (optional)", value="", type="password")
    parallel = st.checkbox("parallel", value=False, help="Process pages in parallel.")
    suppress_stdout = st.checkbox("suppress_stdout", value=False, help="Suppress logs/warnings.")
    debug = st.checkbox("debug", value=False, help="Store debug info during parsing.")
    st.divider()

# ---------- Flavor selection ----------
flavor = st.sidebar.selectbox(
    "Flavor",
    options=["stream", "lattice", "network", "hybrid"],
    index=0,
    help="Hybrid combines Network (text-alignment) + Lattice (lines).",
)

# ---------- Common text/region params ----------
with st.sidebar.expander("📐 Areas / Regions / Columns"):
    table_areas_raw = st.text_area(
        "table_areas (one 'x1,y1,x2,y2' per line)",
        value="",
        help="Top-left (x1,y1) and bottom-right (x2,y2) in PDF space. Leave blank if not used.",
    )
    table_regions_raw = st.text_area(
        "table_regions (one 'x1,y1,x2,y2' per line)",
        value="",
        help="Restrict search to these regions. Leave blank if not used.",
    )
    columns_multiline = st.text_area(
        "columns (one comma-separated list per line)",
        value="",
        help="Provide one line per table area. Leave blank if not used. If one list only, it applies to whole page.",
    )

# Helpers to parse multi-line CSV-like inputs
def parse_bbox_lines(s: str) -> Optional[List[str]]:
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines if lines else None

def parse_columns_lines(s: str) -> Optional[List[str]]:
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines if lines else None

table_areas = parse_bbox_lines(table_areas_raw)
table_regions = parse_bbox_lines(table_regions_raw)
columns = parse_columns_lines(columns_multiline)

# ---------- Text handling flags ----------
with st.sidebar.expander("🔤 Text Handling"):
    strip_text = st.text_input("strip_text (characters to strip)", value="")
    split_text = st.checkbox("split_text", value=False, help="Split strings merged across cells.")
    flag_size = st.checkbox("flag_size", value=False, help="Flag superscripts/subscripts via <s> tags.")

# ---------- layout_kwargs (PDFMiner LAParams) ----------
with st.sidebar.expander("🧩 layout_kwargs (LAParams)"):
    detect_vertical = st.checkbox("detect_vertical", value=True)
    char_margin = st.number_input("char_margin (optional)", value=0.0, step=0.5, format="%.1f")
    line_margin = st.number_input("line_margin (optional)", value=0.0, step=0.5, format="%.1f")
    word_margin = st.number_input("word_margin (optional)", value=0.0, step=0.5, format="%.1f")
    # Build layout_kwargs only with provided (non-zero) margins to avoid surprising effects
    layout_kwargs: Dict[str, Any] = {"detect_vertical": bool(detect_vertical)}
    # Only include margins if user changed them from zero
    if char_margin > 0:
        layout_kwargs["char_margin"] = float(char_margin)
    if line_margin > 0:
        layout_kwargs["line_margin"] = float(line_margin)
    if word_margin > 0:
        layout_kwargs["word_margin"] = float(word_margin)

# ---------- Flavor-specific params ----------
stream_net_kwargs: Dict[str, Any] = {}
lattice_kwargs: Dict[str, Any] = {}

if flavor in ("stream", "network", "hybrid"):
    with st.sidebar.expander("📏 Stream/Network parameters (text-based)"):
        edge_tol = st.slider("edge_tol", min_value=0, max_value=1000, value=50)
        row_tol = st.slider("row_tol", min_value=0, max_value=100, value=2)
        column_tol = st.slider("column_tol", min_value=0, max_value=50, value=0)
        stream_net_kwargs = {
            "edge_tol": int(edge_tol),      # applies to stream/network
            "row_tol": int(row_tol),
            "column_tol": int(column_tol),
        }

if flavor in ("lattice", "hybrid"):
    with st.sidebar.expander("🧮 Lattice parameters (line-based)"):
        # See note: API main shows default 40; class Lattice shows 15. We default to 40.
        line_scale = st.slider(
            "line_scale",
            min_value=1, max_value=200, value=40,
            help="Default 40 per API; class doc shows 15. Larger → detect smaller lines; too large may detect text as lines."
        )
        process_background = st.checkbox("process_background", value=False)
        copy_text_opts = st.multiselect("copy_text", options=["h", "v"], default=[], help="Copy text across spans.")
        shift_text_opts = st.multiselect("shift_text", options=["l", "r", "t", "b"], default=["l", "t"], help="Flow direction for spanning text.")
        line_tol = st.slider("line_tol", min_value=0, max_value=20, value=2)
        joint_tol = st.slider("joint_tol", min_value=0, max_value=20, value=2)
        threshold_blocksize = st.slider("threshold_blocksize", min_value=3, max_value=51, value=15, step=2)
        threshold_constant = st.slider("threshold_constant", min_value=-10, max_value=10, value=-2)
        iterations = st.slider("iterations", min_value=0, max_value=10, value=0)
        backend = st.selectbox("backend", options=["pdfium", "ghostscript"], index=0)
        use_fallback = st.checkbox("use_fallback", value=True)
        resolution = st.slider("resolution (dpi)", min_value=72, max_value=600, value=300, step=12)

        lattice_kwargs = {
            "line_scale": int(line_scale),
            "process_background": bool(process_background),
            "copy_text": copy_text_opts if copy_text_opts else None,
            "shift_text": shift_text_opts if shift_text_opts else ["l", "t"],
            "line_tol": int(line_tol),
            "joint_tol": int(joint_tol),
            "threshold_blocksize": int(threshold_blocksize),
            "threshold_constant": int(threshold_constant),
            "iterations": int(iterations),
            "backend": backend,
            "use_fallback": bool(use_fallback),
            "resolution": int(resolution),
        }

# ---------- Debug panel to verify effective kwargs ----------
with st.sidebar.expander("🔎 Debug – Effective kwargs"):
    base_kwargs: Dict[str, Any] = {
        "flavor": flavor,
        "pages": pages,
        "password": password if password else None,
        "suppress_stdout": bool(suppress_stdout),
        "parallel": bool(parallel),
        "layout_kwargs": layout_kwargs,
        "table_regions": table_regions,
        "table_areas": table_areas,
        "columns": columns,
        "split_text": bool(split_text),
        "flag_size": bool(flag_size),
        "strip_text": strip_text,
        "debug": bool(debug),
    }
    # Merge flavor-specific kwargs
    if flavor in ("stream", "network"):
        base_kwargs.update(stream_net_kwargs)
    elif flavor == "lattice":
        base_kwargs.update(lattice_kwargs)
    elif flavor == "hybrid":
        # Hybrid accepts both groups
        base_kwargs.update(stream_net_kwargs)
        base_kwargs.update(lattice_kwargs)

    st.write("camelot module path:", getattr(camelot, "__file__", "unknown"))
    st.json({k: v for k, v in base_kwargs.items() if v is not None})

# ---------- Caching extractor ----------
@st.cache_data(show_spinner=False)
def extract_tables_cached(pdf_bytes: bytes, kwargs: Dict[str, Any]) -> List[pd.DataFrame]:
    """
    Cached wrapper around camelot.read_pdf. The cache key depends on bytes and kwargs.
    """
    # Write bytes to a temp file each run (required by camelot)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Remove None-valued keys to avoid camelot receiving unexpected None
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Call camelot with validated kwargs
    tables = camelot.read_pdf(tmp_path, **clean_kwargs)

    # Convert TableList -> list of DataFrames (cache-serializable)
    return [t.df for t in tables]

# ---------- Reactive run ----------
if pdf_file is None:
    st.info("Upload a PDF to start extracting tables.")
else:
    st.info(f"Using flavor: `{flavor}`")
    try:
        with st.spinner("Parsing PDF with Camelot..."):
            dfs = extract_tables_cached(pdf_file.getvalue(), base_kwargs)

        if len(dfs) == 0:
            st.warning("No tables detected. Try adjusting parameters (areas/regions, tolerances, or flavor).")
        else:
            st.success(f"✅ {len(dfs)} table(s) extracted.")
            for i, df in enumerate(dfs, start=1):
                st.subheader(f"📊 Table {i}")
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download Table {i} (CSV)",
                    data=csv,
                    file_name=f"table_{i}.csv",
                    mime="text/csv",
                    key=f"dl_{i}",
                )

    except TypeError as te:
        st.error(f"Type error: {te}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
