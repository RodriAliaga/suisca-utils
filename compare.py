# app.py
import io
import re
import pandas as pd
from difflib import SequenceMatcher
import streamlit as st
from scorers import SCORER_REGISTRY
from PIL import Image, ImageDraw, ImageFont
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False
try:
    import scorers.embed_cosine  # noqa: F401
    _EMBED_AVAILABLE = True
except Exception:
    _EMBED_AVAILABLE = False

st.set_page_config(page_title="Comparador de sugerencias", layout="wide")
st.title("📒 Comparador de sugerencias")

st.write("Sube un archivo .xlsx o .xls para visualizarlo.")

with st.sidebar:
    try:
        st.write("Scorers cargados:", list(SCORER_REGISTRY.keys()))
    except Exception:
        pass
    try:
        _tok_usr_nonnull = int(df_out["score_cliente_vs_usuario_tok"].notna().sum()) if "score_cliente_vs_usuario_tok" in df_out.columns else 0
        _tok_ia_nonnull = int(df_out["score_cliente_vs_IA_tok"].notna().sum()) if "score_cliente_vs_IA_tok" in df_out.columns else 0
        st.write("Token Set filas calculadas:", {"PROPUESTA USUARIO": _tok_usr_nonnull, "PROPUESTA IA": _tok_ia_nonnull})
    except Exception:
        pass

archivo = st.file_uploader("Elegir archivo Excel", type=["xlsx", "xls"])

@st.cache_data(show_spinner=False)
def obtener_hojas(bytes_data: bytes):
    # Devuelve solo los nombres de hojas (tipos serializables)
    xls = pd.ExcelFile(io.BytesIO(bytes_data))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def leer_hoja(bytes_data: bytes, hoja: str, header_row: int) -> pd.DataFrame:
    # Lee y devuelve un DataFrame (serializable por pickle)
    return pd.read_excel(
        io.BytesIO(bytes_data), sheet_name=hoja, header=header_row, engine="openpyxl"
    )

# Alias con el nombre usado en algunos entornos/llamadas previas
# Importante: devolver SOLO tipos serializables por pickle (p.ej., DataFrame)
@st.cache_data(show_spinner=False)
def leer_excel(bytes_data: bytes, hoja: str, header_row: int) -> pd.DataFrame:
    return pd.read_excel(
        io.BytesIO(bytes_data), sheet_name=hoja, header=header_row, engine="openpyxl"
    )

if archivo is not None:
    try:
        bytes_data = archivo.getvalue()
        hojas = obtener_hojas(bytes_data)
        hoja = st.selectbox("Selecciona la hoja", hojas, index=0)

        # Opciones de visualización
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            max_filas = st.number_input("Filas a mostrar", min_value=5, max_value=10000, value=100, step=5)
        with col2:
            header_row = st.number_input("Fila de encabezado (0-index)", min_value=0, value=0, step=1)
        with col3:
            dtype_preview = st.checkbox("Mostrar tipos de dato", value=False)

        # Leemos la hoja seleccionada con el header que elija el usuario (en caché)
        # Usamos leer_excel (alias) para evitar cualquier conflicto con caché previa
        df = leer_excel(bytes_data, hoja, int(header_row))

        st.caption(f"Hoja **{hoja}** — {df.shape[0]:,} filas × {df.shape[1]:,} columnas")

        if dtype_preview:
            st.write("**Tipos de columna:**")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

        st.dataframe(df.head(int(max_filas)), use_container_width=True)

        st.subheader("Comparar columnas por similitud (0–1)")
        cols = list(df.columns.astype(str))
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        with c1:
            col_cliente = st.selectbox("Solicitud del cliente (1)", options=cols, index=0 if cols else None, key="sel_cliente")
        with c2:
            col_usuario = st.selectbox("PROPUESTA USUARIO (1)", options=cols, index=min(1, len(cols)-1) if len(cols) > 1 else 0, key="sel_usuario")
        with c3:
            cols_ia = st.multiselect("PROPUESTA IA (1+)", options=cols, default=[cols[0]] if cols else [], key="sel_ia")
        # Todos los métodos se calculan y muestran de forma independiente; sin selector

        def _norm(x):
            if pd.isna(x):
                return ""
            s = str(x).strip().lower()
            return s

        def _sim(a, b) -> float:
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            return float(SequenceMatcher(None, a2, b2).ratio())

        def _sim_tfidf(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            tfidf_fn = SCORER_REGISTRY.get("tfidf_char_cosine")
            if tfidf_fn is None:
                if not st.session_state.get("_tfidf_warned", False):
                    st.warning("No se pudo usar TF-IDF (char 3–5). Instale scikit-learn para habilitarlo.")
                    st.session_state["_tfidf_warned"] = True
                return None
            try:
                return float(tfidf_fn(a2, b2))
            except Exception as e:
                if not st.session_state.get("_tfidf_warned", False):
                    st.warning(f"No se pudo usar TF-IDF (char 3–5). Detalle: {e}")
                    st.session_state["_tfidf_warned"] = True
                return None

        def _tokenize_ws(s: str):
            # Palabras alfanuméricas (incluye acentos unicode)
            return re.findall(r"\w+", s, flags=re.UNICODE)

        def _fallback_token_set_ratio(a: str, b: str) -> float:
            ta = set(_tokenize_ws(a))
            tb = set(_tokenize_ws(b))
            if not ta and not tb:
                return 1.0
            if not ta or not tb:
                return 0.0
            inter = len(ta & tb)
            denom = len(ta) + len(tb)
            if denom == 0:
                return 0.0
            return (2.0 * inter) / float(denom)

        def _sim_token_set(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            tok_fn = SCORER_REGISTRY.get("token_set_ratio")
            if tok_fn is None:
                # Fallback si RapidFuzz no está disponible
                if not st.session_state.get("_token_warned", False):
                    st.warning("RapidFuzz no disponible; usando fallback de conjunto de tokens (aprox.).")
                    st.session_state["_token_warned"] = True
                st.session_state["_token_fallback_used"] = True
                return float(_fallback_token_set_ratio(a2, b2))
            try:
                return float(tok_fn(a2, b2))
            except Exception as e:
                # Fallback ante error de import o ejecución
                if not st.session_state.get("_token_warned", False):
                    st.warning(f"No se pudo usar RapidFuzz token_set_ratio. Usando fallback (aprox). Detalle: {e}")
                    st.session_state["_token_warned"] = True
                st.session_state["_token_fallback_used"] = True
                return float(_fallback_token_set_ratio(a2, b2))

        def _sim_embed(a, b):
            a2, b2 = _norm(a), _norm(b)
            if not a2 and not b2:
                return 1.0
            if not a2 or not b2:
                return 0.0
            fn = SCORER_REGISTRY.get("embed_cosine")
            if fn is None:
                if not st.session_state.get("_embed_warned", False):
                    st.warning("No se pudo usar Sentence-Transformers (embed_cosine). Instale sentence-transformers para habilitarlo.")
                    st.session_state["_embed_warned"] = True
                return None
            try:
                return float(fn(a2, b2))
            except Exception as e:
                if not st.session_state.get("_embed_warned", False):
                    st.warning(f"No se pudo usar Sentence-Transformers (embed_cosine). Detalle: {e}")
                    st.session_state["_embed_warned"] = True
                return None

        if cols:
            df_out = df.copy()

            if col_cliente and col_usuario:
                df_out["score_cliente_vs_usuario"] = [
                    _sim(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])
                ]
                # TF-IDF paralelo (crear columna siempre; puede contener NA)
                tfidf_scores_usr = [_sim_tfidf(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                df_out["score_cliente_vs_usuario_tfidf"] = tfidf_scores_usr
                # Token Set paralelo (columna estándar "_tok") — siempre crear columna
                tok_scores_usr = [_sim_token_set(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                df_out["score_cliente_vs_usuario_tok"] = tok_scores_usr
                # Embed cosine paralelo (crear columna siempre; puede contener NA)
                try:
                    embed_scores_usr = [_sim_embed(a, b) for a, b in zip(df_out[col_cliente], df_out[col_usuario])]
                except Exception:
                    embed_scores_usr = [None] * len(df_out)
                df_out["score_cliente_vs_usuario_embed"] = embed_scores_usr
                # (se eliminan variantes duplicadas *_token_set; usar sufijo _tok)

            if cols_ia:
                best_scores_cli = []
                best_cols_cli = []
                best_scores_usr = []
                best_cols_usr = []
                # TF-IDF parallel
                best_scores_cli_tfidf = []
                best_scores_usr_tfidf = []
                # Token-set parallel (columna estándar "_tok")
                best_scores_cli_tok = []
                best_scores_usr_tok = []
                # Embed parallel
                best_scores_cli_embed = []
                best_scores_usr_embed = []

                for _, row in df_out.iterrows():
                    val_cli = row[col_cliente] if col_cliente else None
                    val_usr = row[col_usuario] if col_usuario else None
                    scores_cli = [(_sim(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr = [(_sim(val_usr, row[c]), c) for c in cols_ia]
                    # TF-IDF scores (may be None)
                    scores_cli_tfidf = [(_sim_tfidf(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr_tfidf = [(_sim_tfidf(val_usr, row[c]), c) for c in cols_ia]
                    # Token-set scores (may be None) – estándar "_tok"
                    scores_cli_tok = [(_sim_token_set(val_cli, row[c]), c) for c in cols_ia]
                    scores_usr_tok = [(_sim_token_set(val_usr, row[c]), c) for c in cols_ia]
                    # Embed scores (may be None)
                    try:
                        scores_cli_embed = [(_sim_embed(val_cli, row[c]), c) for c in cols_ia]
                        scores_usr_embed = [(_sim_embed(val_usr, row[c]), c) for c in cols_ia]
                    except Exception:
                        scores_cli_embed = []
                        scores_usr_embed = []
                    sc_cli, bc_cli = max(scores_cli, key=lambda t: t[0]) if scores_cli else (None, None)
                    sc_usr, bc_usr = max(scores_usr, key=lambda t: t[0]) if scores_usr else (None, None)
                    best_scores_cli.append(sc_cli)
                    best_cols_cli.append(bc_cli)
                    best_scores_usr.append(sc_usr)
                    best_cols_usr.append(bc_usr)

                    # For TF-IDF, ignore None when taking max
                    sc_cli_tfidf = None
                    if scores_cli_tfidf:
                        vals = [t[0] for t in scores_cli_tfidf if t[0] is not None]
                        sc_cli_tfidf = max(vals) if vals else None
                    sc_usr_tfidf = None
                    if scores_usr_tfidf:
                        vals = [t[0] for t in scores_usr_tfidf if t[0] is not None]
                        sc_usr_tfidf = max(vals) if vals else None
                    best_scores_cli_tfidf.append(sc_cli_tfidf)
                    best_scores_usr_tfidf.append(sc_usr_tfidf)

                    # Token-set (estándar _tok), ignore None when taking max
                    sc_cli_tok = None
                    if scores_cli_tok:
                        vals = [t[0] for t in scores_cli_tok if t[0] is not None]
                        sc_cli_tok = max(vals) if vals else None
                    sc_usr_tok = None
                    if scores_usr_tok:
                        vals = [t[0] for t in scores_usr_tok if t[0] is not None]
                        sc_usr_tok = max(vals) if vals else None
                    best_scores_cli_tok.append(sc_cli_tok)
                    best_scores_usr_tok.append(sc_usr_tok)
                    # Embed, ignore None when taking max
                    sc_cli_embed = None
                    if scores_cli_embed:
                        vals = [t[0] for t in scores_cli_embed if t[0] is not None]
                        sc_cli_embed = max(vals) if vals else None
                    sc_usr_embed = None
                    if scores_usr_embed:
                        vals = [t[0] for t in scores_usr_embed if t[0] is not None]
                        sc_usr_embed = max(vals) if vals else None
                    best_scores_cli_embed.append(sc_cli_embed)
                    best_scores_usr_embed.append(sc_usr_embed)

                df_out["score_cliente_vs_IA"] = best_scores_cli
                df_out["score_usuario_vs_IA"] = best_scores_usr
                df_out["mejor_col_IA_para_usuario"] = best_cols_usr
                # TF-IDF columns (crear siempre; pueden contener NA)
                df_out["score_cliente_vs_IA_tfidf"] = best_scores_cli_tfidf
                df_out["score_usuario_vs_IA_tfidf"] = best_scores_usr_tfidf
                # Token-set columns (estándar _tok) — siempre crear columnas
                df_out["score_cliente_vs_IA_tok"] = best_scores_cli_tok
                df_out["score_usuario_vs_IA_tok"] = best_scores_usr_tok
                # Embed columns (crear siempre; pueden contener NA)
                df_out["score_cliente_vs_IA_embed"] = best_scores_cli_embed
                df_out["score_usuario_vs_IA_embed"] = best_scores_usr_embed
            else:
                best_cols_cli = []

            # Solo mostrar columnas seleccionadas + scores respecto al CLIENTE
            # Construcción de listas por método
            score_cols_difflib_display = [c for c in ["score_cliente_vs_usuario", "score_cliente_vs_IA"] if c in df_out.columns]
            score_cols_tfidf_display = [c for c in ["score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"] if c in df_out.columns]
            score_cols_tok_display = [c for c in ["score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"] if c in df_out.columns]
            score_cols_embed_display = [c for c in ["score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"] if c in df_out.columns]

            # Selección automática: prioriza Token Set si existe; si no, Difflib.
            has_tok_nonnull = False
            if score_cols_tok_display:
                try:
                    has_tok_nonnull = any(pd.to_numeric(df_out[c], errors="coerce").notna().any() for c in score_cols_tok_display)
                except Exception:
                    has_tok_nonnull = False
            if has_tok_nonnull:
                score_cols_display = score_cols_tok_display
                primary_usr_col, primary_ia_col = "score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"
            else:
                score_cols_display = score_cols_difflib_display
                primary_usr_col, primary_ia_col = "score_cliente_vs_usuario", "score_cliente_vs_IA"

            # Score final por fila (método primario para resaltado)
            if score_cols_display:
                df_out["score_fila_cliente"] = (
                    df_out[score_cols_display].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            # Scores por fila independientes por método
            cols_diff = [c for c in ["score_cliente_vs_usuario", "score_cliente_vs_IA"] if c in df_out.columns]
            if cols_diff:
                df_out["score_fila_cliente_diff"] = (
                    df_out[cols_diff].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_tfidf = [c for c in ["score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"] if c in df_out.columns]
            if cols_tfidf:
                df_out["score_fila_cliente_tfidf"] = (
                    df_out[cols_tfidf].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_tok = [c for c in ["score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"] if c in df_out.columns]
            if cols_tok:
                df_out["score_fila_cliente_tok"] = (
                    df_out[cols_tok].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )
            cols_embed = [c for c in ["score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"] if c in df_out.columns]
            if cols_embed:
                df_out["score_fila_cliente_embed"] = (
                    df_out[cols_embed].apply(pd.to_numeric, errors="coerce").max(axis=1)
                )

            # Calcular ganador_columna y bandera de empate sin crear columna 'decision'
            eps = 1e-9
            # Usar columnas primarias según selección de método
            if primary_ia_col or primary_usr_col:
                s_ia = pd.to_numeric(df_out.get(primary_ia_col), errors="coerce") if primary_ia_col in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr = pd.to_numeric(df_out.get(primary_usr_col), errors="coerce") if primary_usr_col in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                # serie con mejor columna IA por fila (si existe), sin exponer 'mejor_col_IA_para_cliente'
                best_col_cli_series = pd.Series(best_cols_cli, index=df_out.index) if best_cols_cli else pd.Series([pd.NA]*len(df_out), index=df_out.index)

                ganador = []
                empate_flags = []
                for idx in df_out.index:
                    ia = s_ia.loc[idx]
                    us = s_usr.loc[idx]
                    if pd.notna(ia) and pd.notna(us):
                        if abs(float(ia) - float(us)) <= eps:
                            ganador.append(best_col_cli_series.loc[idx])  # empate: asignamos IA
                            empate_flags.append(True)
                        elif float(ia) > float(us):
                            ganador.append(best_col_cli_series.loc[idx])
                            empate_flags.append(False)
                        else:
                            ganador.append("PROPUESTA USUARIO")
                            empate_flags.append(False)
                    elif pd.notna(ia):
                        ganador.append(best_col_cli_series.loc[idx])
                        empate_flags.append(False)
                    elif pd.notna(us):
                        ganador.append("PROPUESTA USUARIO")
                        empate_flags.append(False)
                    else:
                        ganador.append(pd.NA)
                        empate_flags.append(False)

                df_out["ganador_columna"] = ganador
                df_out["empate"] = empate_flags

            # Ganadores por método (usuario / IA / empate) — para depuración por fila
            def _winner_series(s_usr: pd.Series, s_ia: pd.Series, eps_local: float = 1e-9) -> list:
                out = []
                for u, i in zip(s_usr, s_ia):
                    u_na = pd.isna(u)
                    i_na = pd.isna(i)
                    if not u_na and not i_na:
                        du = float(u)
                        di = float(i)
                        if abs(di - du) <= eps_local:
                            out.append("IGUALES")
                        elif di > du:
                            out.append("PROPUESTA IA")
                        else:
                            out.append("PROPUESTA USUARIO")
                    elif not i_na:
                        out.append("PROPUESTA IA")
                    elif not u_na:
                        out.append("PROPUESTA USUARIO")
                    else:
                        out.append(pd.NA)
                return out

            # Difflib
            if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                s_usr_d = pd.to_numeric(df_out.get("score_cliente_vs_usuario"), errors="coerce") if "score_cliente_vs_usuario" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_d = pd.to_numeric(df_out.get("score_cliente_vs_IA"), errors="coerce") if "score_cliente_vs_IA" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_diff"] = _winner_series(s_usr_d, s_ia_d, eps)

            # TF-IDF
            if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                s_usr_t = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tfidf"), errors="coerce") if "score_cliente_vs_usuario_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_t = pd.to_numeric(df_out.get("score_cliente_vs_IA_tfidf"), errors="coerce") if "score_cliente_vs_IA_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_tfidf"] = _winner_series(s_usr_t, s_ia_t, eps)

            # Token Set
            if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                s_usr_k = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tok"), errors="coerce") if "score_cliente_vs_usuario_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_k = pd.to_numeric(df_out.get("score_cliente_vs_IA_tok"), errors="coerce") if "score_cliente_vs_IA_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_tok"] = _winner_series(s_usr_k, s_ia_k, eps)

            # Embed Cosine
            if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                s_usr_e = pd.to_numeric(df_out.get("score_cliente_vs_usuario_embed"), errors="coerce") if "score_cliente_vs_usuario_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_ia_e = pd.to_numeric(df_out.get("score_cliente_vs_IA_embed"), errors="coerce") if "score_cliente_vs_IA_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                df_out["ganador_embed"] = _winner_series(s_usr_e, s_ia_e, eps)

            # Armar el set de columnas a mostrar: seleccionadas + mejores IA + scores
            view_cols = []
            for csel in [col_cliente, col_usuario]:
                if csel and csel not in view_cols:
                    view_cols.append(csel)
            for cia in cols_ia:
                if cia not in view_cols:
                    view_cols.append(cia)
            # Metadatos útiles (se removió 'mejor_col_IA_para_cliente')
            # Scores
            view_cols.extend(score_cols_display)
            # Añadir ganador del método primario (si procede)
            if score_cols_display == score_cols_tok_display and "ganador_tok" in df_out.columns:
                view_cols.append("ganador_tok")
            elif score_cols_display == score_cols_difflib_display and "ganador_diff" in df_out.columns:
                view_cols.append("ganador_diff")
            # Ocultar agregados por fila en la vista (se mantienen para estadísticas)
            # Add TF-IDF scores to the view if present
            view_cols.extend(score_cols_tfidf_display)
            if score_cols_tfidf_display and "ganador_tfidf" in df_out.columns:
                view_cols.append("ganador_tfidf")
            # Add Token-set scores to the view if present
            view_cols.extend(score_cols_tok_display)
            if score_cols_tok_display and "ganador_tok" in df_out.columns:
                view_cols.append("ganador_tok")
            # Add Embed scores to the view if present
            view_cols.extend(score_cols_embed_display)
            if score_cols_embed_display and "ganador_embed" in df_out.columns:
                view_cols.append("ganador_embed")
            # Add Difflib scores if not already included
            for c in score_cols_difflib_display:
                if c not in view_cols:
                    view_cols.append(c)
            if score_cols_difflib_display and "ganador_diff" in df_out.columns:
                view_cols.append("ganador_diff")
            # No se muestra 'decision'
            if "ganador_columna" in df_out.columns:
                view_cols.append("ganador_columna")

            # Quitar columnas duplicadas manteniendo el orden
            if view_cols:
                _seen = set()
                _unique_cols = []
                for c in view_cols:
                    if c not in _seen:
                        _unique_cols.append(c)
                        _seen.add(c)
                view_cols = _unique_cols
            df_view = df_out[view_cols] if view_cols else df_out

            # Resaltado por fila: en scores, mayor en verde; IGUALES en amarillo
            def _highlight_row_best(row: pd.Series):
                styles = pd.Series("", index=row.index)
                eps = 1e-9
                pairs = [
                    ("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"),
                    ("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"),
                    ("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"),
                    ("score_cliente_vs_usuario", "score_cliente_vs_IA"),
                ]
                for ucol, icol in pairs:
                    if ucol in row.index or icol in row.index:
                        u = pd.to_numeric(row.get(ucol), errors="coerce") if ucol in row.index else pd.NA
                        i = pd.to_numeric(row.get(icol), errors="coerce") if icol in row.index else pd.NA
                        if pd.notna(u) and pd.notna(i):
                            if abs(float(i) - float(u)) <= eps:
                                if icol in row.index:
                                    styles[icol] = "background-color: #fff3cd"  # amarillo IGUALES
                                if ucol in row.index:
                                    styles[ucol] = "background-color: #fff3cd"
                            elif float(i) > float(u):
                                if icol in row.index:
                                    styles[icol] = "background-color: #d4edda"  # verde mayor
                            else:
                                if ucol in row.index:
                                    styles[ucol] = "background-color: #d4edda"  # verde mayor
                        elif pd.notna(i):
                            if icol in row.index:
                                styles[icol] = "background-color: #d4edda"  # verde único
                        elif pd.notna(u):
                            if ucol in row.index:
                                styles[ucol] = "background-color: #d4edda"  # verde único
                return styles

            # Resaltado para columnas de "ganador_*" y "ganador_columna"
            def _highlight_winner_cells(row: pd.Series):
                styles = pd.Series("", index=row.index)
                winners = [
                    "ganador_tok",
                    "ganador_tfidf",
                    "ganador_embed",
                    "ganador_diff",
                    "ganador_columna",
                ]
                for wcol in winners:
                    if wcol in row.index:
                        val = row.get(wcol)
                        if pd.isna(val):
                            continue
                        try:
                            sval = str(val)
                        except Exception:
                            sval = ""
                        if sval == "PROPUESTA USUARIO":
                            styles[wcol] = "background-color: #cfe2ff"  # azul claro
                        elif sval == "PROPUESTA IA":
                            styles[wcol] = "background-color: #ffd8a8"  # naranja claro
                        elif sval == "IGUALES":
                            styles[wcol] = "background-color: #fff3cd"  # amarillo claro
                return styles

            st.caption("Vista con resaltado por fila (mejor score en verde, por método)")
            try:
                st.dataframe(
                    df_view.head(int(max_filas)).
                        style.apply(_highlight_row_best, axis=1).
                        apply(_highlight_winner_cells, axis=1),
                    use_container_width=True,
                )
            except Exception:
                st.dataframe(df_view.head(int(max_filas)), use_container_width=True)

            # ---------- Descarga como PDF (paisaje) ----------
            def _load_font(size: int = 14):
                try:
                    return ImageFont.truetype("DejaVuSans.ttf", size)
                except Exception:
                    try:
                        return ImageFont.truetype("Arial.ttf", size)
                    except Exception:
                        return ImageFont.load_default()

            def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int):
                if text is None or (isinstance(text, float) and pd.isna(text)):
                    return [""]
                s = str(text)
                # simple wrap by words
                words = s.split()
                lines = []
                cur = ""
                for w in words:
                    test = (cur + (" " if cur else "") + w).strip()
                    if draw.textlength(test, font=font) <= max_width:
                        cur = test
                    else:
                        if cur:
                            lines.append(cur)
                        # if single word longer than max_width, hard split
                        if draw.textlength(w, font=font) <= max_width:
                            cur = w
                        else:
                            # naive hard wrap
                            tmp = ""
                            for ch in w:
                                if draw.textlength(tmp + ch, font=font) <= max_width:
                                    tmp += ch
                                else:
                                    if tmp:
                                        lines.append(tmp)
                                    tmp = ch
                            cur = tmp
                if cur:
                    lines.append(cur)
                return lines or [""]

            def _line_height(font_obj: ImageFont.ImageFont) -> int:
                try:
                    return int(getattr(font_obj, "size", None) or font_obj.getbbox("Ag")[3]) + 2
                except Exception:
                    try:
                        return int(font_obj.getsize("Ag")[1]) + 2
                    except Exception:
                        return 16

            def dataframe_to_pdf_bytes_pil(dfpdf: pd.DataFrame, title: str | None = None, scale: float = 1.0) -> bytes:
                # Configuración de página paisaje (ancho > alto)
                scale = max(1.0, float(scale))
                margin = int(30 * scale)
                pad_x = int(8 * scale)
                pad_y = int(6 * scale)
                header_bg = (217, 217, 217)  # gris plomo
                grid_color = (200, 200, 200)
                text_color = (0, 0, 0)
                # Colores de resaltado (alineados a la vista Streamlit)
                COL_GREEN = (212, 237, 218)   # mayor score pares
                COL_YELLOW = (255, 243, 205)  # IGUALES
                COL_BLUE = (207, 226, 255)    # PROPUESTA USUARIO en ganador_*
                COL_ORANGE = (255, 216, 168)  # PROPUESTA IA en ganador_*
                COL_DARKBLUE = (0, 51, 102)   # headers de ganador_* (ya no usado, se deja por compat)
                font = _load_font(int(14 * scale))
                font_bold = _load_font(int(16 * scale))

                # Definir anchos por tipo de columna
                def is_score_col(cname: str) -> bool:
                    s = str(cname)
                    return s.startswith("score_")
                def is_winner_col(cname: str) -> bool:
                    return str(cname).startswith("ganador_") or str(cname) == "ganador_columna"
                def _map_winner_text(v: object) -> str:
                    if isinstance(v, str):
                        if v == "PROPUESTA USUARIO":
                            return "USUARIO"
                        if v == "PROPUESTA IA":
                            return "IA"
                        if v == "IGUALES":
                            return "IGUALES"
                    return ""
                def _fmt_score(val) -> str:
                    try:
                        vv = pd.to_numeric(val, errors="coerce")
                        if pd.isna(vv):
                            return ""
                        return f"{float(vv):.2f}"
                    except Exception:
                        return ""

                col_widths = []
                for c in dfpdf.columns:
                    if is_winner_col(c):
                        col_widths.append(int(150 * scale))
                    elif is_score_col(c):
                        col_widths.append(int(120 * scale))
                    else:
                        col_widths.append(int(380 * scale))

                # Pares de columnas de score (para colorear mayor/IGUALES)
                score_pairs = [
                    ("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"),
                    ("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"),
                    ("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"),
                    ("score_cliente_vs_usuario", "score_cliente_vs_IA"),
                ]
                pair_index = {}
                for u, i in score_pairs:
                    pair_index[u] = (u, i)
                    pair_index[i] = (u, i)

                def _to_float(val):
                    try:
                        v = pd.to_numeric(val, errors="coerce")
                        if pd.isna(v):
                            return None
                        return float(v)
                    except Exception:
                        return None

                def _cell_bg_color(row: pd.Series, col_name: str):
                    # Ganadores: colorear según valor
                    if str(col_name).startswith("ganador_") or str(col_name) == "ganador_columna":
                        v = row.get(col_name)
                        if isinstance(v, str):
                            if v == "PROPUESTA USUARIO":
                                return COL_BLUE
                            if v == "PROPUESTA IA":
                                return COL_ORANGE
                            if v == "IGUALES":
                                return COL_YELLOW
                        return None
                    # Scores por pares: verde al mayor, amarillo si IGUALES
                    if col_name in pair_index:
                        ucol, icol = pair_index[col_name]
                        u = _to_float(row.get(ucol)) if ucol in row.index else None
                        i = _to_float(row.get(icol)) if icol in row.index else None
                        if (u is not None) and (i is not None):
                            if abs(i - u) <= 1e-9:
                                return COL_YELLOW
                            if (col_name == icol and i > u) or (col_name == ucol and u > i):
                                return COL_GREEN
                            return None
                        # Solo uno válido → ese en verde
                        if (col_name == icol and (i is not None) and (u is None)) or (col_name == ucol and (u is not None) and (i is None)):
                            return COL_GREEN
                        return None
                    return None

                # Calcular alto por fila en base al wrap
                dummy_img = Image.new("RGB", (1, 1), "white")
                draw = ImageDraw.Draw(dummy_img)
                header_h = 0
                # Header height
                for c, w in zip(dfpdf.columns, col_widths):
                    lines = _wrap_text(draw, str(c), font_bold, w - 2 * pad_x)
                    header_h = max(header_h, len(lines) * (_line_height(font_bold)) + 2 * pad_y)

                row_heights = []
                for _, row in dfpdf.iterrows():
                    rh = 0
                    for (c, w) in zip(dfpdf.columns, col_widths):
                        val = row.get(c)
                        if is_winner_col(c):
                            val = _map_winner_text(val)
                        elif is_score_col(c):
                            val = _fmt_score(val)
                        lines = _wrap_text(draw, val, font, w - 2 * pad_x)
                        rh = max(rh, len(lines) * (_line_height(font)) + 2 * pad_y)
                    row_heights.append(rh)

                table_width = sum(col_widths) + (len(col_widths) + 1)  # + grid lines
                table_height = header_h + sum(row_heights) + (len(row_heights) + 2)

                # Asegurar paisaje: si demasiado alto, incrementa ancho base
                img_w = max(int(1600 * scale), table_width + 2 * margin)
                img_h = table_height + 2 * margin
                # Si aún no es paisaje, ensanchar
                if img_w <= img_h:
                    img_w = img_h + int(400 * scale)

                img = Image.new("RGB", (img_w, img_h), "white")
                dr = ImageDraw.Draw(img)

                x = margin
                y = margin
                # Función de color de encabezado por columna (esquema de colores)
                def _header_bg_for_col(cname: str):
                    # Todos gris; ganador_* ahora fondo blanco
                    if str(cname).startswith("ganador_") or str(cname) == "ganador_columna":
                        return (255, 255, 255)
                    return header_bg

                # Dibujar headers con fondo por columna
                cx = x + 1
                for c, w in zip(dfpdf.columns, col_widths):
                    cell_w = w
                    # fondo por columna
                    hdr_bg = _header_bg_for_col(c)
                    dr.rectangle([cx, y, cx + cell_w, y + header_h], fill=hdr_bg, outline=grid_color)
                    txt_lines = _wrap_text(dr, str(c), font_bold, cell_w - 2 * pad_x)
                    ty = y + pad_y
                    # texto negro para ganador_* (fondo blanco)
                    hdr_text_color = text_color
                    for line in txt_lines:
                        dr.text((cx + pad_x, ty), line, fill=hdr_text_color, font=font_bold)
                        ty += _line_height(font_bold)
                    # vertical grid line
                    dr.line([(cx + cell_w, y), (cx + cell_w, y + header_h)], fill=grid_color)
                    cx += cell_w + 1

                # horizontal line under header
                dr.line([(x, y + header_h), (x + table_width, y + header_h)], fill=grid_color)

                # Dibujar filas
                cy = y + header_h + 1
                for r_idx in range(len(dfpdf)):
                    row = dfpdf.iloc[r_idx]
                    cx = x + 1
                    # calcular alto de la fila (precalculado)
                    rh = row_heights[r_idx]
                    # celdas
                    for c_idx, (c, w) in enumerate(zip(dfpdf.columns, col_widths)):
                        val = row.get(c)
                        if is_winner_col(c):
                            val = _map_winner_text(val)
                        elif is_score_col(c):
                            val = _fmt_score(val)
                        cell_w = w
                        # fondo según reglas
                        bg = _cell_bg_color(row, str(c))
                        if bg is not None:
                            dr.rectangle([cx, cy - 1, cx + cell_w, cy + rh], fill=bg)
                        # texto
                        lines = _wrap_text(dr, val, font, cell_w - 2 * pad_x)
                        ty = cy + pad_y
                        for line in lines:
                            dr.text((cx + pad_x, ty), line, fill=text_color, font=font)
                            ty += _line_height(font)
                        # bordes verticales
                        dr.line([(cx + cell_w, cy - 1), (cx + cell_w, cy + rh)], fill=grid_color)
                        cx += cell_w + 1
                    # bordes horizontales
                    dr.line([(x, cy + rh), (x + table_width, cy + rh)], fill=grid_color)
                    cy += rh + 1

                # Exportar a PDF
                bio = io.BytesIO()
                img.save(bio, format="PDF", resolution=300.0)
                return bio.getvalue()

            def dataframe_to_pdf_bytes_reportlab(dfpdf: pd.DataFrame, df_summary: pd.DataFrame | None = None, title: str | None = None, font_size: int = 9) -> bytes:
                # Colores (RGB 0-1)
                COL_GREEN = rl_colors.Color(212/255.0, 237/255.0, 218/255.0)
                COL_YELLOW = rl_colors.Color(255/255.0, 243/255.0, 205/255.0)
                COL_BLUE = rl_colors.Color(207/255.0, 226/255.0, 255/255.0)
                COL_ORANGE = rl_colors.Color(255/255.0, 216/255.0, 168/255.0)
                HEADER_BG = rl_colors.Color(217/255.0, 217/255.0, 217/255.0)  # gris plomo
                HEADER_DARKBLUE = rl_colors.Color(0/255.0, 51/255.0, 102/255.0)
                GRID = rl_colors.Color(200/255.0, 200/255.0, 200/255.0)

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
                styles = getSampleStyleSheet()
                body = styles["BodyText"]
                body.fontSize = font_size
                body.leading = int(font_size * 1.2)

                # Prepare data with Paragraphs for wrapping
                def is_score_col(cname: str) -> bool:
                    s = str(cname)
                    return s.startswith("score_")
                def is_winner_col(cname: str) -> bool:
                    return str(cname).startswith("ganador_") or str(cname) == "ganador_columna"
                def _map_winner_text(v: object) -> str:
                    if isinstance(v, str):
                        if v == "PROPUESTA USUARIO":
                            return "USUARIO"
                        if v == "PROPUESTA IA":
                            return "IA"
                        if v == "IGUALES":
                            return "IGUALES"
                    return ""
                def _fmt_score(val) -> str:
                    try:
                        vv = pd.to_numeric(val, errors="coerce")
                        if pd.isna(vv):
                            return ""
                        return f"{float(vv):.2f}"
                    except Exception:
                        return ""
                cols = list(dfpdf.columns)
                table_data = [[Paragraph(str(c), body) for c in cols]]
                for _, r in dfpdf.iterrows():
                    row_cells = []
                    for c in cols:
                        v = r.get(c)
                        if is_winner_col(c):
                            txt = _map_winner_text(v)
                        elif is_score_col(c):
                            txt = _fmt_score(v)
                        else:
                            if v is None or (isinstance(v, float) and pd.isna(v)):
                                txt = ""
                            else:
                                txt = str(v)
                        row_cells.append(Paragraph(txt, body))
                    table_data.append(row_cells)

                # Column widths relative to page width
                page_w, page_h = landscape(A4)
                avail = page_w - (doc.leftMargin + doc.rightMargin)
                weights = []
                for c in cols:
                    if is_winner_col(c):
                        weights.append(1.2)
                    elif is_score_col(c):
                        weights.append(1.0)
                    else:
                        weights.append(3.2)
                total_w = sum(weights)
                col_widths = [avail * (w/total_w) for w in weights]

                tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
                ts = [
                    ("GRID", (0,0), (-1,-1), 0.3, GRID),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                ]

                # Header backgrounds by column (gris para todos; ganador_* fondo blanco con texto negro)
                for c_idx, cname in enumerate(cols):
                    bg = HEADER_BG
                    ts.append(("BACKGROUND", (c_idx, 0), (c_idx, 0), bg))
                    sname = str(cname)
                    if sname.startswith("ganador_") or sname == "ganador_columna":
                        ts.append(("BACKGROUND", (c_idx, 0), (c_idx, 0), rl_colors.white))
                        ts.append(("TEXTCOLOR", (c_idx, 0), (c_idx, 0), rl_colors.black))

                # Helper for score pairs
                score_pairs = [
                    ("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok"),
                    ("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf"),
                    ("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed"),
                    ("score_cliente_vs_usuario", "score_cliente_vs_IA"),
                ]
                pair_map = {}
                for u,i in score_pairs:
                    pair_map[u] = (u,i)
                    pair_map[i] = (u,i)

                def _to_float(v):
                    try:
                        vv = pd.to_numeric(v, errors="coerce")
                        if pd.isna(vv):
                            return None
                        return float(vv)
                    except Exception:
                        return None

                # Color per cell
                for r_idx in range(1, len(table_data)):
                    row = dfpdf.iloc[r_idx-1]
                    for c_idx, cname in enumerate(cols):
                        sname = str(cname)
                        # winner cols
                        if sname.startswith("ganador_") or sname == "ganador_columna":
                            val = row.get(cname)
                            if isinstance(val, str):
                                if val == "PROPUESTA USUARIO":
                                    ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_BLUE))
                                elif val == "PROPUESTA IA":
                                    ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_ORANGE))
                                elif val == "IGUALES":
                                    ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_YELLOW))
                            continue
                        # score pairs
                        if sname in pair_map:
                            ucol, icol = pair_map[sname]
                            u = _to_float(row.get(ucol)) if ucol in row.index else None
                            i = _to_float(row.get(icol)) if icol in row.index else None
                            if (u is not None) and (i is not None):
                                if abs(i - u) <= 1e-9:
                                    ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_YELLOW))
                                elif (sname == icol and i > u) or (sname == ucol and u > i):
                                    ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_GREEN))
                            elif (u is None) ^ (i is None):
                                # only one valid
                                ts.append(("BACKGROUND", (c_idx, r_idx), (c_idx, r_idx), COL_GREEN))
                        # alignment for scores
                        if sname.startswith("score_"):
                            ts.append(("ALIGN", (c_idx, r_idx), (c_idx, r_idx), "CENTER"))

                tbl.setStyle(TableStyle(ts))
                elements = []
                if title:
                    elements.append(Paragraph(str(title), styles["Heading2"]))
                    elements.append(Spacer(1, 10))
                if df_summary is not None and not df_summary.empty:
                    # Build summary table
                    s_cols = list(df_summary.columns)
                    s_data = [[Paragraph(str(c), body) for c in s_cols]]
                    for _, rr in df_summary.iterrows():
                        s_row = []
                        for c in s_cols:
                            vv = rr.get(c)
                            s_row.append(Paragraph(str(vv), body))
                        s_data.append(s_row)
                    s_tbl = Table(s_data, repeatRows=1)
                    s_style = [
                        ("GRID", (0,0), (-1,-1), 0.3, GRID),
                        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
                        ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ]
                    # Highlight the larger between "Mejores USUARIO" and "Mejores IA" per row in green
                    try:
                        idx_usr = s_cols.index("Mejores USUARIO")
                        idx_ia = s_cols.index("Mejores IA")
                        for r_i, rr in enumerate(df_summary.itertuples(index=False), start=1):
                            try:
                                v_u = float(rr[idx_usr])
                            except Exception:
                                v_u = None
                            try:
                                v_i = float(rr[idx_ia])
                            except Exception:
                                v_i = None
                            if v_u is not None and v_i is not None:
                                if v_u > v_i:
                                    s_style.append(("BACKGROUND", (idx_usr, r_i), (idx_usr, r_i), COL_GREEN))
                                elif v_i > v_u:
                                    s_style.append(("BACKGROUND", (idx_ia, r_i), (idx_ia, r_i), COL_GREEN))
                                # if equal, no highlight requested
                    except ValueError:
                        pass
                    s_tbl.setStyle(TableStyle(s_style))
                    elements.append(s_tbl)
                    elements.append(Spacer(1, 12))
                elements.append(tbl)
                doc.build(elements)
                pdf = buf.getvalue()
                buf.close()
                return pdf

            st.subheader("Descargar PDF (paisaje)")
            try:
                df_export = df_view.head(int(max_filas)).copy()
                if _REPORTLAB_AVAILABLE:
                    fontsize = st.slider("Tamaño de letra (PDF texto)", min_value=5, max_value=14, value=8)
                    # Construir resumen para PDF
                    resumen_rows_pdf = []
                    def _compute_counts_pdf(col_usr: str, col_ia: str):
                        eps_local = 1e-9
                        s_ia_loc = pd.to_numeric(df_out.get(col_ia), errors="coerce") if col_ia in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                        s_usr_loc = pd.to_numeric(df_out.get(col_usr), errors="coerce") if col_usr in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                        valid = s_ia_loc.notna() & s_usr_loc.notna()
                        ia_w = valid & (s_ia_loc > s_usr_loc)
                        usr_w = valid & (s_usr_loc > s_ia_loc)
                        emp = valid & ((s_ia_loc - s_usr_loc).abs() <= eps_local)
                        return int(pd.to_numeric(usr_w, errors="coerce").fillna(0).sum()), int(pd.to_numeric(ia_w, errors="coerce").fillna(0).sum()), int(pd.to_numeric(emp, errors="coerce").fillna(0).sum())

                    if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                        u,i,e = _compute_counts_pdf("score_cliente_vs_usuario", "score_cliente_vs_IA")
                        resumen_rows_pdf.append({"Método": "Difflib", "Mejores USUARIO": u, "Mejores IA": i, "IGUALES": e})
                    if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                        u,i,e = _compute_counts_pdf("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf")
                        resumen_rows_pdf.append({"Método": "TF-IDF (char 3–5)", "Mejores USUARIO": u, "Mejores IA": i, "IGUALES": e})
                    if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                        u,i,e = _compute_counts_pdf("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok")
                        resumen_rows_pdf.append({"Método": "Token Set (RapidFuzz)", "Mejores USUARIO": u, "Mejores IA": i, "IGUALES": e})
                    if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                        u,i,e = _compute_counts_pdf("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed")
                        resumen_rows_pdf.append({"Método": "Embed Cosine (ST)", "Mejores USUARIO": u, "Mejores IA": i, "IGUALES": e})

                    df_resumen_pdf = pd.DataFrame(resumen_rows_pdf) if resumen_rows_pdf else pd.DataFrame()
                    file_label = getattr(archivo, "name", None)
                    title_text = f"Resumen general de métodos — Archivo: {file_label}" if file_label else "Resumen general de métodos"
                    pdf_bytes = dataframe_to_pdf_bytes_reportlab(df_export, df_resumen_pdf, title=title_text, font_size=int(fontsize))
                else:
                    st.caption("Para PDF de texto seleccionable instala 'reportlab'. Usando imagen como fallback.")
                    quality = st.select_slider("Calidad PDF (imagen)", options=["Normal", "Alta", "Máxima"], value="Alta")
                    scale_map = {"Normal": 1.0, "Alta": 1.5, "Máxima": 2.0}
                    pdf_bytes = dataframe_to_pdf_bytes_pil(df_export, title=f"{hoja}", scale=scale_map.get(quality, 1.5))
                st.download_button(
                    label="Descargar tabla en PDF",
                    data=pdf_bytes,
                    file_name="comparacion.pdf",
                    mime="application/pdf",
                )
            except Exception as _e:
                st.caption(f"No se pudo generar PDF: {_e}")

            # Estadísticas de conteo de "buenas" (verde) por columna (sin 'decision')
            if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES")
                s_ia = pd.to_numeric(df_out.get("score_cliente_vs_IA"), errors="coerce") if "score_cliente_vs_IA" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr = pd.to_numeric(df_out.get("score_cliente_vs_usuario"), errors="coerce") if "score_cliente_vs_usuario" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                ia_wins = (s_ia > s_usr)
                usr_wins = (s_usr > s_ia)
                empate_mask = s_ia.notna() & s_usr.notna() & ((s_ia - s_usr).abs() <= eps)

                total_usuario = int(usr_wins.sum())
                total_ia = int(ia_wins.sum())
                total_empates = int(empate_mask.sum())

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO", f"{total_usuario}")
                with cB:
                    st.metric("Mejores PROPUESTA IA", f"{total_ia}")
                with cC:
                    st.metric("IGUALES", f"{total_empates}")
                # Mostrar método (difflib)
                st.caption("Método de similitud usado: Difflib (SequenceMatcher)")

            # TF-IDF stats block if both columns exist
            if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (TF-IDF)")
                s_ia_t = pd.to_numeric(df_out.get("score_cliente_vs_IA_tfidf"), errors="coerce") if "score_cliente_vs_IA_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_t = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tfidf"), errors="coerce") if "score_cliente_vs_usuario_tfidf" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                ia_wins_t = (s_ia_t > s_usr_t)
                usr_wins_t = (s_usr_t > s_ia_t)
                empate_mask_t = s_ia_t.notna() & s_usr_t.notna() & ((s_ia_t - s_usr_t).abs() <= eps)

                total_usuario_t = int(usr_wins_t.sum())
                total_ia_t = int(ia_wins_t.sum())
                total_empates_t = int(empate_mask_t.sum())

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (TF-IDF)", f"{total_usuario_t}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (TF-IDF)", f"{total_ia_t}")
                with cC:
                    st.metric("IGUALES (TF-IDF)", f"{total_empates_t}")
                st.caption("Método de similitud usado: TF-IDF (char 3–5) + coseno")

            # Token-set stats block (estándar con sufijo _tok)
            if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (Token Set)")
                s_ia_tk = pd.to_numeric(df_out.get("score_cliente_vs_IA_tok"), errors="coerce") if "score_cliente_vs_IA_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_tk = pd.to_numeric(df_out.get("score_cliente_vs_usuario_tok"), errors="coerce") if "score_cliente_vs_usuario_tok" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                valid_tk = s_ia_tk.notna() & s_usr_tk.notna()
                ia_wins_tk = valid_tk & (s_ia_tk > s_usr_tk)
                usr_wins_tk = valid_tk & (s_usr_tk > s_ia_tk)
                empate_mask_tk = valid_tk & ((s_ia_tk - s_usr_tk).abs() <= eps)

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (Token Set)", f"{int(usr_wins_tk.sum())}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (Token Set)", f"{int(ia_wins_tk.sum())}")
                with cC:
                    st.metric("IGUALES (Token Set)", f"{int(empate_mask_tk.sum())}")
                if st.session_state.get("_token_fallback_used", False):
                    st.caption("Método de similitud usado: Token Set (Fallback aproximado; instale rapidfuzz para el oficial)")
                else:
                    st.caption("Método de similitud usado: RapidFuzz token_set_ratio")

            # Embed cosine stats block (Sentence-Transformers)
            if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                st.subheader("Estadísticas — Mejores y IGUALES (Embed Cosine)")
                s_ia_e = pd.to_numeric(df_out.get("score_cliente_vs_IA_embed"), errors="coerce") if "score_cliente_vs_IA_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                s_usr_e = pd.to_numeric(df_out.get("score_cliente_vs_usuario_embed"), errors="coerce") if "score_cliente_vs_usuario_embed" in df_out.columns else pd.Series([pd.NA]*len(df_out), index=df_out.index)
                valid_e = s_ia_e.notna() & s_usr_e.notna()
                ia_wins_e = valid_e & (s_ia_e > s_usr_e)
                usr_wins_e = valid_e & (s_usr_e > s_ia_e)
                empate_e = valid_e & ((s_ia_e - s_usr_e).abs() <= eps)

                cA, cB, cC = st.columns(3)
                with cA:
                    st.metric("Mejores PROPUESTA USUARIO (Embed)", f"{int(usr_wins_e.sum())}")
                with cB:
                    st.metric("Mejores PROPUESTA IA (Embed)", f"{int(ia_wins_e.sum())}")
                with cC:
                    st.metric("IGUALES (Embed)", f"{int(empate_e.sum())}")
                st.caption("Método de similitud usado: Sentence-Transformers embed_cosine")

            # (Se removió la sección "Métricas" y el control de Umbral a solicitud del usuario)

            # Descarga deshabilitada por solicitud del usuario

        # Descarga deshabilitada por solicitud del usuario
            # Resumen general (todas las métricas)
            try:
                import pandas as _pd  # local alias to avoid shadowing
                resumen_rows = []
                def _compute_counts(col_usr: str, col_ia: str):
                    eps_local = 1e-9
                    s_ia_loc = _pd.to_numeric(df_out.get(col_ia), errors="coerce") if col_ia in df_out.columns else _pd.Series([_pd.NA]*len(df_out), index=df_out.index)
                    s_usr_loc = _pd.to_numeric(df_out.get(col_usr), errors="coerce") if col_usr in df_out.columns else _pd.Series([_pd.NA]*len(df_out), index=df_out.index)
                    valid = s_ia_loc.notna() & s_usr_loc.notna()
                    ia_w = valid & (s_ia_loc > s_usr_loc)
                    usr_w = valid & (s_usr_loc > s_ia_loc)
                    emp = valid & ((s_ia_loc - s_usr_loc).abs() <= eps_local)
                    return int(_pd.to_numeric(usr_w, errors="coerce").fillna(0).sum()), int(_pd.to_numeric(ia_w, errors="coerce").fillna(0).sum()), int(_pd.to_numeric(emp, errors="coerce").fillna(0).sum())

                # Difflib (base)
                if ("score_cliente_vs_usuario" in df_out.columns) or ("score_cliente_vs_IA" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario", "score_cliente_vs_IA")
                    resumen_rows.append({"Método": "Difflib", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # TF-IDF
                if ("score_cliente_vs_usuario_tfidf" in df_out.columns) or ("score_cliente_vs_IA_tfidf" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_tfidf", "score_cliente_vs_IA_tfidf")
                    resumen_rows.append({"Método": "TF-IDF (char 3–5)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # Token Set (RapidFuzz)
                if ("score_cliente_vs_usuario_tok" in df_out.columns) or ("score_cliente_vs_IA_tok" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_tok", "score_cliente_vs_IA_tok")
                    resumen_rows.append({"Método": "Token Set (RapidFuzz)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                # Embed Cosine (Sentence-Transformers)
                if ("score_cliente_vs_usuario_embed" in df_out.columns) or ("score_cliente_vs_IA_embed" in df_out.columns):
                    u,i,e = _compute_counts("score_cliente_vs_usuario_embed", "score_cliente_vs_IA_embed")
                    resumen_rows.append({"Método": "Embed Cosine (ST)", "Mejores PROPUESTA USUARIO": u, "Mejores PROPUESTA IA": i, "IGUALES": e})

                if resumen_rows:
                    _file_label = getattr(archivo, "name", None)
                    if _file_label:
                        st.subheader(f"Resumen general de métodos — Archivo: {_file_label}")
                    else:
                        st.subheader("Resumen general de métodos")
                    df_resumen = _pd.DataFrame(resumen_rows)
                    try:
                        # Resaltar en verde el mayor entre 'Mejores PROPUESTA USUARIO' y 'Mejores PROPUESTA IA' por fila
                        if all(c in df_resumen.columns for c in ["Mejores PROPUESTA USUARIO", "Mejores PROPUESTA IA"]):
                            sty = df_resumen.style.highlight_max(subset=["Mejores PROPUESTA USUARIO", "Mejores PROPUESTA IA"], axis=1, color="#d4edda")
                            st.dataframe(sty, use_container_width=True)
                        else:
                            st.table(df_resumen)
                    except Exception:
                        st.table(df_resumen)
            except Exception:
                pass
    except Exception as e:
        st.error(f"Ocurrió un error leyendo el archivo: {e}")
else:
    st.info("👆 Arrastra o selecciona un archivo para comenzar.")
