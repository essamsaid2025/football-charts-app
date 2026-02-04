import os
import tempfile
import streamlit as st

from charts import load_data, validate_and_clean, build_report_from_df

st.set_page_config(page_title="Football Charts Generator", layout="wide")

st.title("⚽ Football Charts Generator (Upload CSV / Excel)")
st.caption("Required columns: outcome, x, y  | Pass rows need x2, y2 for arrows")

with st.expander("🎛️ Settings", expanded=True):
    title_text = st.text_input("Title", value="Match Report")

    attack_dir_ui = st.selectbox("Attack direction", ["Left → Right", "Right → Left"])
    attack_dir = "ltr" if attack_dir_ui == "Left → Right" else "rtl"

    flip_y = st.checkbox("Flip Y axis (use this if your Y=0 is at the bottom)", value=False)

    pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular (recommended)", "Square (0-100)"])
    pitch_mode = "rect" if pitch_mode_ui.startswith("Rectangular") else "square"

    # For rectangular pitch: choose width (controls how tall it is)
    pitch_width = st.slider("Rect pitch width (0-100 scale mapped to this)", min_value=50.0, max_value=80.0, value=64.0, step=1.0)

    st.markdown("### Pass colors")
    col_pass_success = st.color_picker("Successful", "#00FF6A")
    col_pass_unsuccess = st.color_picker("Unsuccessful", "#FF4D4D")
    col_pass_key = st.color_picker("Key pass", "#00C2FF")
    col_pass_assist = st.color_picker("Assist", "#FFD400")

    st.markdown("### Shot colors")
    col_shot_off = st.color_picker("Off target", "#FF8A00")
    col_shot_on = st.color_picker("On target", "#00C2FF")
    col_shot_goal = st.color_picker("Goal", "#00FF6A")

pass_colors = {
    "successful": col_pass_success,
    "unsuccessful": col_pass_unsuccess,
    "key pass": col_pass_key,
    "assist": col_pass_assist,
}
shot_colors = {
    "off target": col_shot_off,
    "ontarget": col_shot_on,
    "goal": col_shot_goal,
}

mode = st.radio("Choose output type", ["Match Charts", "Pizza Chart"])

uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

if uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            df = load_data(path)
            df = validate_and_clean(df)
        except Exception as e:
            st.error(str(e))
            st.stop() 
                    except Exception as e:
            st.error(str(e))
            st.stop()
        # ----------------------------
        # PIZZA MODE
        # ----------------------------
        if mode == "Pizza Chart":
            # df currently contains the uploaded file contents (loaded above)
            st.success(f"Loaded ✅ rows: {len(df)}")
            st.dataframe(df.head(25), use_container_width=True)

            pizza_title = st.text_input("Pizza title", value="Player – League")
            pizza_subtitle = st.text_input("Pizza subtitle", value="Percentile Rank vs Peers")

            if st.button("Generate Pizza"):
                from charts import pizza_chart  # we will add this function in charts.py

                fig = pizza_chart(df, title=pizza_title, subtitle=pizza_subtitle)

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)

                png_path = os.path.join(out_dir, "pizza.png")
                pdf_path = os.path.join(out_dir, "pizza.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight")

                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")

                st.pyplot(fig)

                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download pizza.pdf", f, file_name="pizza.pdf")

                with open(png_path, "rb") as f:
                    st.download_button("⬇️ Download pizza.png", f, file_name="pizza.png")

            st.stop()


        st.success(f"Loaded ✅ rows: {len(df)}")

        with st.expander("Preview data (first 25 rows)"):
            st.dataframe(df.head(25), use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.write("Detected event types:")
            st.write(df["event_type"].value_counts())
        with cols[1]:
            st.write("Detected outcomes (top 12):")
            st.write(df["outcome"].value_counts().head(12))

        if st.button("Generate Report"):
            out_dir = os.path.join(tmp, "output")
            pdf_path, png_paths = build_report_from_df(
                df,
                out_dir=out_dir,
                title=title_text,
                attack_direction=attack_dir,
                flip_y=flip_y,
                pitch_mode=pitch_mode,
                pitch_width=pitch_width,
                pass_colors=pass_colors,
                shot_colors=shot_colors,
            )

            st.subheader("Preview")
            for p in png_paths:
                st.image(p, use_container_width=True)

            st.subheader("Downloads")
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download report.pdf", f, file_name="report.pdf")

            for p in png_paths:
                name = os.path.basename(p)
                with open(p, "rb") as f:
                    st.download_button(f"⬇️ Download {name}", f, file_name=name)
