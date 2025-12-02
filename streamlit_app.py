# python
import os
import tempfile
import io
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from inference import process_file

st.set_page_config(page_title="AI prediction for Class III")

# Header: show logo at left and title + caption at right
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
with col_title:
    st.title("AI prediction for Class III")
    st.caption("Upload an xlsx/xls file following the template linked below. The app will process it with a pretrained Graph Neural Network and show the prediction plot. Full code openly available on [GitHub](https://github.com/Buffoni/Class-III-GNN)")

# Replace with your actual template URL
TEMPLATE_URL = "https://raw.githubusercontent.com/Buffoni/Class-III-GNN/main/test_data_sample.xlsx"
st.markdown(f"Template file: [Download xlsx template]({TEMPLATE_URL})")

uploaded = st.file_uploader("Upload input file", type=["xls", "xlsx"])
years = st.number_input("Enter the prediction time (yrs):", value=1.0, format="%.2f", step=0.1)

def save_uploaded_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

if st.button("Predict"):
    selected_path = None
    cleanup_temp = False

    if uploaded:
        selected_path = save_uploaded_to_temp(uploaded)
        cleanup_temp = True
    else:
        st.error("Please upload a file (see template link above).")
        st.stop()

    try:
        result = process_file(selected_path, float(years))

        if (not isinstance(result, (list, tuple))) or len(result) < 2:
            raise ValueError("process_file returned unexpected format. Expected (start_coords, prediction).")

        start = result[0]
        pred = result[1]

        if not (hasattr(start, "shape") and hasattr(pred, "shape") and start.shape[1] >= 2 and pred.shape[1] >= 2):
            raise ValueError("Start/prediction arrays do not have expected shape (N, >=2).")

        # Original larger plotting
        fig, ax = plt.subplots(figsize=(7,6))
        ax.scatter(pred[:,0], pred[:,1], c='tab:orange', s=40, alpha=0.85, label='Model Prediction', edgecolors='w', linewidths=0.5)
        ax.scatter(start[:,0], start[:,1], c='tab:blue', s=30, alpha=0.8, label='Starting Coordinates', marker='o', edgecolors='k', linewidths=0.4)
        ax.set_title(f'Prediction after {years:.2f} years')
        ax.legend(frameon=True)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

        all_x = list(start[:,0]) + list(pred[:,0])
        all_y = list(start[:,1]) + list(pred[:,1])
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        xpad = max(0.5, 0.05 * (xmax - xmin if xmax - xmin != 0 else 1.0))
        ypad = max(0.5, 0.05 * (ymax - ymin if ymax - ymin != 0 else 1.0))
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        fig.tight_layout()
        st.pyplot(fig)

        # Create XLSX in simple format (start, prediction, metadata) using ExcelWriter
        out_buf = io.BytesIO()
        with pd.ExcelWriter(out_buf, engine='openpyxl') as writer:
            df_start = pd.DataFrame(start[:, :2], columns=['X', 'Y'])
            df_pred = pd.DataFrame(pred[:, :2], columns=['X', 'Y'])
            df_start.to_excel(writer, sheet_name='start_coordinates', index=False)
            df_pred.to_excel(writer, sheet_name='prediction', index=False)
            meta = {
                'generated_at': [datetime.utcnow().isoformat() + 'Z'],
                'prediction_years': [float(years)],
                'source_file': [uploaded.name if uploaded else ""]
            }
            pd.DataFrame(meta).to_excel(writer, sheet_name='metadata', index=False)
        out_buf.seek(0)
        st.download_button("Download XLSX output (same format as template)", data=out_buf, file_name="prediction_output.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.success("XLSX file ready for download.")

        # Provide model output in original format if inference saved it, otherwise .npz
        saved_path = None
        if len(result) >= 3 and isinstance(result[2], str) and os.path.exists(result[2]):
            saved_path = result[2]

        if saved_path:
            with open(saved_path, "rb") as f:
                data = f.read()
            basename = os.path.basename(saved_path)
            st.download_button("Download model output", data=data, file_name=basename, mime="application/octet-stream")
        else:
            npz_buf = io.BytesIO()
            try:
                np.savez_compressed(npz_buf,
                                    start=start,
                                    prediction=pred,
                                    metadata=np.array([{"generated_at": datetime.utcnow().isoformat() + 'Z',
                                                        "prediction_years": float(years),
                                                        "source_file": uploaded.name if uploaded else ""}]))
                npz_buf.seek(0)
                st.download_button("Download raw model output (.npz)", data=npz_buf, file_name="model_output.npz", mime="application/octet-stream")
            except Exception as ser_exc:
                st.warning("Could not serialize model output to .npz buffer.")
                st.exception(ser_exc)

    except Exception as e:
        st.error("An error occurred during processing. See details for debugging.")
        st.exception(e)

    finally:
        if cleanup_temp and selected_path and os.path.exists(selected_path):
            try:
                os.remove(selected_path)
            except Exception:
                pass
