import streamlit as st
from global_bloc_finance.visualization_registry import visualization_options

st.set_page_config(page_title="Global BLOC", layout="wide")

st.title("Global BLOC Financial Intelligence")
st.caption("Metric-agnostic visualization workspace")

metric = {
    "id": 1,
    "name": "Simple Return",
    "category": "return",
}

options = visualization_options(metric)
recommended = [x for x in options if x["recommended"]]

st.subheader(f"{metric['name']} — Visualization")
st.selectbox(
    "Choose a visualization",
    options,
    index=next((i for i, x in enumerate(options) if x["recommended"]), 0),
    format_func=lambda x: f"★ {x['name']} — Recommended" if x["recommended"] else x["name"],
    key="visualization_selector",
)

selected = st.session_state.visualization_selector
st.info(selected["reason"] if selected["recommended"] else f"{selected['name']} is available for manual selection.")

st.markdown("### Recommended visualizations")
for item in recommended:
    st.write(f"**★ {item['name']}** — {item['reason']}")

st.markdown("### Visualization library")
st.caption(f"{len(options)} visualizations are available. Recommendations do not restrict selection.")

for item in options:
    label = f"★ {item['name']}" if item["recommended"] else item["name"]
    st.write(f"- {label}")
