import streamlit as st

from src.utils.load_data import load_vix


def main() -> None:
    st.title("Crisis Forecaster prototype")

    vix = load_vix()
    st.subheader("Latest VIX readings")
    st.dataframe(vix.tail(10))

    st.subheader("VIX time series")
    st.line_chart(vix)


if __name__ == "__main__":
    main()
