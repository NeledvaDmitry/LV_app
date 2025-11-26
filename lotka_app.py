
import json
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
import streamlit as st


st.set_page_config(page_title="Модель Лотки–Вольтерра", layout="wide")

st.title("Модель Лотки–Вольтерра для системы «заяц–рысь»")

st.markdown(
    "Загрузите JSON с параметрами модели или отредактируйте пример ниже. "
    "Обязательные параметры: `alpha, beta, gamma, delta, t_start, t_end`."
)

default_json = """{
  "alpha": 0.4807,
  "beta": 0.0248,
  "gamma": 0.9272,
  "delta": 0.0276,
  "x0": 30.0,
  "y0": 4.0,
  "t_start": 0.0,
  "t_end": 50.0,
  "n_points": 2000,
  "method": "DOP853"
}"""

uploaded = st.file_uploader("JSON-файл с параметрами", type=["json"])
if uploaded is not None:
    json_text = uploaded.read().decode("utf-8")
else:
    json_text = st.text_area("Параметры в формате JSON", default_json, height=220)

config_ok = True
try:
    cfg = json.loads(json_text)
except Exception as e:
    st.error(f"Ошибка разбора JSON: {e}")
    config_ok = False

if config_ok:
    alpha = float(cfg.get("alpha"))
    beta = float(cfg.get("beta"))
    gamma = float(cfg.get("gamma"))
    delta = float(cfg.get("delta"))

    x0 = float(cfg.get("x0", 30.0))
    y0 = float(cfg.get("y0", 4.0))

    t_start = float(cfg.get("t_start"))
    t_end = float(cfg.get("t_end"))
    n_points = int(cfg.get("n_points", 2000))
    method = cfg.get("method", "DOP853")

    t_eval = np.linspace(t_start, t_end, n_points)

    def lotka_volterra(t, z):
        x, y = z
        dxdt = alpha * x - beta * x * y
        dydt = -gamma * y + delta * x * y
        return [dxdt, dydt]

    sol = solve_ivp(
        lotka_volterra,
        (t_start, t_end),
        [x0, y0],
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=1e-12,
    )

    if not sol.success:
        st.error(f"Решатель вернул ошибку: {sol.message}")
    else:
        t = sol.t
        x = sol.y[0]
        y = sol.y[1]

        def H(x, y):
            eps = 1e-12
            x_safe = np.maximum(x, eps)
            y_safe = np.maximum(y, eps)
            return delta * x_safe - gamma * np.log(x_safe) + \
                   beta * y_safe - alpha * np.log(y_safe)

        H_values = H(x, y)
        x_eq = gamma / delta
        y_eq = alpha / beta

        fmt_options = ["png", "svg", "pdf"]

        tab_time, tab_phase, tab_H = st.tabs(
            ["Динамика во времени", "Фазовый портрет", "Инвариант H(t)"]
        )

        plt.style.use("dark_background")

        with tab_time:
            st.subheader("Графики x(t) и y(t)")
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            ax.plot(t, x, label="Жертвы x(t)", linewidth=2, color="#39ff14")
            ax.plot(t, y, label="Хищники y(t)", linewidth=2, color="#00eaff")
            ax.set_xlabel("t")
            ax.set_ylabel("Численность")
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig)

            fmt = st.selectbox("Формат сохранения", fmt_options, key="time_fmt")
            buf = BytesIO()
            fig.savefig(buf, format=fmt, bbox_inches="tight")
            st.download_button(
                label=f"Скачать график ({fmt.upper()})",
                data=buf.getvalue(),
                file_name=f"time_plot.{fmt}",
                mime="image/" + ("png" if fmt == "png" else "svg+xml" if fmt == "svg" else "pdf"),
            )

        with tab_phase:
            st.subheader("Фазовый портрет")
            fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
            ax2.plot(x, y, linewidth=2, color="#ff009d", label="Фазовая траектория")
            ax2.scatter([x_eq], [y_eq], color="white", s=30, label="Стационарная точка")
            ax2.axhline(y=y_eq, linestyle="--", linewidth=1, alpha=0.5)
            ax2.axvline(x=x_eq, linestyle="--", linewidth=1, alpha=0.5)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.legend()
            ax2.grid(alpha=0.2)
            fig2.tight_layout()
            st.pyplot(fig2)

            fmt2 = st.selectbox("Формат сохранения", fmt_options, key="phase_fmt")
            buf2 = BytesIO()
            fig2.savefig(buf2, format=fmt2, bbox_inches="tight")
            st.download_button(
                label=f"Скачать график ({fmt2.upper()})",
                data=buf2.getvalue(),
                file_name=f"phase_plot.{fmt2}",
                mime="image/" + ("png" if fmt2 == "png" else "svg+xml" if fmt2 == "svg" else "pdf"),
            )

        with tab_H:
            st.subheader("Инвариант системы H(t)")
            fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=150)
            ax3.plot(t, H_values, linewidth=2, color="#39ff14")
            ax3.set_xlabel("t")
            ax3.set_ylabel("H(x(t), y(t))")
            ax3.grid(alpha=0.2)
            fig3.tight_layout()
            st.pyplot(fig3)

            fmt3 = st.selectbox("Формат сохранения", fmt_options, key="H_fmt")
            buf3 = BytesIO()
            fig3.savefig(buf3, format=fmt3, bbox_inches="tight")
            st.download_button(
                label=f"Скачать график ({fmt3.upper()})",
                data=buf3.getvalue(),
                file_name=f"invariant_plot.{fmt3}",
                mime="image/" + ("png" if fmt3 == "png" else "svg+xml" if fmt3 == "svg" else "pdf"),
            )