import json
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from model import Parameters, InitialConditions, solve_lotka_volterra, invariant_values, equilibrium_point


st.set_page_config(page_title="Lotka–Volterra App", layout="wide")

plt.style.use("dark_background")

st.title("Модель Лотки–Вольтерра")
model_info_md = r"""
### Что за модель используется

В приложении используется классическая модель Лотки–Вольтерра для описания взаимодействия двух популяций: **жертвы** и **хищника**.  
Это простая математическая модель, которая показывает, как численность одного вида влияет на динамику другого.

В нашей интерпретации:

- \(x(t)\) — численность жертв (например, зайцы);
- \(y(t)\) — численность хищников (например, рыси).

Модель описывает, как \(x(t)\) и \(y(t)\) меняются во времени.

---

### Уравнения модели

$$
\begin{cases}
\dfrac{dx}{dt} = \alpha x - \beta x y, \\\\[6pt]
\dfrac{dy}{dt} = -\gamma y + \delta x y.
\end{cases}
$$

Здесь $\alpha, \beta, \gamma, \delta > 0$ — параметры модели.

---

### Смысл коэффициентов

- **$\alpha$** — скорость роста популяции жертв в отсутствие хищников.
- **$\beta$** — «сила» хищничества.
- **$\gamma$** — скорость убывания хищников.
- **$\delta$** — эффективность преобразования жертв в прирост хищников.

---

### Типичное поведение системы

Даже при фиксированных коэффициентах система ведёт себя нелинейно. Типичная картина:

- численность жертв сначала растёт;
- за ней, с задержкой, растёт численность хищников;
- увеличение хищников приводит к уменьшению числа жертв;
- после падения численности жертв уменьшается и численность хищников;
- затем цикл повторяется.

---

### Что показывают графики в этом приложении

**1. Динамика во времени (x(t), y(t))**  

Показывает, как численность жертв и хищников меняется во времени, позволяет оценить цикличность и сдвиг фаз между популяциями.

---

**2. Фазовый портрет (y в зависимости от x)**  

График в координатах \((x, y)\), на котором видно замкнутую траекторию вокруг точки равновесия и общее взаимное поведение популяций.

---

**3. Инвариант системы (закон сохранения)**  

$$
H(x, y) = \delta x - \gamma \ln x + \beta y - \alpha \ln y.
$$

Если численное решение получено корректно, значение \(H(x(t), y(t))\) почти не меняется во времени. Этот график служит проверкой точности и устойчивости вычислений.
"""

st.markdown(
    "Загрузите JSON с параметрами или отредактируйте пример. "
    "Обязательные поля: `alpha, beta, gamma, delta, t_start, t_end`."
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

try:
    cfg = json.loads(json_text)
except Exception as e:
    st.error(f"Ошибка разбора JSON: {e}")
    st.stop()

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

p = Parameters(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
ic = InitialConditions(
    x0=x0,
    y0=y0,
    t_start=t_start,
    t_end=t_end,
    n_points=n_points,
    method=method,
)

try:
    t, x, y = solve_lotka_volterra(p, ic)
except RuntimeError as e:
    st.error(f"Ошибка решателя: {e}")
    st.stop()

H_values = invariant_values(x, y, p)
x_eq, y_eq = equilibrium_point(p)

neon1 = "#39ff14"
neon2 = "#00eaff"
neon3 = "#ff009d"

fmt_options = ["png", "svg", "pdf"]

tab_info, tab_time, tab_phase, tab_H = st.tabs(
    ["О модели", "Динамика во времени", "Фазовый портрет", "Инвариант H(t)"]
)

with tab_info:
    st.markdown(model_info_md)

with tab_time:
    st.subheader("Графики x(t) и y(t)")

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(t, x, label="Жертвы x(t)", linewidth=2, color=neon1)
    ax.plot(t, y, label="Хищники y(t)", linewidth=2, color=neon2)
    ax.set_xlabel("t")
    ax.set_ylabel("Численность")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    fmt = st.selectbox("Формат сохранения", fmt_options, key="time_fmt")
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    mime = "image/png" if fmt == "png" else "image/svg+xml" if fmt == "svg" else "application/pdf"
    st.download_button(
        label=f"Скачать график ({fmt.upper()})",
        data=buf.getvalue(),
        file_name=f"time_plot.{fmt}",
        mime=mime,
    )

with tab_phase:
    st.subheader("Фазовый портрет")

    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
    ax2.plot(x, y, linewidth=2, color=neon3, label="Фазовая траектория")
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
    mime2 = "image/png" if fmt2 == "png" else "image/svg+xml" if fmt2 == "svg" else "application/pdf"
    st.download_button(
        label=f"Скачать график ({fmt2.upper()})",
        data=buf2.getvalue(),
        file_name=f"phase_plot.{fmt2}",
        mime=mime2,
    )

with tab_H:
    st.subheader("Инвариант системы H(t)")

    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=150)
    ax3.plot(t, H_values, linewidth=2, color=neon1)
    ax3.set_xlabel("t")
    ax3.set_ylabel("H(x(t), y(t))")
    ax3.grid(alpha=0.2)
    fig3.tight_layout()
    st.pyplot(fig3)

    fmt3 = st.selectbox("Формат сохранения", fmt_options, key="H_fmt")
    buf3 = BytesIO()
    fig3.savefig(buf3, format=fmt3, bbox_inches="tight")
    mime3 = "image/png" if fmt3 == "png" else "image/svg+xml" if fmt3 == "svg" else "application/pdf"
    st.download_button(
        label=f"Скачать график ({fmt3.upper()})",
        data=buf3.getvalue(),
        file_name=f"invariant_plot.{fmt3}",
        mime=mime3,
    )