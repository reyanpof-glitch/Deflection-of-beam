# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Beam Deflection", layout="centered")

st.title("Beam Deflection Calculator")

def beam_deflection(beam, load, L, E, I, mag, a):
    x = np.linspace(0, L, 300)
    y = np.zeros_like(x)

    if beam == "Cantilever":
        if load == "Point":
            a = min(max(a, 0.0), L)
            for i, xi in enumerate(x):
                if xi <= a:
                    y[i] = mag * xi**2 * (3*a - xi) / (6*E*I)
                else:
                    y[i] = mag * a**2 * (3*xi - a) / (6*E*I)
        else:  # UDL
            y = mag * x**2 * (6*L**2 - 4*L*x + x**2) / (24*E*I)

    else:  # Simply supported
        if load == "Point":
            a = min(max(a, 0.0), L)
            b = L - a
            for i, xi in enumerate(x):
                if xi <= a:
                    y[i] = mag * b * xi * (L**2 - b**2 - xi**2) / (6*L*E*I)
                else:
                    y[i] = mag * a * (L - xi) * (L**2 - a**2 - (L - xi)**2) / (6*L*E*I)
        else:  # UDL
            y = mag * x * (L**3 - 2*L*x**2 + x**3) / (24*E*I)

    y = -y  # downward deflection
    return x, y, y.min()

beam = st.radio("Beam Type", ["Cantilever", "Simply-supported"])
load = st.radio("Load Type", ["Point", "UDL"])

L = st.number_input("Length L", value=1.0)
E = st.number_input("Young's Modulus E", value=2.1e11)
I = st.number_input("Moment of Inertia I", value=1e-6)
mag = st.number_input("Load (P or w)", value=100.0)

a = 0.0
if load == "Point":
    a = st.number_input("Load Location a", value=0.5)

if st.button("Compute"):
    x, y, max_def = beam_deflection(beam, load, L, E, I, mag, a)

    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Deflection")
    ax.set_title("Beam Deflection Curve")
    ax.grid(True)

    st.pyplot(fig)
    st.success(f"Maximum Deflection = {max_def:.6g}")