import numpy as np, matplotlib.pyplot as plt, gradio as gr

def beam_deflection(beam, load, L, E, I, mag, a):
    L, E, I, mag = map(float, (L, E, I, mag))
    a = float(a) if load == "Point" else None
    x = np.linspace(0, L, 300)
    y = np.zeros_like(x)

    if beam == "Cantilever":
        if load == "Point":
            a = min(max(a, 0.0), L)
            # piecewise for cantilever with point load at a
            for i, xi in enumerate(x):
                if xi <= a:
                    y[i] = mag * xi**2 * (3*a - xi) / (6*E*I)
                else:
                    y[i] = mag * a**2 * (3*xi - a) / (6*E*I)
        else:  # UDL over full length, intensity = mag (force/length)
            y = mag * x**2 * (6*L**2 - 4*L*x + x**2) / (24*E*I)

    else:  # Simply-supported
        if load == "Point":
            a = min(max(a, 0.0), L)
            b = L - a
            for i, xi in enumerate(x):
                if xi <= a:
                    y[i] = mag * b * xi * (L**2 - b**2 - xi**2) / (6*L*E*I)
                else:
                    y[i] = mag * a * (L - xi) * (L**2 - a**2 - (L - xi)**2) / (6*L*E*I)
        else:  # UDL over full length (w)
            y = mag * x * (L**3 - 2*L*x**2 + x**3) / (24*E*I)

    # convention: negative downwards for plotting clarity
    y = -y
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=100)
    ax.plot(x, y, lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x (same units as L)")
    ax.set_ylabel("Deflection")
    ax.set_title(f"{beam} - {'Point load at a='+str(a) if load=='Point' else 'UDL'}")
    ax.grid(True, lw=0.5)
    plt.tight_layout()

    maxdef = y.min()  # since negative down
    txt = f"Max deflection = {maxdef:.6g} (same units as load*L^3/(E*I))"
    return fig, txt

css = ".gradio-container {max-width:420px;margin:0 auto;}"

with gr.Blocks(css=css, analytics_enabled=False) as app:
    gr.Markdown("## Beam deflection (compact mobile)")
    beam = gr.Radio(["Cantilever", "Simply-supported"], value="Cantilever", label="Beam type")
    load = gr.Radio(["Point", "UDL"], value="Point", label="Load type")
    with gr.Column():
        L = gr.Number(1.0, label="Length L")
        E = gr.Number(2.1e11, label="Young's Modulus E")
        I = gr.Number(1e-6, label="Moment of Inertia I")
        mag = gr.Number(100.0, label="Load (P or w)")
        a = gr.Number(0.5, label="Load location a (for point load)")
    out_plot = gr.Plot()
    out_txt = gr.Textbox(lines=1)
    btn = gr.Button("Compute")
    btn.click(beam_deflection, inputs=[beam, load, L, E, I, mag, a], outputs=[out_plot, out_txt])

if __name__ == "__main__":
    app.launch()