import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_qudit_state(psi, mode = "coeffs"):
    """
    Plot the coefficients of a qudit state or the probabilities

        |psi> = sum_k c_k |k>

    using:
        - bar height = |c_k|
        - bar color  = arg(c_k) in [0, 2*pi)

    Parameters
    ----------
    psi : np.array
        Array of complex coefficients.
        Ket representing the qudit state.
    mode : str
        If "coeffs", plots the coefficients.
        If "probs", plots the probabilities.
    savename : str
        Name of the file to save the figure.
        Figure is saved only of the name is provided.

    Returns
    -------
    None
    """
    assert mode in ["coeffs", "probs"], "mode must be 'coeffs' or 'probs'"

    if mode == "probs":
        psi = np.abs(psi) ** 2

    N = len(psi)
    k = np.arange(N)

    # Magnitudes
    magnitudes = np.abs(psi)

    # Phases mapped to [0, 2*pi)
    phases = np.mod(np.angle(psi), 2 * np.pi)

    # Cyclic colormap appropriate for phase
    cmap = plt.colormaps["twilight"]
    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=2 * np.pi
    )

    colors = cmap(norm(phases))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4.5))

    ax.bar(
        k,
        magnitudes,
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # Axes labels
    ax.set_xlabel(r"$k$")
    y_label = r"$|c_k|$" if mode == "coeffs" else r"$p_k$"
    ax.set_ylabel(y_label)

    ax.set_xlim(-0.7, N - 0.3)

    # Avoid problems for zero vector / extremely small states
    if np.max(magnitudes) > 0:
        ax.set_ylim(0, 1.05 * np.max(magnitudes))

    # Show every basis-state index for qudits up to dimension 32
    if N <= 32:
        ax.set_xticks(k)


    if mode == "coeffs":
        # Phase colorbar
        sm = mpl.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            pad=0.02
        )



        cbar.set_label(r"$\arg(c_k)$")

        cbar.set_ticks([
            0,
            np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            2 * np.pi
        ])

        cbar.set_ticklabels([
            r"$0$",
            r"$\pi/2$",
            r"$\pi$",
            r"$3\pi/2$",
            r"$2\pi$"
        ])

    plt.tight_layout()
    plt.show()

