"""
Figures, animations, and black-hole disk rendering for the relativity project.

This module depends on ``calculation_functions`` for v/c conversion and time dilation.
Animations return ``IPython.display.HTML`` (JS-backed) for use in Jupyter notebooks.
Static helpers call ``matplotlib.pyplot.show()`` or mutate/pass back ``pandas`` frames.
"""

from io import StringIO

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from calculation_functions import kmh_to_c_fraction, time_dilation


def light_clock_stationary():
    """
    Build an HTML animation of a stationary light clock (vertical photon bounce).

    Purpose
    -------
    Visual teaching aid: mirrors at y=0 and y=d, photon oscillates vertically in the
    clock’s rest frame. Uses ``FuncAnimation`` and embeds frames as JavaScript HTML.

    Parameters
    ----------
    None
        Mirror separation, frame count, and timing are fixed inside the function
        (d=5, 120 frames, 40 ms interval).

    Steps
    -----
    1. Create a matplotlib figure and axes with fixed x/y limits and labels.
    2. Draw fixed mirror markers and a red point artist for the photon.
    3. Define ``animate(frame)``: map frame index to bounce phase (up then down).
    4. Instantiate ``FuncAnimation``, close the figure to suppress duplicate static views.
    5. Return ``HTML(ani.to_jshtml())`` for inline notebook playback.

    Returns
    -------
    IPython.display.HTML
        Embeddable animation object.
    """
    # Step 1: Geometry and animation length (model units; d = mirror separation).
    d = 5
    frames = 120

    # Step 2: Figure and axes with fixed view and axis labels.
    fig, ax = plt.subplots(figsize=(4, 6))

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, d + 1)

    ax.set_title("Light Clock (Stationary)")
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")

    # Step 3: Static mirrors and text labels.
    ax.scatter([0, 0], [0, d], color="black")
    ax.text(0.1, d, "Top mirror")
    ax.text(0.1, 0, "Bottom mirror")

    # Step 4: Empty line artist for the photon (updated each frame).
    light, = ax.plot([], [], "ro", markersize=10)

    def animate(frame):
        """Update photon height for one animation frame (one half-cycle up, one down)."""
        # Sub-step 1: Phase within one full up-down cycle (two half-periods).
        period = 60
        phase = frame % (2 * period)

        # Sub-step 2: Vertical position y: ascend then descend between mirrors.
        if phase < period:
            y = (phase / period) * d
        else:
            y = d - ((phase - period) / period) * d

        # Sub-step 3: Move the photon marker on the axis.
        light.set_data([0], [y])

        # Return: tuple of artists for blitting.
        return (light,)

    # Step 5: Build animation and close figure to avoid a duplicate static window.
    ani = FuncAnimation(fig, animate, frames=frames, interval=40)
    plt.close(fig)

    # Return: HTML embedding (JS) for Jupyter.
    return HTML(ani.to_jshtml())


def light_clock_moving():
    """
    Build an HTML animation of a moving light clock as seen by a lab observer.

    Purpose
    -------
    The clock moves horizontally with speed v while the photon still bounces in the
    clock frame; the lab sees a zig-zag path. Traces show clock position vs photon path.

    Parameters
    ----------
    None
        Uses fixed d=5, v=0.3 (units consistent with axes), 200 frames.

    Steps
    -----
    1. Set up axes, mirror artists, photon, dashed “instant” hypotenuse, and trace lines.
    2. For each frame: advance clock position x = v * t, compute vertical bounce phase.
    3. Update mirror and photon positions; append to trace lists; refresh line artists.
    4. Build ``FuncAnimation``, close figure, return JS HTML.

    Returns
    -------
    IPython.display.HTML
        Embeddable animation for the moving-clock scenario.
    """
    # Step 1: Mirror spacing, lab-frame clock speed, and frame count.
    d = 5
    v = 0.3
    frames = 200

    # Step 2: Axes sized to show horizontal drift of the clock.
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, d + 1)

    ax.set_title("Moving Light Clock (external observer)")
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")

    # Step 3: Artists for mirrors, photon, dashed “instant” path, and history traces.
    top_mirror, = ax.plot([], [], "ks", markersize=8)
    bottom_mirror, = ax.plot([], [], "ks", markersize=8)
    light, = ax.plot([], [], "ro", markersize=8)
    path, = ax.plot([], [], "r--", alpha=0.6)

    clock_trace_x = []
    clock_trace_y = []
    light_trace_x = []
    light_trace_y = []

    clock_trace_line, = ax.plot([], [], "k-", alpha=0.3)
    light_trace_line, = ax.plot([], [], "r-", alpha=0.3)

    def animate(frame):
        """One timestep: translate mirrors, update photon and diagnostic segments."""
        # Sub-step 1: Lab time and horizontal position of the clock.
        t = frame * 0.05
        x_clock = v * t

        # Sub-step 2: Internal bounce phase (same idea as stationary clock).
        period = 40
        phase = frame % (2 * period)

        if phase < period:
            y = (phase / period) * d
        else:
            y = d - ((phase - period) / period) * d

        # Sub-step 3: Photon sits at the moving mirrors’ x, bounce height y.
        light_x = x_clock
        light_y = y

        # Sub-step 4: Update mirror and photon positions and the dashed leg.
        top_mirror.set_data([x_clock], [d])
        bottom_mirror.set_data([x_clock], [0])
        light.set_data([light_x], [light_y])
        path.set_data([x_clock, light_x], [0, light_y])

        # Sub-step 5: Append to path histories and refresh polyline artists.
        clock_trace_x.append(x_clock)
        clock_trace_y.append(0)
        light_trace_x.append(light_x)
        light_trace_y.append(light_y)

        clock_trace_line.set_data(clock_trace_x, clock_trace_y)
        light_trace_line.set_data(light_trace_x, light_trace_y)

        # Return: all animated artists.
        return (
            top_mirror,
            bottom_mirror,
            light,
            path,
            clock_trace_line,
            light_trace_line,
        )

    # Step 4: Encode animation and suppress extra GUI figure.
    ani = FuncAnimation(fig, animate, frames=frames, interval=40)
    plt.close(fig)

    # Return: notebook-ready HTML.
    return HTML(ani.to_jshtml())


def geometry_of_light_clock():
    """
    Plot the right-triangle geometry used in the moving light-clock derivation.

    Purpose
    -------
    Illustrates (ct)² = d² + (vt)² in normalised units (c=1) for chosen v, d, t.

    Parameters
    ----------
    None
        Uses fixed d=5, v=0.6, t=4, c=1 for a single static diagram.

    Steps
    -----
    1. Compute horizontal leg v*t, vertical leg d, and hypotenuse c*t.
    2. Draw the triangle with matplotlib, label sides, equal aspect, grid.

    Returns
    -------
    None
        ``plt.show()`` displays the figure; return value of ``show()`` is passed through.
    """
    # Step 1: Fixed demonstration values (c = 1 in these units).
    d = 5
    v = 0.6
    t = 4
    c = 1

    # Step 2: Triangle side lengths: horizontal v·t, vertical d, hypotenuse c·t.
    horizontal = v * t
    vertical = d
    hypotenuse = c * t

    # Step 3: New figure for the schematic.
    plt.figure(figsize=(6, 6))

    # Step 4: Closed polygon vertices for the right triangle.
    x = [0, horizontal, 0, 0]
    y = [0, 0, vertical, 0]

    plt.plot(x, y)

    # Step 5: Annotate each side.
    plt.text(horizontal / 2, -0.3, "v*t", ha="center")
    plt.text(-0.3, vertical / 2, "d", va="center")
    plt.text(horizontal / 2, vertical / 2, "c*t")

    plt.scatter([0, horizontal, 0], [0, 0, vertical])

    plt.title("Geometry of the Light Clock")
    plt.xlabel("horizontal distance")
    plt.ylabel("vertical distance")

    plt.axis("equal")
    plt.grid(True)

    # Return: trigger display; value is whatever matplotlib’s show() returns.
    return plt.show()


def stationary_vs_moving_clock():
    """
    Plot coordinate time vs v/c for a stationary clock vs a dilated moving clock.

    Purpose
    -------
    For fixed proper time t₀=1 s, show t = γ(v) t₀ across v/c ∈ [0, 0.99) and compare
    to the constant stationary case.

    Parameters
    ----------
    None
        Uses t0=1 s and 500 samples along v/c.

    Steps
    -----
    1. Sample v/c with ``np.linspace``.
    2. Compute γ and t_moving = γ * t0; build t_stationary constant array.
    3. Plot both curves, add reference lines at v/c = 0.1 and 0.9, labels, legend.
    4. Call ``plt.show()``.

    Returns
    -------
    None
    """
    # Step 1: Fixed proper-time reference (1 second in the moving clock).
    t0 = 1

    # Step 2: Sample v/c from 0 up to 0.99 (stay subluminal).
    v_over_c = np.linspace(0, 0.99, 500)

    # Step 3: Lorentz factor and dilated coordinate time t = γ t₀ for each v/c.
    gamma = 1 / np.sqrt(1 - v_over_c**2)

    t_moving = gamma * t0

    # Step 4: Reference curve: no dilation (always t₀).
    t_stationary = np.full_like(v_over_c, t0)

    # Step 5: Plot both curves and add vertical guides at v/c = 0.1 and 0.9.
    plt.figure(figsize=(8, 5))
    plt.plot(
        v_over_c,
        t_stationary,
        color="gray",
        linestyle="--",
        linewidth=2,
        label="Stationary clock",
    )
    plt.plot(
        v_over_c,
        t_moving,
        color="blue",
        linewidth=2,
        label="Moving clock (time dilation)",
    )

    plt.axvline(x=0.1, color="green", linestyle="--", alpha=0.5)
    plt.axvline(x=0.9, color="red", linestyle="--", alpha=0.5)
    plt.text(0.11, t0 + 0.02, "v=0.1c", color="green")
    plt.text(0.91, t_moving[-1] - 0.05, "v=0.9c", color="red")

    plt.title("Time Dilation: Stationary vs Moving Clock", fontsize=14)
    plt.xlabel("v/c (fraction of speed of light)", fontsize=12)
    plt.ylabel("Time observed (s)", fontsize=12)
    plt.grid(True)
    plt.legend()

    # Step 6: Display interactively (no explicit return).
    plt.show()


def planet_orbits(url):
    """
    Load planet orbital data from a URL, enrich with velocities, and plot elliptic orbits.

    Purpose
    -------
    Fetches a CSV (e.g. semi-major axis, eccentricity, diameter), merges hand-picked
    orbital and rotation data, computes a scalar “total speed” estimate, and draws
    each planet’s orbit in the ecliptic plane (AU).

    Parameters
    ----------
    url : str
        HTTP(S) URL returning CSV text compatible with ``pandas.read_csv`` (must include
        columns such as ``name``, ``semi_majoraxis``, ``eccentricity``, ``diameter`` after
        normalisation).

    Steps
    -----
    1. GET the URL; parse CSV into a DataFrame.
    2. Normalise column names (lowercase, spaces/dashes to underscores).
    3. Map each planet to orbital speed (km/s) and rotation period (hours); join to df.
    4. Derive radius from diameter; surface speed from 2πR / |rotation period|.
    5. Convert orbital km/s to km/h; set total_speed_kmh = orbital + surface (scalar sum).
    6. For each row, build ellipse from semi-major axis and eccentricity; plot with legend.
    7. Style axes (equal aspect, grid, Sun at origin) and ``show()`` the figure.

    Returns
    -------
    pandas.DataFrame
        The enriched table (orbital velocities, rotation, speeds, etc.) for downstream use.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails (``raise_for_status``).
    """
    # Step 1: Download CSV text from the URL.
    resp = requests.get(url)
    resp.raise_for_status()

    # Step 2: Parse into a DataFrame.
    df = pd.read_csv(StringIO(resp.text))

    # Step 3: Normalise column names for robust downstream access.
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]

    # Step 4: Tabulated orbital speeds (km/s) around the Sun.
    velocity_data = {
        "Mercury": 47.4,
        "Venus": 35.0,
        "Earth": 29.8,
        "Mars": 24.1,
        "Jupiter": 13.1,
        "Saturn": 9.7,
        "Uranus": 6.8,
        "Neptune": 5.4,
    }

    # Rotation period (hours): sign = prograde (+) vs retrograde (−), e.g. Venus, Uranus.
    rotation_hours_data = {
        "Mercury": 1407.6,
        "Venus": -5832.5,
        "Earth": 23.934,
        "Mars": 24.623,
        "Jupiter": 9.925,
        "Saturn": 10.656,
        "Uranus": -17.24,
        "Neptune": 16.11,
    }

    # Step 5: Join orbital and rotation metadata by capitalised planet name.
    df["orbital_velocity_kms"] = df["name"].str.capitalize().map(velocity_data)
    df["rotation_hours"] = df["name"].str.capitalize().map(rotation_hours_data)

    # Step 6: Physical radius and equatorial rotation speed (km/h).
    df["radius_km"] = df["diameter"] / 2
    df["surface_speed_kmh"] = (2 * np.pi * df["radius_km"]) / np.abs(df["rotation_hours"])

    # Step 7: Orbital speed in km/h and a scalar “total” speed (orbit + spin).
    df["orbital_velocity_kmh"] = df["orbital_velocity_kms"] * 3600
    df["total_speed_kmh"] = df["orbital_velocity_kmh"] + df["surface_speed_kmh"]

    # Step 8: Plot each orbit as an ellipse in the ecliptic plane (AU).
    plt.figure(figsize=(10, 10))

    for _, row in df.iterrows():
        a = row["semi_majoraxis"]
        e = row["eccentricity"]

        b = a * np.sqrt(1 - e**2)

        theta = np.linspace(0, 2 * np.pi, 360)
        x = a * np.cos(theta)
        y = b * np.sin(theta)

        label_text = (
            f"{row['name']}: Orb {row['orbital_velocity_kmh']:.0f} km/h | "
            f"Surf {row['surface_speed_kmh']:.1f} km/h | "
            f"Total {row['total_speed_kmh']:.0f} km/h"
        )

        plt.plot(x, y, label=label_text, lw=1.5)

    # Step 9: Sun at origin (approximate focus).
    plt.scatter(0, 0, color="yellow", s=250, label="Sun", edgecolors="orange")

    # Step 10: Labels, legend, equal aspect, styling.
    plt.title("Planetary Orbits: Orbital + Surface Rotation Speeds", fontsize=14)
    plt.xlabel("Distance (AU)", fontsize=12)
    plt.ylabel("Distance (AU)", fontsize=12)
    plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1), fontsize=9)

    plt.gca().set_aspect("equal")
    plt.gca().set_facecolor("#f0f0f0")
    plt.grid(color="white", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Return: enriched DataFrame for further analysis (e.g. time dilation plot).
    return df


def planets_time_dilation_linear_plot(df):
    """
    Plot per-planet special-relativistic time-difference (vs one Earth year) in seconds.

    Purpose
    -------
    Uses ``total_speed_kmh`` from ``planet_orbits`` output, converts to v/c, applies
    ``time_dilation`` with t₀ = one Julian year in hours, then shows Δt in seconds.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``name`` and ``total_speed_kmh``. Mutated in-place with new columns.

    Steps
    -----
    1. Compute v/c column via ``kmh_to_c_fraction``.
    2. Set t0_hours = 365.25 * 24; call ``time_dilation`` on the v/c column.
    3. Store γ, observed time, and Δt in the frame; add Δt in seconds.
    4. Line plot planet names vs ``time_difference_seconds``; label and show.

    Returns
    -------
    None
        The figure is shown; ``df`` is updated in place for caller inspection.
    """
    # Step 1: Planet surface “total” speed as v/c.
    df["v_over_c"] = kmh_to_c_fraction(df["total_speed_kmh"])

    # Step 2: One Julian year in hours as proper-time baseline on the planet.
    t0_hours = 365.25 * 24

    # Step 3: γ, dilated year length, and Δt vs that baseline (hours).
    gamma, t_hours, delta_t_hours = time_dilation(df["v_over_c"], t0=t0_hours)

    df["gamma"] = gamma
    df["observed_time_hours"] = t_hours
    df["time_difference_hours"] = delta_t_hours

    # Step 4: Same Δt in seconds for the y-axis.
    df["time_difference_seconds"] = df["time_difference_hours"] * 3600

    # Step 5: Line plot of Δt (seconds per year) vs planet name.
    plt.figure(figsize=(10, 6))
    plt.plot(
        df["name"],
        df["time_difference_seconds"],
        marker="o",
        color="blue",
        linewidth=2,
    )
    plt.ylabel("Time difference vs Earth (seconds per year)", fontsize=12)
    plt.xlabel("Planet", fontsize=12)
    plt.title("Time Dilation on Planetary Surfaces (Linear Plot)", fontsize=14)
    plt.ylim(0)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# ---------------------------------------------------------------------------
# Black hole disk (Schwarzschild lensing-style visualisation helpers)
# ---------------------------------------------------------------------------


def minrs(b):
    """
    Closest-approach radius (in geometric units) for a photon with impact parameter b.

    Purpose
    -------
    For the Schwarzschild black hole, null geodesics yield a cubic condition; this
    function returns a real turning-point radius r_t for integration setup, or 0 if
    the ray is captured / numerically ill-defined.

    Parameters
    ----------
    b : float
        Impact parameter (dimensionless, model-specific normalisation as in the notebook).

    Steps
    -----
    1. Evaluate a Cardano-type closed form under relaxed floating-point error state.
    2. If the value is NaN, return 0.
    3. If the imaginary part is negligible, return the real part; else return 0.

    Returns
    -------
    float
        Turning radius, or 0 when not applicable.
    """
    # Step 1: Evaluate Cardano-type closed form; ignore FP warnings in degenerate cases.
    with np.errstate(all="ignore"):
        val = np.cbrt(b**2) * (
            (-1 + np.sqrt(1 - b**2 / 27 + 0j)) ** (1 / 3)
            + (-1 - np.sqrt(1 - b**2 / 27 + 0j)) ** (1 / 3)
        )

    # Return: undefined turning radius → signal “no usable r_t”.
    if np.isnan(val):
        return 0

    # Return: essentially real root → use its real part as r_t.
    if np.abs(np.imag(val)) < 1e-8:
        return np.real(val)

    # Return: strongly complex / non-physical → 0.
    return 0


def d_phi_dx(x, b, rt):
    """
    Integrand for the bending angle φ along a transformed radial coordinate x.

    Purpose
    -------
    Expresses dφ/dx so numerical quadrature avoids the turning-point singularity at r = r_t.

    Parameters
    ----------
    x : float
        Transformed radial coordinate (shifted from r).
    b : float
        Impact parameter of the ray.
    rt : float
        Turning-point radius from ``minrs(b)``.

    Steps
    -----
    1. Form denominator (x² + r_t) and the radicand for the Schwarzschild null condition.
    2. If the radicand is non-positive, return 0 (no contribution).
    3. Otherwise return the analytic expression for the integrand.

    Returns
    -------
    float
        Local contribution to dφ/dx.
    """
    # Step 1: Denominator (x² + r_t) and radicand for the null geodesic in x-coordinates.
    denom = x**2 + rt
    inside = (x**2 + rt) ** 2 - (b**2 * (-2 + x**2 + rt)) / denom

    # Return: non-positive radicand → no real contribution here.
    if inside <= 0:
        return 0

    # Return: Schwarzschild integrand value at this x.
    return -2 * b * np.abs(x) / (denom * np.sqrt(inside))


def psi_func(phi_val, theta):
    """
    Map deflection angle φ to polar angle ψ on the disk plane for a fixed inclination θ.

    Purpose
    -------
    Geometric helper for projecting bent rays onto the observer’s view of the disk.

    Parameters
    ----------
    phi_val : float
        Bending / azimuth-related angle along the ray (model convention).
    theta : float
        Observer inclination in radians (disk normal vs line of sight).

    Steps
    -----
    1. Evaluate arccos(sin θ sin φ).

    Returns
    -------
    float
        Polar angle ψ in radians.
    """
    # Step 1: Compute ψ = arccos(sin θ sin φ) from observer inclination θ and angle φ.
    # Return: polar angle ψ on the disk plane (radians).
    return np.arccos(np.sin(theta) * np.sin(phi_val))


def varphi_func(phi_val, theta):
    """
    Map deflection angle φ to azimuthal angle φ̃ on the disk plane.

    Purpose
    -------
    Companion to ``psi_func`` for building (α, β) sky-plane coordinates of disk features.

    Parameters
    ----------
    phi_val : float
        Same meaning as in ``psi_func``.
    theta : float
        Observer inclination in radians.

    Steps
    -----
    1. Use ``arctan2`` on cos φ and cos θ sin φ for a stable quadrant-aware angle.

    Returns
    -------
    float
        Azimuthal angle in radians.
    """
    # Step 1: Quadrant-safe azimuth via arctan2(cos φ, cos θ sin φ).
    # Return: azimuthal angle on the disk plane (radians).
    return np.arctan2(np.cos(phi_val), np.cos(theta) * np.sin(phi_val))


def plot_blackhole(ax, phiFunction, theta_deg=80):
    """
    Draw a stylised accretion-disk image on ``ax`` for a given observer inclination.

    Purpose
    -------
    Loops over disk radii and ray orders (direct, secondary, photon ring), matches
    geodesic deflections supplied by ``phiFunction`` to impact parameters, and plots
    curves in the (α, β) plane. Adds a circular black-hole shadow.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (typically cleared inside this function).
    phiFunction : callable
        Function ``phi(xi, b)`` returning integrated deflection; often wraps numerical
        quadrature (e.g. ``scipy.integrate.quad``) in the notebook or another module.
    theta_deg : float, optional
        Inclination of the disk normal to the line of sight, in degrees (default 80).

    Steps
    -----
    1. Clear axes; convert ``theta_deg`` to radians.
    2. For each image order n and sampled disk radius r_s, compute maximum impact b_max.
    3. Sample φ values; build target ψ arrays via ``psi_func`` / ``varphi_func``.
    4. Scan impact parameters b; for each b get r_t = ``minrs(b)``; if r_s > r_t, evaluate
       ``phiFunction`` at ±√(r_s − r_t) to pair (ψ, b).
    5. For each target ψ, pick the closest matching b from combined samples; convert to
       (α, β) and plot.
    6. Add shadow circle of radius √27, set limits, labels, and dark background.

    Returns
    -------
    None
        Drawing is performed on ``ax`` in place.
    """
    # Step 1: Remove any previous drawing on this axes.
    ax.clear()

    # Step 2: Observer inclination in radians.
    theta = np.radians(theta_deg)

    # Step 3: Disk radius samples and angular resolution for φ grids.
    rmin, rmax, rsteps = 5, 25, 5
    steps = 300

    # Step 4: Image orders (direct, lensed paths, photon ring) with plot colours.
    orders = [
        {"n": 0, "color": "navy", "label": "Direct"},
        {"n": 1, "color": "darkgreen", "label": "Secondary"},
        {"n": 2, "color": "darkred", "label": "Photon ring"},
    ]

    for order in orders:
        n = order["n"]

        # Step 5: For each disk radius r_s, build a closed curve in the (α, β) plane.
        for rs in range(rmin, rmax, rsteps):
            # Step 5a: Maximum impact parameter compatible with this disk radius.
            bmax = np.sqrt(rs**3 / (rs - 2))

            # Step 5b: Sample φ along the ray; map to disk-plane angles.
            phivals = np.linspace(0.001 * np.pi, 1.999 * np.pi, steps)

            varphivals = np.array([varphi_func(p, theta) for p in phivals])

            psivals1 = n * np.pi + np.array([psi_func(p, theta) for p in phivals])

            # Step 5c: Scan impact parameters b to tabulate (ψ, b) pairs from φ integrals.
            b_search = np.linspace(3.001, bmax, 400)

            psivals2 = []
            psivals3 = []

            for b in b_search:
                rt = minrs(b)

                if rs > rt:
                    val1 = phiFunction(-np.sqrt(rs - rt), b)
                    val2 = phiFunction(np.sqrt(rs - rt), b)

                    psivals2.append([val1, b])
                    psivals3.append([val2, b])

            combined = np.vstack((psivals2, psivals3))

            # Step 5d: For each target ψ, pick the closest matching b from the table.
            bvals = []

            for psi in psivals1:
                idx = np.nanargmin(np.abs(combined[:, 0] - psi))
                bvals.append(combined[idx, 1])

            bvals = np.array(bvals)

            # Step 5e: Convert (b, φ̃) to sky coordinates (α, β) and plot the ring segment.
            alpha = bvals * np.cos(varphivals)
            beta = -bvals * np.sin(varphivals)

            x_rot = beta
            y_rot = -alpha

            ax.plot(x_rot, y_rot, color=order["color"], alpha=0.8)

    # Step 6: Schwarzschild shadow (fixed radius √27 in these units).
    shadow = plt.Circle((0, 0), np.sqrt(27), color="black")
    ax.add_patch(shadow)

    # Step 7: View limits, aspect, title, and axis labels.
    ax.set_facecolor("black")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_aspect("equal")

    ax.set_title(f"Black Hole Accretion Disk (Inclination: {theta_deg}°)")
    ax.set_xlabel(r"$\alpha$ (Impact Parameter X)")
    ax.set_ylabel(r"$\beta$ (Impact Parameter Y)")
