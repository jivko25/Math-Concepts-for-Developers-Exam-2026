import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import pandas as pd
import warnings
import requests
from io import StringIO

from calculation_functions import kmh_to_c_fraction, time_dilation

# Light clock stationary animation
def light_clock_stationary():
    d = 5  # distance between mirrors
    frames = 120
    
    fig, ax = plt.subplots(figsize=(4,6))
    
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,d+1)
    
    ax.set_title("Light Clock (Stationary)")
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")
    
    # mirrors
    ax.scatter([0,0],[0,d], color="black")
    
    ax.text(0.1,d,"Top mirror")
    ax.text(0.1,0,"Bottom mirror")
    
    light, = ax.plot([], [], 'ro', markersize=10)
    
    def animate(frame):
    
        period = 60
        
        phase = frame % (2*period)
    
        if phase < period:
            y = (phase/period) * d
        else:
            y = d - ((phase-period)/period) * d
    
        light.set_data([0],[y])
    
        return light,
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=40)
    
    plt.close()
    
    return HTML(ani.to_jshtml())

# Light clock moving animation
def light_clock_moving():
    d = 5
    v = 0.3
    frames = 200
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set_xlim(-1,8)
    ax.set_ylim(-1,d+1)
    
    ax.set_title("Moving Light Clock (external observer)")
    ax.set_xlabel("horizontal position")
    ax.set_ylabel("vertical position")
    
    # mirrors
    top_mirror, = ax.plot([], [], 'ks', markersize=8)
    bottom_mirror, = ax.plot([], [], 'ks', markersize=8)
    
    # light particle
    light, = ax.plot([], [], 'ro', markersize=8)
    
    # diagonal path (instant triangle side)
    path, = ax.plot([], [], 'r--', alpha=0.6)
    
    # traces
    clock_trace_x = []
    clock_trace_y = []
    
    light_trace_x = []
    light_trace_y = []
    
    clock_trace_line, = ax.plot([], [], 'k-', alpha=0.3)
    light_trace_line, = ax.plot([], [], 'r-', alpha=0.3)
    
    
    def animate(frame):
    
        t = frame * 0.05
        x_clock = v * t
    
        period = 40
        phase = frame % (2*period)
    
        if phase < period:
            y = (phase/period) * d
        else:
            y = d - ((phase-period)/period) * d
    
        light_x = x_clock
        light_y = y
    
        # move mirrors
        top_mirror.set_data([x_clock],[d])
        bottom_mirror.set_data([x_clock],[0])
    
        # move light
        light.set_data([light_x],[light_y])
    
        # triangle hypotenuse
        path.set_data([x_clock, light_x],[0, light_y])
    
        # save traces
        clock_trace_x.append(x_clock)
        clock_trace_y.append(0)
    
        light_trace_x.append(light_x)
        light_trace_y.append(light_y)
    
        # draw traces
        clock_trace_line.set_data(clock_trace_x, clock_trace_y)
        light_trace_line.set_data(light_trace_x, light_trace_y)
    
        return (
            top_mirror,
            bottom_mirror,
            light,
            path,
            clock_trace_line,
            light_trace_line
        )
    
    
    ani = FuncAnimation(fig, animate, frames=frames, interval=40)
    
    plt.close(fig)
    
    return HTML(ani.to_jshtml())

def geometry_of_light_clock():
    d = 5
    v = 0.6
    t = 4
    c = 1
    
    horizontal = v * t
    vertical = d
    hypotenuse = c * t
    
    plt.figure(figsize=(6,6))
    
    # triangle points
    x = [0, horizontal, 0, 0]
    y = [0, 0, vertical, 0]
    
    plt.plot(x, y)
    
    # label sides
    plt.text(horizontal/2, -0.3, "v*t", ha="center")
    plt.text(-0.3, vertical/2, "d", va="center")
    plt.text(horizontal/2, vertical/2, "c*t")
    
    plt.scatter([0, horizontal, 0], [0, 0, vertical])
    
    plt.title("Geometry of the Light Clock")
    plt.xlabel("horizontal distance")
    plt.ylabel("vertical distance")
    
    plt.axis("equal")
    plt.grid(True)
    
    return plt.show()

def stationary_vs_moving_clock():
    # Proper time for moving clock
    t0 = 1  # seconds
    
    # Speeds as fraction of c
    v_over_c = np.linspace(0, 0.99, 500)
    
    # Lorentz factor
    gamma = 1 / np.sqrt(1 - v_over_c**2)
    
    # Time measured by stationary observer (moving clock)
    t_moving = gamma * t0
    
    # Time for stationary clock (always t0)
    t_stationary = np.full_like(v_over_c, t0)
    
    # Plotting both curves
    plt.figure(figsize=(8,5))
    plt.plot(v_over_c, t_stationary, color='gray', linestyle='--', linewidth=2, label='Stationary clock')
    plt.plot(v_over_c, t_moving, color='blue', linewidth=2, label='Moving clock (time dilation)')
    
    # Highlight example points
    plt.axvline(x=0.1, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=0.9, color='red', linestyle='--', alpha=0.5)
    plt.text(0.11, t0+0.02, f'v=0.1c', color='green')
    plt.text(0.91, t_moving[-1]-0.05, f'v=0.9c', color='red')
    
    # Labels, title and legend
    plt.title('Time Dilation: Stationary vs Moving Clock', fontsize=14)
    plt.xlabel('v/c (fraction of speed of light)', fontsize=12)
    plt.ylabel('Time observed (s)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()


def planet_orbits(url):
    # TODO: !!! Improve handling of the data !!!
    # ----------- Step 1: Fetching and cleaning the dataset -----------
    # Download CSV data from the provided URL
    resp = requests.get(url)
    resp.raise_for_status()  # Ensure the request was successful
    
    # Load data into a DataFrame
    df = pd.read_csv(StringIO(resp.text))
    
    # Normalize column names: lowercase, remove spaces and dashes
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    
    
    # ----------- Step 2: Adding manual data (orbital velocity and rotation) -----------
    # Orbital velocity in km/s (average orbital speed around the Sun)
    velocity_data = {
        "Mercury": 47.4, "Venus": 35.0, "Earth": 29.8, "Mars": 24.1,
        "Jupiter": 13.1, "Saturn": 9.7, "Uranus": 6.8, "Neptune": 5.4
    }
    
    # This dictionary stores the rotation period of each planet in hours.
    # The values represent how long it takes for each planet to complete one full rotation on its axis (a "day").
    # Positive values indicate prograde rotation (same direction as most planets, like Earth).
    # Negative values indicate retrograde rotation (the planet rotates in the opposite direction compared to the Earth, e.g., Venus and Uranus).
    # For example, Earth rotates once every ~23.934 hours, while Venus rotates very slowly and in reverse (~-5832.5 hours).
    rotation_hours_data = {
        "Mercury": 1407.6, "Venus": -5832.5, "Earth": 23.934, "Mars": 24.623,
        "Jupiter": 9.925, "Saturn": 10.656, "Uranus": -17.24, "Neptune": 16.11
    }
    
    # Map planet names to their corresponding velocities and rotation periods
    # Capitalization is adjusted to match dictionary keys
    df['orbital_velocity_kms'] = df['name'].str.capitalize().map(velocity_data)
    df['rotation_hours'] = df['name'].str.capitalize().map(rotation_hours_data)
    
    
    # ----------- Step 3: Calculations for surface and total velocity -----------
    # Compute planetary radius (km) from diameter
    df['radius_km'] = df['diameter'] / 2
    
    # Surface rotational speed (km/h)
    # Formula: circumference / rotation period = (2πR) / T
    # Absolute value is used because rotation direction (sign) does not affect speed magnitude
    df['surface_speed_kmh'] = (2 * np.pi * df['radius_km']) / np.abs(df['rotation_hours'])
    
    # Convert orbital velocity from km/s to km/h
    df['orbital_velocity_kmh'] = df['orbital_velocity_kms'] * 3600
    
    # Total effective speed (orbital motion + rotational surface speed)
    # Note: This is a simplified scalar sum and does not account for vector directions
    df['total_speed_kmh'] = df['orbital_velocity_kmh'] + df['surface_speed_kmh']
    
    
    # ----------- Step 4: Orbit visualization -----------
    plt.figure(figsize=(10, 10))
    
    for _, row in df.iterrows():
        # Semi-major axis (a) and eccentricity (e) define the ellipse
        a = row['semi_majoraxis']  # in Astronomical Units (AU)
        e = row['eccentricity']
        
        # Semi-minor axis (b) derived from ellipse equation
        b = a * np.sqrt(1 - e**2)
        
        # Parametric equation of an ellipse
        theta = np.linspace(0, 2*np.pi, 360)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        
        # Label includes orbital, surface, and total speeds
        label_text = (f"{row['name']}: Orb {row['orbital_velocity_kmh']:.0f} km/h | "
                      f"Surf {row['surface_speed_kmh']:.1f} km/h | "
                      f"Total {row['total_speed_kmh']:.0f} km/h")
        
        plt.plot(x, y, label=label_text, lw=1.5)
    
    # Plot the Sun at the center (focus approximation)
    plt.scatter(0, 0, color='yellow', s=250, label='Sun', edgecolors='orange')
    
    # Plot styling
    plt.title("Planetary Orbits: Orbital + Surface Rotation Speeds", fontsize=14)
    plt.xlabel("Distance (AU)", fontsize=12)
    plt.ylabel("Distance (AU)", fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=9)
    
    # Ensure equal scaling on both axes (important for correct orbit shapes)
    plt.gca().set_aspect('equal')
    
    # Background and grid styling
    plt.gca().set_facecolor('#f0f0f0')
    plt.grid(color='white', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    
    return df

def planets_time_dilation_linear_plot(df):
    # ----------- Convert speed to fraction of c -----------
    df['v_over_c'] = kmh_to_c_fraction(df['total_speed_kmh'])
    
    # ----------- Calculate time dilation -----------
    t0_hours = 365.25 * 24  # 1 Earth year in hours
    gamma, t_hours, delta_t_hours = time_dilation(df['v_over_c'], t0=t0_hours)
    
    df['gamma'] = gamma
    df['observed_time_hours'] = t_hours
    df['time_difference_hours'] = delta_t_hours
    
    # Convert time difference to seconds for visualization
    df['time_difference_seconds'] = df['time_difference_hours'] * 3600
    
    # ----------- Linear line plot of time differences -----------
    plt.figure(figsize=(10,6))
    plt.plot(df['name'], df['time_difference_seconds'], marker='o', color='blue', linewidth=2)
    plt.ylabel("Time difference vs Earth (seconds per year)", fontsize=12)
    plt.xlabel("Planet", fontsize=12)
    plt.title("Time Dilation on Planetary Surfaces (Linear Plot)", fontsize=14)
    plt.ylim(0)  # start from 0 for linear visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


# Start of Black Hole functions #

def minrs(b):
    with np.errstate(all='ignore'):
        val = np.cbrt(b**2) * (
            (-1 + np.sqrt(1 - b**2/27 + 0j))**(1/3) +
            (-1 - np.sqrt(1 - b**2/27 + 0j))**(1/3)
        )

    if np.isnan(val):
        return 0
    if np.abs(np.imag(val)) < 1e-8:
        return np.real(val)
    return 0


def d_phi_dx(x, b, rt):
    denom = (x**2 + rt)
    inside = (x**2 + rt)**2 - (b**2 * (-2 + x**2 + rt)) / denom

    if inside <= 0:
        return 0

    return -2*b*np.abs(x) / (denom*np.sqrt(inside))

def psi_func(phi_val, theta):
    return np.arccos(np.sin(theta)*np.sin(phi_val))


def varphi_func(phi_val, theta):
    return np.arctan2(np.cos(phi_val), np.cos(theta)*np.sin(phi_val))

def plot_blackhole(ax, phiFunction, theta_deg=80):
    """
    Main function to render the accretion disk visualization.
    Designed to work with animations (draws on existing axis).
    """

    ax.clear()

    # Convert observer inclination to radians
    theta = np.radians(theta_deg)

    rmin, rmax, rsteps = 5, 25, 5
    steps = 300

    orders = [
        {'n':0, 'color':'navy', 'label':'Direct'},
        {'n':1, 'color':'darkgreen', 'label':'Secondary'},
        {'n':2, 'color':'darkred', 'label':'Photon ring'}
    ]

    for order in orders:

        n = order['n']

        for rs in range(rmin, rmax, rsteps):

            bmax = np.sqrt(rs**3/(rs-2))

            phivals = np.linspace(0.001*np.pi, 1.999*np.pi, steps)

            varphivals = np.array([
                varphi_func(p, theta) for p in phivals
            ])

            psivals1 = n*np.pi + np.array([
                psi_func(p, theta) for p in phivals
            ])

            b_search = np.linspace(3.001, bmax, 400)

            psivals2 = []
            psivals3 = []

            for b in b_search:

                rt = minrs(b)

                if rs > rt:

                    val1 = phiFunction(-np.sqrt(rs-rt), b)
                    val2 = phiFunction(np.sqrt(rs-rt), b)

                    psivals2.append([val1, b])
                    psivals3.append([val2, b])

            combined = np.vstack((psivals2, psivals3))

            bvals = []

            for psi in psivals1:
                idx = np.nanargmin(np.abs(combined[:,0] - psi))
                bvals.append(combined[idx,1])

            bvals = np.array(bvals)

            alpha = bvals * np.cos(varphivals)
            beta = -bvals * np.sin(varphivals)

            x_rot = beta
            y_rot = -alpha

            ax.plot(x_rot, y_rot, color=order['color'], alpha=0.8)

    # Draw black hole shadow
    shadow = plt.Circle((0,0), np.sqrt(27), color='black')
    ax.add_patch(shadow)

    # Visual formatting
    ax.set_facecolor("black")
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_aspect("equal")

    ax.set_title(f"Black Hole Accretion Disk (Inclination: {theta_deg}°)")
    ax.set_xlabel(r"$\alpha$ (Impact Parameter X)")
    ax.set_ylabel(r"$\beta$ (Impact Parameter Y)")

# End of Black Hole functions