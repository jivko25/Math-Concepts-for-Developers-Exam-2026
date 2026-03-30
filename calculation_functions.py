"""
Calculation helpers for special-relativistic time dilation and related demos.

This module converts everyday speeds to fractions of *c*, applies the Lorentz factor,
and bundles a few educational examples (satellite, muons, twin paradox) that print
human-readable summaries. All functions use SI-friendly inputs where noted.
"""

import math

import numpy as np


def kmh_to_c_fraction(speed_kmh):
    """
    Express a speed given in km/h as a dimensionless fraction of the speed of light (v/c).

    Purpose
    -------
    Relativistic formulas use v/c. Converting from km/h makes intuitive speeds (cars,
    satellites) comparable to *c* without manual unit algebra.

    Parameters
    ----------
    speed_kmh : float or array-like
        Speed in kilometres per hour. May be a scalar or a NumPy-compatible array for
        vectorised use.

    Steps
    -----
    1. Define the speed of light in km/h from the exact value c = 299792458 m/s.
    2. Divide the input speed by that constant to obtain v/c.

    Returns
    -------
    float or array-like
        Same shape as ``speed_kmh``: speed as a fraction of *c* (not clamped; callers
        should keep |v/c| < 1 for time-dilation formulas).
    """
    # Step 1: Express c in m/s as km/h (×3600 s/h, ÷1000 m/km).
    c_kmh = 299792458 * 3600 / 1000

    # Step 2: Dimensionless ratio v/c = (km/h) / (km/h).
    v_over_c = speed_kmh / c_kmh

    # Return: v/c with the same shape as speed_kmh.
    return v_over_c


def c_fraction_to_kmh(v_over_c):
    """
    Convert a speed expressed as v/c back into kilometres per hour.

    Purpose
    -------
    Inverse of ``kmh_to_c_fraction``; useful for reporting results (e.g. “equivalent
    speed in km/h”) after working in natural units.

    Parameters
    ----------
    v_over_c : float or array-like
        Speed as a fraction of the speed of light. Must satisfy 0 <= v/c < 1 for every
        element (strictly below 1 to stay subluminal).

    Steps
    -----
    1. Coerce the input to a NumPy array for uniform validation and broadcasting.
    2. Validate that all values lie in [0, 1).
    3. Multiply v/c by the speed of light in km/h.

    Returns
    -------
    float or numpy.ndarray
        Speed(s) in km/h.

    Raises
    ------
    ValueError
        If any element is outside [0, 1).
    """
    # Step 1: Speed of light in km/h (same convention as kmh_to_c_fraction).
    c_kmh = 299_792_458 * 3600 / 1000

    # Step 2: Coerce to ndarray for uniform validation and broadcasting.
    v_over_c = np.array(v_over_c)

    # Step 3: Reject negative or superluminal (v/c ≥ 1) values.
    if np.any(v_over_c < 0) or np.any(v_over_c >= 1):
        raise ValueError("v/c must be between 0 and 1 (not inclusive).")

    # Step 4: Recover speed in km/h from v/c.
    speed_kmh = v_over_c * c_kmh

    # Return: speed(s) in km/h.
    return speed_kmh


def time_dilation(v_over_c, t0=1):
    """
    Compute special-relativistic time dilation for one or more subluminal speeds.

    Purpose
    -------
    Given proper time ``t0`` in the moving frame, return the corresponding coordinate
    time for a stationary observer using t = γ t₀ with γ = 1/√(1 − v²/c²).

    Parameters
    ----------
    v_over_c : float or array-like
        Speed as v/c for each case. Must satisfy 0 <= v/c < 1 for every element.
    t0 : float, optional
        Proper time interval (default 1), in the same time unit you want for outputs
        (often seconds or hours).

    Steps
    -----
    1. Coerce ``v_over_c`` to an array and validate the [0, 1) range.
    2. Compute the Lorentz factor γ = 1 / sqrt(1 − (v/c)²).
    3. Multiply γ by ``t0`` to get observed time ``t``.
    4. Form the excess Δt = t − t0.

    Returns
    -------
    gamma : float or numpy.ndarray
        Lorentz factor γ (same shape as broadcast inputs).
    t : float or numpy.ndarray
        Time measured by the “lab” observer.
    delta_t : float or numpy.ndarray
        Difference t − t0.

    Raises
    ------
    ValueError
        If any v/c is outside [0, 1).
    """
    # Step 1: Normalise v/c to an array and enforce 0 ≤ v/c < 1.
    v_over_c = np.array(v_over_c)
    if np.any(v_over_c >= 1) or np.any(v_over_c < 0):
        raise ValueError("v/c must be between 0 and 1 (not inclusive).")

    # Step 2: Lorentz factor γ = 1 / √(1 − (v/c)²).
    gamma = 1 / np.sqrt(1 - v_over_c**2)

    # Step 3: Coordinate time in the lab frame: t = γ t₀.
    t = gamma * t0

    # Step 4: Extra time elapsed vs proper time: Δt = t − t₀.
    delta_t = t - t0

    # Return: γ, lab-frame time t, and difference Δt.
    return gamma, t, delta_t


def satellite_time_dilation(t0, speed_in_km_per_h):
    """
    Demonstrate time dilation for a satellite using printed diagnostics.

    Purpose
    -------
    Wraps ``kmh_to_c_fraction`` and ``time_dilation`` for a typical scenario (e.g. GPS
    orbital speed) and prints a formatted summary. Intended for notebooks and teaching,
    not as a silent numerical API.

    Parameters
    ----------
    t0 : float
        Proper-time interval on the satellite clock (e.g. one year in seconds).
    speed_in_km_per_h : float
        Satellite speed relative to the chosen Earth frame, in km/h (e.g. ~13932 for GPS).

    Steps
    -----
    1. Convert the satellite speed from km/h to v/c.
    2. Call ``time_dilation`` with that v/c and ``t0``.
    3. Print v/c, γ, proper time, observed time, and Δt in seconds (and minutes).

    Returns
    -------
    None
        Results are only printed to standard output.
    """
    # Step 1: Satellite speed as a fraction of c.
    v = kmh_to_c_fraction(speed_in_km_per_h)

    # Step 2: Lorentz factor, dilated time, and Δt for proper time t₀.
    gamma, t, delta_t = time_dilation(v, t0)

    # Step 3: Human-readable summary (side effect only; no return value).
    print("=== Time Dilation for Satellite Example ===")
    print(f"Satellite speed compared to the speed of light: {v:.10f} c")
    print(f"Lorentz factor γ: {gamma:.10f}")
    print(f"Proper time on satellite clock t0: {t0:.10f} s")
    print(f"Time measured by Earth observer t: {t:.10f} s")
    print(f"Time difference Δt = t - t0: {delta_t:.3f} s (~{delta_t/60:.3f} minutes)")


def muons_time_dilation(tau_0, v_fraction):
    """
    Illustrate muon lifetime extension via time dilation (printed example).

    Purpose
    -------
    Muons decay in their rest frame with mean life τ₀; Earth observers see a longer
    lifetime τ = γ τ₀. This function computes and prints those quantities for pedagogy.

    Parameters
    ----------
    tau_0 : float
        Proper mean lifetime (or interval) in seconds in the muon’s rest frame.
    v_fraction : float
        Speed as v/c (e.g. 0.98 for a fast atmospheric muon). Should satisfy |v/c| < 1.

    Steps
    -----
    1. Compute γ using the scalar ``math.sqrt`` (single speed).
    2. Multiply τ₀ by γ to get the dilated lifetime seen from the lab/Earth frame.
    3. Print γ, τ₀, τ, and Δτ = τ − τ₀.

    Returns
    -------
    None
        Output is textual only.
    """
    # Step 1: Lorentz factor for the given v/c (scalar path).
    gamma = 1 / math.sqrt(1 - v_fraction**2)

    # Step 2: Dilated mean lifetime seen from Earth: τ = γ τ₀.
    tau_observed = gamma * tau_0

    # Step 3: Extension Δτ relative to rest-frame lifetime.
    delta_tau = tau_observed - tau_0

    # Step 4: Print summary (no return).
    print("=== Time Dilation for Muon Example ===")
    print(f"Muon speed compared to c: {v_fraction} c")
    print(f"Lorentz factor γ: {gamma:.3f}")
    print(f"Proper lifetime τ0: {tau_0:.2e} s")
    print(f"Lifetime observed from Earth: {tau_observed:.2e} s")
    print(f"Difference Δτ = τ - τ0: {delta_tau:.2e} s")


def speed_from_time_dilation(t_observed, t_proper=1):
    """
    Infer speed v/c (and γ) when coordinate time and proper time are known.

    Purpose
    -------
    From t = γ t₀ we get γ = t/t₀, then v/c = √(1 − 1/γ²). Useful for “inverse”
    problems and thought experiments.

    Parameters
    ----------
    t_observed : float or array-like
        Time measured in the “stationary” frame (coordinate time).
    t_proper : float or array-like, optional
        Proper time in the moving frame (default 1). Must broadcast with
        ``t_observed``; every pair must satisfy t_observed >= t_proper.

    Steps
    -----
    1. Coerce inputs to NumPy arrays for vectorised comparison and arithmetic.
    2. Require t_observed >= t_proper (otherwise γ < 1, unphysical for this model).
    3. Compute γ = t_observed / t_proper.
    4. Recover v/c = sqrt(1 − 1/γ²).

    Returns
    -------
    v_over_c : float or numpy.ndarray
        Inferred speed as a fraction of c.
    gamma : float or numpy.ndarray
        Corresponding Lorentz factor.

    Raises
    ------
    ValueError
        If any observed time is strictly less than proper time.
    """
    # Step 1: Coerce times to arrays for vectorised checks and algebra.
    t_observed = np.array(t_observed)
    t_proper = np.array(t_proper)

    # Step 2: Require t ≥ t₀ so that γ = t/t₀ ≥ 1 (time dilation regime).
    if np.any(t_observed < t_proper):
        raise ValueError("Observed time must be >= proper time.")

    # Step 3: Lorentz factor from the ratio of coordinate to proper time.
    gamma = t_observed / t_proper

    # Step 4: Invert γ to obtain v/c = √(1 − 1/γ²).
    v_over_c = np.sqrt(1 - 1 / gamma**2)

    # Return: inferred v/c and the corresponding γ.
    return v_over_c, gamma


def twin_paradox(distance_km, speed_kmh):
    """
    Compute round-trip twin-paradox times for a constant-speed journey (simplified model).

    Purpose
    -------
    Assume instantaneous turnaround and constant speed: Earth time for the round trip
    is 2d/v (kinematic); the traveller’s proper time is that divided by γ. Prints a
    summary and returns all key numbers in hours/days/years.

    Parameters
    ----------
    distance_km : float
        One-way distance from Earth to turnaround point, in kilometres.
    speed_kmh : float
        Constant speed of the spaceship, in km/h (must be > 0 and subluminal in practice).

    Steps
    -----
    1. Convert ``speed_kmh`` to v/c via ``kmh_to_c_fraction``.
    2. Compute Earth-frame round-trip duration: t_Earth = 2 * distance / speed (hours).
    3. Compute γ from v/c.
    4. Traveller proper time: t_traveller = t_Earth / γ.
    5. Difference Δt = t_Earth − t_traveller; convert hours to days and years (365.25 d/yr).
    6. Print a readable report.
    7. Return a dictionary of numeric results for further plotting or tables.

    Returns
    -------
    dict
        Keys include ``Earth_time_years``, ``Traveler_time_years``,
        ``Time_difference_years``, parallel *_days* and *_hours* entries, and ``Gamma``.
    """
    # Step 1: Ship speed as v/c from km/h.
    v_over_c = kmh_to_c_fraction(speed_kmh)

    # Step 2: Earth-frame round-trip duration (hours): 2 × (one-way distance / speed).
    t_earth_hours = 2 * distance_km / speed_kmh

    # Step 3: Lorentz factor for that cruise speed.
    gamma = 1 / np.sqrt(1 - v_over_c**2)

    # Step 4: Traveller’s proper time over the same journey: t_trav = t_Earth / γ.
    t_traveler_hours = t_earth_hours / gamma

    # Step 5: Clock disagreement in hours.
    delta_hours = t_earth_hours - t_traveler_hours

    # Step 6: Convert Earth time to days and Julian years (365.25 d/yr).
    t_earth_days = t_earth_hours / 24
    t_earth_years = t_earth_days / 365.25

    # Step 7: Convert traveller time to days and years.
    t_traveler_days = t_traveler_hours / 24
    t_traveler_years = t_traveler_days / 365.25

    # Step 8: Convert Δt to days and years.
    delta_days = delta_hours / 24
    delta_years = delta_days / 365.25

    # Step 9: Print narrative summary.
    print(f"Distance traveled (one-way): {distance_km:,.0f} km")
    print(f"Traveling spaceship speed: {speed_kmh:,.0f} km/h")
    print(
        f"Total time elapsed on Earth (round trip): {t_earth_years:.2f} years / "
        f"{t_earth_days:,.0f} days / {t_earth_hours:,.0f} hours"
    )
    print(
        f"Total time experienced by traveling twin: {t_traveler_years:.2f} years / "
        f"{t_traveler_days:,.0f} days / {t_traveler_hours:,.0f} hours"
    )
    print(
        f"Time difference between Earth and traveling twin: {delta_years:.6f} years / "
        f"{delta_days:.2f} days / {delta_hours:.2f} hours"
    )
    print(f"Lorentz gamma factor: {gamma:.6f}")

    # Return: all computed durations and γ for reuse (plots, tables).
    return {
        "Earth_time_years": t_earth_years,
        "Traveler_time_years": t_traveler_years,
        "Time_difference_years": delta_years,
        "Earth_time_days": t_earth_days,
        "Traveler_time_days": t_traveler_days,
        "Time_difference_days": delta_days,
        "Earth_time_hours": t_earth_hours,
        "Traveler_time_hours": t_traveler_hours,
        "Time_difference_hours": delta_hours,
        "Gamma": gamma,
    }

def calculate_gravitational_dilation(name, mass_kg, radius_m):
    """
    Calculates the gravitational time dilation for a specific celestial body.
    
    Parameters:
    name (str): Name of the object (e.g., "Earth", "Sun")
    mass_kg (float): Mass of the object in kilograms
    radius_m (float): Radius from the center in meters
    
    Returns:
    None: Prints the Schwarzschild radius, dilation factor, and time offset.
    """
    # Physical Constants (SI Units)
    G = 6.67430e-11  # Universal Gravitational Constant
    c = 299792458    # Speed of light in a vacuum
    seconds_in_year = 365.25 * 24 * 3600
    
    # 1. Calculate the Schwarzschild Radius (rs = 2GM / c^2)
    # This is the radius where the event horizon would form for this mass.
    rs = (2 * G * mass_kg) / (c**2)
    
    # 2. Calculate the Dilation Factor (sqrt(1 - rs / r))
    # This factor tells us the ratio between local time and distant time.
    # If factor = 0.9, 1 second locally = 0.9 seconds for the distant observer.
    dilation_factor = math.sqrt(1 - rs / radius_m)
    
    # 3. Calculate Time Difference over 1 Year
    # How many seconds of "extra" time pass in deep space for 1 local year.
    time_offset_seconds = seconds_in_year * (1 - dilation_factor)
    
    print(f"--- Object: {name} ---")
    print(f"Schwarzschild Radius (rs): {rs:.6f} meters")
    print(f"Dilation Factor: {dilation_factor:.12f}")
    print(f"Time difference per year: {time_offset_seconds:.6f} seconds")
    print("-" * 40)

def calculate_millers_planet_kerr():
    """
    Improved approximation using Kerr black hole intuition.
    Returns:
    - orbital radius (km)
    - height above horizon (km)
    """

    # Constants
    c = 299792.458  # km/s
    G = 6.67430e-20 # km^3 / kg / s^2
    M_sun = 1.98847e30  # kg

    # Black hole mass (Gargantua)
    M = 100e6 * M_sun

    # Schwarzschild radius
    rs = 2 * G * M / (c**2)

    # Time dilation target
    t_proper = 1  # hour
    t_far = 7 * 365.25 * 24  # hours
    dilation = t_proper / t_far

    # --- Kerr assumptions ---
    spin = 0.999  # near-maximal spin

    # Horizon radius for Kerr:
    # r+ = rs/2 * (1 + sqrt(1 - a^2))
    r_horizon = (rs / 2) * (1 + math.sqrt(1 - spin**2))

    # Empirical factor:
    # Miller's planet sits slightly above horizon
    # tuned to match ~1/61320 dilation
    k = 1.08  # ~8% above horizon

    r = r_horizon * k

    height = r - r_horizon

    # Estimate orbital velocity (more realistic range)
    v = 0.6 * c
    gamma = 1 / math.sqrt(1 - (v/c)**2)

    print("--- Miller's Planet (Kerr Approx) ---")
    print(f"Schwarzschild radius: {rs:,.0f} km")
    print(f"Kerr horizon radius: {r_horizon:,.0f} km")
    print(f"Orbital radius: {r:,.0f} km")
    print(f"Height above horizon: {height:,.0f} km")
    print(f"Estimated orbital speed: {v/c:.2f} c")
    print(f"Lorentz gamma: {gamma:.2f}")

    return r, height
