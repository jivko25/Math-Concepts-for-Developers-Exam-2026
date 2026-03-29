import numpy as np
import math

# Speed expressed as a fraction of the speed of light is a little difficult to imagine, so for convenience, we will convert from our familiar kilometers per hour to a fraction of the speed of light.
def kmh_to_c_fraction(speed_kmh):
    """
    Convert speed in km/h to fraction of the speed of light (v/c).
    
    Parameters:
    speed_kmh : float or array-like
        Speed in kilometers per hour
    
    Returns:
    v_over_c : float or array-like
        Speed as fraction of c
    """
    c_kmh = 299792458 * 3600 / 1000  # convert m/s to km/h
    v_over_c = speed_kmh / c_kmh
    return v_over_c

def c_fraction_to_kmh(v_over_c):
    """
    Convert speed as a fraction of the speed of light (v/c) to km/h.
    
    Parameters:
    v_over_c : float or array-like
        Speed as a fraction of the speed of light (0 <= v/c < 1)
    
    Returns:
    speed_kmh : float or array-like
        Speed in kilometers per hour
    """
    c_kmh = 299_792_458 * 3600 / 1000  # speed of light in km/h
    v_over_c = np.array(v_over_c)  # ensure array for vectorized operations
    if np.any(v_over_c < 0) or np.any(v_over_c >= 1):
        raise ValueError("v/c must be between 0 and 1 (not inclusive).")
    speed_kmh = v_over_c * c_kmh
    return speed_kmh

def time_dilation(v_over_c, t0=1):
    """
    Calculate time dilation for a moving object.
    
    Parameters:
    v_over_c : float or array-like
        Speed of the moving object as a fraction of the speed of light (0 <= v/c < 1)
    t0 : float
        Proper time in seconds (time measured in the moving clock)
    
    Returns:
    gamma : float or array-like
        Lorentz factor
    t : float or array-like
        Time measured by stationary observer
    delta_t : float or array-like
        Difference between stationary and proper time
    """
    v_over_c = np.array(v_over_c)  # ensure array for vectorized operations
    if np.any(v_over_c >= 1) or np.any(v_over_c < 0):
        raise ValueError("v/c must be between 0 and 1 (not inclusive).")
    
    gamma = 1 / np.sqrt(1 - v_over_c**2)
    t = gamma * t0
    delta_t = t - t0
    
    return gamma, t, delta_t

def satellite_time_dilation(t0, speed_in_km_per_h):    
    # Satellite speed: 13932 km/h (typical GPS satellite)
    v = kmh_to_c_fraction(speed_in_km_per_h)
    
    # Calculate Lorentz factor, observed time, and difference
    gamma, t, delta_t = time_dilation(v, t0)
    
    # --------------------------
    # Output results
    # --------------------------
    print("=== Time Dilation for Satellite Example ===")
    print(f"Satellite speed compared to the speed of light: {v:.10f} c")
    print(f"Lorentz factor γ: {gamma:.10f}")
    print(f"Proper time on satellite clock t0: {t0:.10f} s")
    print(f"Time measured by Earth observer t: {t:.10f} s")
    print(f"Time difference Δt = t - t0: {delta_t:.3f} s (~{delta_t/60:.3f} minutes)")

def muons_time_dilation(tau_0, v_fraction):    
    # Lorentz factor
    gamma = 1 / math.sqrt(1 - v_fraction**2)
    
    # Lifetime as seen from Earth
    tau_observed = gamma * tau_0
    delta_tau = tau_observed - tau_0
    
    print("=== Time Dilation for Muon Example ===")
    print(f"Muon speed compared to c: {v_fraction} c")
    print(f"Lorentz factor γ: {gamma:.3f}")
    print(f"Proper lifetime τ0: {tau_0:.2e} s")
    print(f"Lifetime observed from Earth: {tau_observed:.2e} s")
    print(f"Difference Δτ = τ - τ0: {delta_tau:.2e} s")

def speed_from_time_dilation(t_observed, t_proper=1):
    """
    Calculate speed (as a fraction of c) from time dilation.

    Parameters:
    t_observed : float or array-like
        Time measured by stationary observer
    t_proper : float
        Proper time measured in the moving frame (default 1 second)

    Returns:
    v_over_c : float or array-like
        Speed of the moving object as a fraction of the speed of light
    gamma : float or array-like
        Lorentz factor
    """
    t_observed = np.array(t_observed)
    t_proper = np.array(t_proper)

    if np.any(t_observed < t_proper):
        raise ValueError("Observed time must be >= proper time.")

    gamma = t_observed / t_proper
    v_over_c = np.sqrt(1 - 1/gamma**2)

    return v_over_c, gamma

def twin_paradox(distance_km, speed_kmh):
    """
    Twin Paradox calculation with proper units and descriptive prints.
    
    Parameters:
    distance_km : float
        One-way distance the spaceship travels in kilometers
    speed_kmh : float
        Constant speed of the spaceship in km/h
    
    Returns:
    dict : with all results in years, days, hours
    """
    # Convert speed to fraction of c
    v_over_c = kmh_to_c_fraction(speed_kmh)
    
    # Round-trip time as measured on Earth (hours)
    t_earth_hours = 2 * distance_km / speed_kmh
    
    # Lorentz factor
    gamma = 1 / np.sqrt(1 - v_over_c**2)
    
    # Traveler proper time
    t_traveler_hours = t_earth_hours / gamma
    
    # Time difference
    delta_hours = t_earth_hours - t_traveler_hours
    
    # Convert hours to days and years
    t_earth_days = t_earth_hours / 24
    t_earth_years = t_earth_days / 365.25
    
    t_traveler_days = t_traveler_hours / 24
    t_traveler_years = t_traveler_days / 365.25
    
    delta_days = delta_hours / 24
    delta_years = delta_days / 365.25
    
    # ----------- Prints with descriptive text -----------
    print(f"Distance traveled (one-way): {distance_km:,.0f} km")
    print(f"Traveling spaceship speed: {speed_kmh:,.0f} km/h")
    print(f"Total time elapsed on Earth (round trip): {t_earth_years:.2f} years / {t_earth_days:,.0f} days / {t_earth_hours:,.0f} hours")
    print(f"Total time experienced by traveling twin: {t_traveler_years:.2f} years / {t_traveler_days:,.0f} days / {t_traveler_hours:,.0f} hours")
    print(f"Time difference between Earth and traveling twin: {delta_years:.6f} years / {delta_days:.2f} days / {delta_hours:.2f} hours")
    print(f"Lorentz gamma factor: {gamma:.6f}")
    
    # Return dictionary for further use
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
        "Gamma": gamma
    }