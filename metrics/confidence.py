from scipy.stats import f

def compute_dynamic_ci_bounds(
    n_years: int, 
    n_members: int, 
    detrend: bool = True
) -> tuple[list[float], list[float], float, tuple[float, float]]:
    """
    Computes exact F-test critical bounds for SVR and NVR maps based on N years and M members.
    """
    df_time = max(1, n_years - 2 if detrend else n_years - 1)
    df_noise = max(1, n_years * (n_members - 1))

    # 1. SVR: 1-tailed 95% critical threshold for variance ratio
    svr_crit = float(f.ppf(0.95, df_time, df_time))
    
    # Custom colorbar bounds for SVR anchored on svr_crit
    svr_bounds = [0.0, 0.25, 0.5, 0.75, 1.0, round(svr_crit, 2), 3.0, 5.0]
    # Ensure monotonic ordering in case svr_crit exceeds 3.0 or falls below 1.0
    svr_bounds = sorted(list(set(svr_bounds)))

    # 2. NVR: 2-tailed 95% CI bounds centered around 1.0
    nvr_lower = float(f.ppf(0.025, df_noise, df_time))
    nvr_upper = float(f.ppf(0.975, df_noise, df_time))

    nvr_bounds = [0.0, round(nvr_lower / 2, 2), round(nvr_lower, 2), round(nvr_upper, 2), round(nvr_upper * 2, 2), 8.0]
    nvr_bounds = sorted(list(set(nvr_bounds)))

    return svr_bounds, nvr_bounds, svr_crit, (nvr_lower, nvr_upper)
