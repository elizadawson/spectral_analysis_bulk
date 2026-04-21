import os
import warnings
import numpy as np
import pandas as pd
import pyproj
from tqdm import tqdm
from scipy.stats import linregress
from scipy.signal import peak_prominences


def calc_amp_spectra(
    rx_complex, fs, fc, tidx_sfc, tidx_bottom, window_len=256, bad_traces=None
):
    """
    Compute the STFT for each trace in rx_complex and apply spectral ratio method.

    Parameters:
    -----------
    rx_complex : ndarray
        The complex radar data (2D array with traces and samples).
    fs : float
        Sampling frequency in Hz.
    fc : float
        Center frequency in Hz.
    window_len : int, optional
        Length of the window for STFT (default 256).
    tidx_sfc : ndarray
        Array of indices for the surface time window for each trace.
    tidx_bottom : ndarray
        Array of indices for the bottom time window for each trace.
    bad_traces : ndarray, optional
        Array indicating bad traces (1 for bad, 0 for good).

    Returns:
    --------
    radar_amp : dict
        Dictionary containing arrays of amp_sfc, amp_bottom, and amp_noise for each trace.
        f : frequency from the STFT.
        f_mhz : frequency in MHz.
        t : travel time from the STFT.
    """

    if bad_traces is None:
        bad_traces = np.zeros(rx_complex.shape[1])

    tidx_sfc = np.array(tidx_sfc)
    tidx_bottom = np.array(tidx_bottom)

    # Initialize arrays to store results
    amp_sfc_arr = np.nan * np.zeros((rx_complex.shape[1], window_len))
    amp_bottom_arr = np.nan * np.zeros((rx_complex.shape[1], window_len))
    amp_noise_arr = np.nan * np.zeros((rx_complex.shape[1], window_len))

    # Loop over each trace in rx_complex
    for i in tqdm(np.where(bad_traces == 0)[0], desc="Processing Good Traces"):

        # Extract the signal for the current trace
        rx_complex_trace = rx_complex[:, i].flatten()

        # Surface window extraction
        if tidx_sfc[i] - window_len // 2 < 0:
            tidx_sfc[i] -= tidx_sfc[i] - window_len // 2

        rx_complex_sfc = rx_complex_trace[
            tidx_sfc[i] - window_len // 2 : tidx_sfc[i] + window_len // 2
        ]
        rx_complex_bottom = rx_complex_trace[
            tidx_bottom[i] - window_len // 2 : tidx_bottom[i] + window_len // 2
        ]

        # Noise floor window: halfway between bottom and the end of the radargram
        midpoint_noise = (tidx_bottom[i] + len(rx_complex_trace)) // 2
        if midpoint_noise - window_len // 2 < 0:
            midpoint_noise = window_len // 2
        if midpoint_noise + window_len // 2 > len(rx_complex_trace):
            midpoint_noise = len(rx_complex_trace) - window_len // 2

        rx_complex_noise = rx_complex_trace[
            midpoint_noise - window_len // 2 : midpoint_noise + window_len // 2
        ]

        # FFT for surface, bottom, and noise
        amp_sfc = np.fft.fftshift(np.fft.fft(rx_complex_sfc, n=window_len))
        amp_bottom = np.fft.fftshift(np.fft.fft(rx_complex_bottom, n=window_len))
        amp_noise = np.fft.fftshift(np.fft.fft(rx_complex_noise, n=window_len))

        # Convert to amplitude spectra
        amp_sfc = np.abs(amp_sfc)
        amp_bottom = np.abs(amp_bottom)
        amp_noise = np.abs(amp_noise)

        # Store results
        amp_sfc_arr[i, :] = amp_sfc
        amp_bottom_arr[i, :] = amp_bottom
        amp_noise_arr[i, :] = amp_noise

    # Frequency and time arrays
    f = np.fft.fftshift(np.fft.fftfreq(window_len, 1 / fs))
    f = f + fc
    f_mhz = f / 1e6
    t = 1 / fs * np.arange(len(rx_complex))

    # Store results in a dictionary
    radar_amp = {
        "amp_sfc": amp_sfc_arr,
        "amp_bottom": amp_bottom_arr,
        "amp_noise": amp_noise_arr,  # Noise amplitude spectrum
    }

    return radar_amp, f, f_mhz, t


def evaluate_trace_quality(
    radar_dict,
    radar_amp,
    rx_complex,
    f_mhz,
    slope_fit_start,
    slope_fit_end,
    snr_threshold,
    prom_threshold,
    snr_mode="low",
):
    """
    Mark bad traces based on selected SNR band and peak prominence.

    Parameters
    ----------
    snr_mode : str, optional
        Which SNR band to use when marking bad traces.
        Options:
            "low"  -> use low frequency quartile SNR
            "high" -> use high frequency quartile SNR

    Returns
    -------
    low_freq_snr : np.ndarray
        SNR in dB for the low frequency quartile
    high_freq_snr : np.ndarray
        SNR in dB for the high frequency quartile
    prom_arr : np.ndarray
        Peak prominence values for bed picks
    """
    import numpy as np
    import warnings
    from scipy.signal import peak_prominences

    if snr_mode not in ["low", "high"]:
        raise ValueError("snr_mode must be either 'low' or 'high'")

    n_traces = len(radar_amp["amp_bottom"])
    low_freq_snr = np.full(n_traces, np.nan)
    high_freq_snr = np.full(n_traces, np.nan)
    prom_arr = np.full(n_traces, np.nan)

    if "bad_trace" not in radar_dict:
        raise KeyError("radar_dict must include a 'bad_trace' key initialized to a boolean array")

    tidx_bottom = np.asarray(radar_dict["tidx_bottom"]).astype(int)

    # Mark invalid bottom picks as bad before entering the loop
    invalid_bottom = (tidx_bottom < 0) | (tidx_bottom >= rx_complex.shape[0])
    radar_dict.loc[invalid_bottom, "bad_trace"] = True

    freq_range_mask = (f_mhz >= slope_fit_start) & (f_mhz <= slope_fit_end)
    freq_range_indices = np.where(freq_range_mask)[0]

    if len(freq_range_indices) == 0:
        raise ValueError("No frequency indices found in specified range")

    quartile_index = int(len(freq_range_indices) * 0.25)
    if quartile_index == 0:
        raise ValueError("Frequency range is too small to define quartiles")

    sorted_indices = np.argsort(f_mhz[freq_range_indices])
    lowest_freq_indices = freq_range_indices[sorted_indices[:quartile_index]]
    highest_freq_indices = freq_range_indices[sorted_indices[-quartile_index:]]

    for idx in range(n_traces):
        # Skip traces already marked bad
        if radar_dict.loc[idx, "bad_trace"]:
            continue

        bed_idx = tidx_bottom[idx]

        # Extra safety check
        if bed_idx < 0 or bed_idx >= rx_complex.shape[0]:
            radar_dict.loc[idx, "bad_trace"] = True
            continue

        amp_bottom = radar_amp["amp_bottom"][idx]
        amp_noise = radar_amp["amp_noise"][idx]

        power_bottom = amp_bottom ** 2
        power_noise = amp_noise ** 2

        mean_power_bottom_low = np.mean(power_bottom[lowest_freq_indices])
        mean_power_noise_low = np.mean(power_noise[lowest_freq_indices])
        mean_power_bottom_high = np.mean(power_bottom[highest_freq_indices])
        mean_power_noise_high = np.mean(power_noise[highest_freq_indices])

        snr_low = np.nan
        snr_high = np.nan

        if mean_power_noise_low > 0:
            snr_low = 10 * np.log10(mean_power_bottom_low / mean_power_noise_low)
            low_freq_snr[idx] = snr_low

        if mean_power_noise_high > 0:
            snr_high = 10 * np.log10(mean_power_bottom_high / mean_power_noise_high)
            high_freq_snr[idx] = snr_high

        # Apply selected SNR filter
        if snr_mode == "low":
            if np.isfinite(snr_low) and snr_low < snr_threshold:
                radar_dict.loc[idx, "bad_trace"] = True
                continue
        elif snr_mode == "high":
            if np.isfinite(snr_high) and snr_high < snr_threshold:
                radar_dict.loc[idx, "bad_trace"] = True
                continue

        rx_values = 10 * np.log10(np.clip(np.abs(rx_complex[:, idx]), 1e-12, None))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prom, _, _ = peak_prominences(rx_values, [bed_idx])

        if len(prom) > 0 and prom[0] > 0:
            prom_arr[idx] = prom[0]
            if prom[0] < prom_threshold:
                radar_dict.loc[idx, "bad_trace"] = True

    low_freq_min = np.min(f_mhz[lowest_freq_indices])
    low_freq_max = np.max(f_mhz[lowest_freq_indices])
    high_freq_min = np.min(f_mhz[highest_freq_indices])
    high_freq_max = np.max(f_mhz[highest_freq_indices])

    if snr_mode == "low":
        bad_from_snr = np.sum(low_freq_snr < snr_threshold)
        print(f"Using low frequency SNR filter")
        print(f"Low frequency quartile: {low_freq_min:.2f} to {low_freq_max:.2f} MHz")
    else:
        bad_from_snr = np.sum(high_freq_snr < snr_threshold)
        print(f"Using high frequency SNR filter")
        print(f"High frequency quartile: {high_freq_min:.2f} to {high_freq_max:.2f} MHz")

    bad_total = np.sum(radar_dict["bad_trace"])
    good_total = np.sum(~radar_dict["bad_trace"])
    n_total = len(radar_dict)

    pct_from_snr = 100 * bad_from_snr / n_total
    pct_bad_total = 100 * bad_total / n_total

    print(f"Bad traces added from selected SNR filter: {bad_from_snr} ({pct_from_snr:.1f}% of total)")
    print(f"Total bad traces, including prominence filter: {bad_total} ({pct_bad_total:.1f}% of total)")
    print(f"Total good traces: {good_total}")

    return low_freq_snr, high_freq_snr, prom_arr

def compute_all_slopes_per_trace(
    radar_amp, f_mhz, delta_tau, f_min, f_max, window_size, bad_traces,
    outlier_substitution=True, circular=False
):
    """
    Compute slope and RMSE for surface, bed, and log spectral ratio, with optional outlier substitution.

    Parameters
    ----------
    radar_amp : dict
        Dictionary with 'amp_sfc' and 'amp_bottom', each (n_traces, n_freq).
    f_mhz : np.ndarray
        Frequency array in MHz.
    delta_tau : np.ndarray
        Delay time array (in seconds), one per trace.
    f_min, f_max : float
        Frequency range for fitting (in MHz).
    window_size : int
        Number of traces to average over for smoothing.
    bad_traces : np.ndarray
        Binary array indicating bad traces (1 for bad, 0 for good).
    outlier_substitution : bool
        If True, performs slope substitution and recalculates spectral slope for surface/bed outliers.
    circular : bool
        If True, use circular smoothing at edges.

    Returns
    -------
    slope_sfc_arr, slope_bottom_arr : Surface and bed slopes
    rmse_sfc_arr, rmse_bottom_arr : RMSE for surface and bed slope fits
    slope_arr : Slope of log(amp_bed/amp_sfc)
    delta_tau_smooth_arr : Smoothed delay time per trace
    rmse_arr : RMSE of spectral ratio slope
    center_idx_arr : Index for each center trace
    good_pct_arr : Percent of good traces in each window
    std_err_arr : Standard error of spectral ratio slope
    """

    half_win = window_size // 2
    n_traces = radar_amp["amp_sfc"].shape[0]
    f_mask = (f_mhz >= f_min) & (f_mhz <= f_max)
    x_fit = f_mhz[f_mask]

    # Output arrays
    slope_sfc_arr = np.full(n_traces, np.nan)
    slope_bottom_arr = np.full(n_traces, np.nan)
    rmse_sfc_arr = np.full(n_traces, np.nan)
    rmse_bottom_arr = np.full(n_traces, np.nan)

    slope_arr = np.full(n_traces, np.nan)
    delta_tau_smooth_arr = np.full(n_traces, np.nan)
    rmse_arr = np.full(n_traces, np.nan)
    center_idx_arr = np.full(n_traces, np.nan)
    good_pct_arr = np.full(n_traces, np.nan)
    std_err_arr = np.full(n_traces, np.nan)

    valid_idxs = np.where(bad_traces == 0)[0]
    for i in tqdm(valid_idxs, desc="Computing Slopes"):
        if circular:
            idxs = [(i + j) % n_traces for j in range(-half_win, half_win + 1)]
        else:
            start = max(0, i - half_win)
            end = min(n_traces, i + half_win + 1)
            idxs = list(range(start, end))

        amp_sfc = np.nanmean(radar_amp["amp_sfc"][idxs, :], axis=0)
        amp_bottom = np.nanmean(radar_amp["amp_bottom"][idxs, :], axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            ln_sfc = np.log(amp_sfc)
            ln_bottom = np.log(amp_bottom)
            amp_ratio = amp_bottom / amp_sfc
            ln_amp_ratio = np.log(amp_ratio)

        y_sfc = ln_sfc[f_mask]
        y_bottom = ln_bottom[f_mask]
        y_ratio = ln_amp_ratio[f_mask]

        # Surface slope
        if np.all(np.isfinite(y_sfc)) and len(y_sfc) == len(x_fit):
            slope_sfc, intercept_sfc, *_ = linregress(x_fit, y_sfc)
            y_pred_sfc = slope_sfc * x_fit + intercept_sfc
            rmse_sfc = np.sqrt(np.mean((y_sfc - y_pred_sfc) ** 2))
            slope_sfc_arr[i] = slope_sfc
            rmse_sfc_arr[i] = rmse_sfc

        # Bed slope
        if np.all(np.isfinite(y_bottom)) and len(y_bottom) == len(x_fit):
            slope_bottom, intercept_bottom, *_ = linregress(x_fit, y_bottom)
            y_pred_bottom = slope_bottom * x_fit + intercept_bottom
            rmse_bottom = np.sqrt(np.mean((y_bottom - y_pred_bottom) ** 2))
            slope_bottom_arr[i] = slope_bottom
            rmse_bottom_arr[i] = rmse_bottom

        # Spectral ratio slope
        if np.all(np.isfinite(y_ratio)) and len(y_ratio) == len(x_fit):
            slope, intercept, _, _, std_err = linregress(x_fit, y_ratio)
            y_pred = slope * x_fit + intercept
            rmse = np.sqrt(np.mean((y_ratio - y_pred) ** 2))
            slope_arr[i] = slope
            rmse_arr[i] = rmse
            std_err_arr[i] = std_err

        delta_tau_smooth_arr[i] = np.nanmean(delta_tau[idxs])
        good_pct_arr[i] = 100 * np.mean(bad_traces[idxs] == 0)
        center_idx_arr[i] = i

    # Outlier substitution (second pass only for outliers)
    if outlier_substitution:
        sfc_clean = slope_sfc_arr[np.isfinite(slope_sfc_arr)]
        bottom_clean = slope_bottom_arr[np.isfinite(slope_bottom_arr)]
        mean_sfc, std_sfc = np.nanmean(sfc_clean), np.nanstd(sfc_clean)
        mean_bottom, std_bottom = np.nanmean(bottom_clean), np.nanstd(bottom_clean)

        sfc_outliers = np.abs(slope_sfc_arr - mean_sfc) > 2 * std_sfc
        bottom_outliers = np.abs(slope_bottom_arr - mean_bottom) > 2 * std_bottom
        any_outliers = sfc_outliers | bottom_outliers
        outlier_idxs = np.where(any_outliers & (bad_traces == 0))[0]

        for i in tqdm(outlier_idxs, desc="Outlier Substitution"):
            if circular:
                idxs = [(i + j) % n_traces for j in range(-half_win, half_win + 1)]
            else:
                start = max(0, i - half_win)
                end = min(n_traces, i + half_win + 1)
                idxs = list(range(start, end))

            amp_sfc = np.nanmean(radar_amp["amp_sfc"][idxs, :], axis=0)
            amp_bottom = np.nanmean(radar_amp["amp_bottom"][idxs, :], axis=0)

            if sfc_outliers[i]:
                amp_sfc = np.exp(mean_sfc * f_mhz)
            if bottom_outliers[i]:
                amp_bottom = np.exp(mean_bottom * f_mhz)

            with np.errstate(divide="ignore", invalid="ignore"):
                amp_ratio = amp_bottom / amp_sfc
                ln_amp_ratio = np.log(amp_ratio)

            y_fit = ln_amp_ratio[f_mask]
            if np.all(np.isfinite(y_fit)) and len(y_fit) == len(x_fit):
                slope, intercept, _, _, std_err = linregress(x_fit, y_fit)
                y_pred = slope * x_fit + intercept
                rmse = np.sqrt(np.mean((y_fit - y_pred) ** 2))

                slope_arr[i] = slope
                rmse_arr[i] = rmse
                std_err_arr[i] = std_err

    return (
        slope_sfc_arr,
        slope_bottom_arr,
        rmse_sfc_arr,
        rmse_bottom_arr,
        slope_arr,
        delta_tau_smooth_arr,
        rmse_arr,
        center_idx_arr,
        good_pct_arr,
        std_err_arr,
    )


def compute_slope_per_trace(
    radar_amp, f_mhz, delta_tau, f_min, f_max, window_size, bad_traces,
    circular=False,
):
    """
    Compute the spectral ratio slope (ln(bed / sfc) vs frequency) for each trace,
    using three different per-trace aggregation pipelines:

      * ``average`` – average raw sfc and bed spectra across the smoothing window,
        then fit the slope of `ln(mean_bed / mean_sfc)`.
      * ``normalize_average`` – divide each trace by its own raw peak (so every
        trace contributes equally), average the normalized spectra, then fit.
      * ``subtract_normalize_average`` – subtract the per-trace noise spectrum,
        divide each trace by the peak of its noise-corrected spectrum, average,
        then fit.

    Parameters
    ----------
    radar_amp : dict
        Dictionary with 'amp_sfc', 'amp_bottom', and 'amp_noise', each shaped
        (n_traces, n_freq).
    f_mhz : np.ndarray
        Frequency array in MHz.
    delta_tau : np.ndarray
        Two-way travel time difference (bottom - surface), one value per trace (s).
    f_min, f_max : float
        Frequency band for the linear fit (MHz).
    window_size : int
        Number of neighboring traces averaged for each center trace (odd integer;
        1 = no smoothing).
    bad_traces : np.ndarray
        Binary array indicating bad traces (1 for bad, 0 for good).
    circular : bool, optional
        If True, wrap neighbor indices at segment edges.

    Returns
    -------
    results : dict
        ``{
            "pipelines": {
                "average":                    {"slope": ..., "rmse": ..., "std_err": ...},
                "normalize_average":          {"slope": ..., "rmse": ..., "std_err": ...},
                "subtract_normalize_average": {"slope": ..., "rmse": ..., "std_err": ...},
            },
            "delta_tau_smooth": ..., "center_idx": ...,
            "good_pct": ...,         "snr_bed":    ...,
        }``
        Pipeline-specific slope / RMSE / slope standard error are nested under
        ``pipelines[name]``; quantities that do not depend on the pipeline
        (smoothed delta_tau, center index, window good-pct, bed-over-noise SNR)
        live at the top level. Use :func:`select_pipeline` to unpack a pipeline
        into flat arrays for plotting.
    """

    half_win = window_size // 2
    n_traces = radar_amp["amp_sfc"].shape[0]
    f_mask = (f_mhz >= f_min) & (f_mhz <= f_max)
    x_fit = f_mhz[f_mask]

    amp_sfc_all    = np.asarray(radar_amp["amp_sfc"],    dtype=float)
    amp_bottom_all = np.asarray(radar_amp["amp_bottom"], dtype=float)
    amp_noise_all  = np.asarray(radar_amp["amp_noise"],  dtype=float)

    pipeline_names = ("average", "normalize_average", "subtract_normalize_average")
    pipelines = {
        name: {
            "slope":   np.full(n_traces, np.nan),
            "rmse":    np.full(n_traces, np.nan),
            "std_err": np.full(n_traces, np.nan),
        }
        for name in pipeline_names
    }

    delta_tau_smooth_arr = np.full(n_traces, np.nan)
    center_idx_arr       = np.full(n_traces, np.nan)
    good_pct_arr         = np.full(n_traces, np.nan)
    snr_bed_arr          = np.full(n_traces, np.nan)

    def _safe_peak(x):
        p = np.nanmax(x, axis=1, keepdims=True)
        return np.where(p > 0, p, np.nan)

    def _fit_slope(amp_sfc_mean, amp_bottom_mean):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(amp_bottom_mean / amp_sfc_mean)
        y_fit = y[f_mask]
        valid = np.isfinite(y_fit)
        if valid.sum() < 2:
            return np.nan, np.nan, np.nan
        slope, intercept, _, _, std_err = linregress(x_fit[valid], y_fit[valid])
        y_pred = slope * x_fit[valid] + intercept
        rmse = np.sqrt(np.mean((y_fit[valid] - y_pred) ** 2))
        return slope, rmse, std_err

    valid_idxs = np.where(bad_traces == 0)[0]
    for i in tqdm(valid_idxs, desc="Computing Slopes"):
        if circular:
            idxs = [(i + j) % n_traces for j in range(-half_win, half_win + 1)]
        else:
            start = max(0, i - half_win)
            end = min(n_traces, i + half_win + 1)
            idxs = list(range(start, end))

        sfc_win   = amp_sfc_all[idxs, :]
        bed_win   = amp_bottom_all[idxs, :]
        noise_win = amp_noise_all[idxs, :]

        sfc_corr_win = sfc_win - noise_win
        bed_corr_win = bed_win - noise_win

        # 1) average raw spectra
        amp_sfc_avg    = np.nanmean(sfc_win, axis=0)
        amp_bottom_avg = np.nanmean(bed_win, axis=0)

        # 2) normalize per-trace (by raw peak) then average
        sfc_peak_raw = _safe_peak(sfc_win)
        bed_peak_raw = _safe_peak(bed_win)
        amp_sfc_norm_avg    = np.nanmean(sfc_win / sfc_peak_raw, axis=0)
        amp_bottom_norm_avg = np.nanmean(bed_win / bed_peak_raw, axis=0)

        # 3) subtract noise, normalize per-trace (by noise-subtracted peak), then average
        sfc_peak_corr = _safe_peak(sfc_corr_win)
        bed_peak_corr = _safe_peak(bed_corr_win)
        amp_sfc_sub_norm_avg    = np.nanmean(sfc_corr_win / sfc_peak_corr, axis=0)
        amp_bottom_sub_norm_avg = np.nanmean(bed_corr_win / bed_peak_corr, axis=0)

        for name, (s_mean, b_mean) in zip(
            pipeline_names,
            (
                (amp_sfc_avg,          amp_bottom_avg),
                (amp_sfc_norm_avg,     amp_bottom_norm_avg),
                (amp_sfc_sub_norm_avg, amp_bottom_sub_norm_avg),
            ),
        ):
            slope, rmse, std_err = _fit_slope(s_mean, b_mean)
            pipelines[name]["slope"][i]   = slope
            pipelines[name]["rmse"][i]    = rmse
            pipelines[name]["std_err"][i] = std_err

        delta_tau_smooth_arr[i] = np.nanmean(delta_tau[idxs])
        good_pct_arr[i] = 100 * np.mean(bad_traces[idxs] == 0)
        center_idx_arr[i] = i

        # Report-only bed SNR over the fit band (mean-power ratio, in dB),
        # from the raw window-averaged spectra (pipeline-independent).
        bed_band   = amp_bottom_avg[f_mask]
        noise_band = np.nanmean(noise_win, axis=0)[f_mask]
        if np.all(np.isfinite(bed_band) & np.isfinite(noise_band)):
            mean_p_bed = np.mean(bed_band ** 2)
            mean_p_noise = np.mean(noise_band ** 2)
            if mean_p_noise > 0:
                snr_bed_arr[i] = 10.0 * np.log10(mean_p_bed / mean_p_noise)

    return {
        "pipelines":        pipelines,
        "delta_tau_smooth": delta_tau_smooth_arr,
        "center_idx":       center_idx_arr,
        "good_pct":         good_pct_arr,
        "snr_bed":          snr_bed_arr,
    }


def select_pipeline(results, name):
    """
    Unpack one pipeline's outputs from :func:`compute_slope_per_trace` into the
    flat tuple ``(slope, delta_tau_smooth, rmse, center_idx, good_pct, std_err,
    snr_bed)`` used by the along-track / attenuation plots.
    """
    if name not in results["pipelines"]:
        raise KeyError(
            f"unknown pipeline {name!r}; options are {tuple(results['pipelines'])}"
        )
    p = results["pipelines"][name]
    return (
        p["slope"],
        results["delta_tau_smooth"],
        p["rmse"],
        results["center_idx"],
        results["good_pct"],
        p["std_err"],
        results["snr_bed"],
    )


def compute_slopes(
    radar_amp, f_mhz, f_min, f_max, bad_traces,
):
    """
    Compute spectral ratio slopes for each trace using two approaches:

    1. Per-trace ratio: ln(amp_bottom_i / amp_sfc_i)
    2. Mean-surface ratio: ln(amp_bottom_i / amp_sfc_mean)

    where amp_sfc_mean is the mean surface amplitude across all good traces.

    Parameters
    ----------
    radar_amp : dict
        Dictionary with 'amp_sfc' and 'amp_bottom', each (n_traces, n_freq).
    f_mhz : np.ndarray
        Frequency array in MHz.
    f_min, f_max : float
        Frequency range for fitting (in MHz).
    bad_traces : np.ndarray
        Binary array indicating bad traces (1 for bad, 0 for good).

    Returns
    -------
    slope_arr : np.ndarray
        Slope of ln(amp_bottom_i / amp_sfc_i) per trace.
    rmse_arr : np.ndarray
        RMSE of per-trace ratio slope fit.
    std_err_arr : np.ndarray
        Standard error of per-trace ratio slope.
    slope_mean_sfc_arr : np.ndarray
        Slope of ln(amp_bottom_i / amp_sfc_mean) per trace.
    rmse_mean_sfc_arr : np.ndarray
        RMSE of mean-surface ratio slope fit.
    std_err_mean_sfc_arr : np.ndarray
        Standard error of mean-surface ratio slope.
    """

    n_traces = radar_amp["amp_sfc"].shape[0]
    f_mask = (f_mhz >= f_min) & (f_mhz <= f_max)
    x_fit = f_mhz[f_mask]

    valid_idxs = np.where(bad_traces == 0)[0]

    # Mean surface amplitude across all good traces
    amp_sfc_mean = np.nanmean(radar_amp["amp_sfc"][valid_idxs, :], axis=0)

    # Per-trace ratio output arrays
    slope_arr = np.full(n_traces, np.nan)
    rmse_arr = np.full(n_traces, np.nan)
    std_err_arr = np.full(n_traces, np.nan)

    # Mean-surface ratio output arrays
    slope_mean_sfc_arr = np.full(n_traces, np.nan)
    rmse_mean_sfc_arr = np.full(n_traces, np.nan)
    std_err_mean_sfc_arr = np.full(n_traces, np.nan)

    for i in tqdm(valid_idxs, desc="Computing Slopes"):
        amp_sfc_i = radar_amp["amp_sfc"][i, :]
        amp_bottom_i = radar_amp["amp_bottom"][i, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            # Per-trace ratio
            ln_ratio = np.log(amp_bottom_i / amp_sfc_i)
            # Mean-surface ratio
            ln_ratio_mean_sfc = np.log(amp_bottom_i / amp_sfc_mean)

        # Per-trace ratio fit
        y_ratio = ln_ratio[f_mask]
        if np.all(np.isfinite(y_ratio)) and len(y_ratio) == len(x_fit):
            slope, intercept, _, _, std_err = linregress(x_fit, y_ratio)
            y_pred = slope * x_fit + intercept
            rmse = np.sqrt(np.mean((y_ratio - y_pred) ** 2))
            slope_arr[i] = slope
            rmse_arr[i] = rmse
            std_err_arr[i] = std_err

        # Mean-surface ratio fit
        y_ratio_ms = ln_ratio_mean_sfc[f_mask]
        if np.all(np.isfinite(y_ratio_ms)) and len(y_ratio_ms) == len(x_fit):
            slope_ms, intercept_ms, _, _, std_err_ms = linregress(x_fit, y_ratio_ms)
            y_pred_ms = slope_ms * x_fit + intercept_ms
            rmse_ms = np.sqrt(np.mean((y_ratio_ms - y_pred_ms) ** 2))
            slope_mean_sfc_arr[i] = slope_ms
            rmse_mean_sfc_arr[i] = rmse_ms
            std_err_mean_sfc_arr[i] = std_err_ms

    return (
        slope_arr,
        rmse_arr,
        std_err_arr,
        slope_mean_sfc_arr,
        rmse_mean_sfc_arr,
        std_err_mean_sfc_arr,
    )


def calculate_attenuation(
    slope_arr, std_err_arr, delta_tau_smooth_arr, fc, epsilon_r=3.15,
):
    """
    Convert spectral ratio slope to one-way attenuation rate Na (dB/km) and
    propagate the slope standard error into an uncertainty Na_unc (dB/km).

    No filtering or outlier rejection — just the math. Na is linear in slope
    (for fixed delta_tau), so propagation reduces to substituting std_err_arr
    for slope_arr in the same formula and taking the magnitude.

    Parameters
    ----------
    slope_arr : np.ndarray
        Spectral ratio slopes in per-MHz (slope of ln(amp_bottom / amp_sfc)).
    std_err_arr : np.ndarray
        Standard error on each slope (same units as slope_arr).
    delta_tau_smooth_arr : np.ndarray
        Smoothed two-way travel time difference between bed and surface (s).
    fc : float
        Center frequency (Hz).
    epsilon_r : float, optional
        Relative permittivity of ice (default 3.15).

    Returns
    -------
    Na_arr : np.ndarray
        Attenuation rate in dB/km.
    Na_unc_arr : np.ndarray
        1-sigma uncertainty in Na (dB/km), propagated from std_err_arr.
    """

    c_light = 2.998e8
    vp = c_light / np.sqrt(epsilon_r)

    with np.errstate(divide="ignore", invalid="ignore"):
        Q = delta_tau_smooth_arr * np.pi / slope_arr
        alpha = np.pi * fc / (vp * Q) / 1e6  # slope_arr is per-MHz; 1/m -> 1/km
        Na_arr = -8.686 * alpha * 1000       # dB/km

        Q_unc = delta_tau_smooth_arr * np.pi / std_err_arr
        alpha_unc = np.pi * fc / (vp * Q_unc) / 1e6
        Na_unc_arr = np.abs(-8.686 * alpha_unc * 1000)

    return Na_arr, Na_unc_arr


def save_frequency_attenuation_results(
    radar_dict,
    center_idx_arr,
    slope_arr,
    rmse_arr,
    std_err_arr,
    good_pct_arr,
    Na_err_filtered,
    Na_filtered,
    radar_atten_dir,
    seg_name,
    ice_sheet,  # must be 'antarctica' or 'greenland'
    slope_fit_start,
    slope_fit_end,
    window_size,
    fft_win,
):
    """
    Save frequency attenuation results along with radar trace attributes to a CSV file.

    Parameters
    ----------
    radar_dict : pandas.DataFrame
        DataFrame containing radar trace attributes (e.g., lat, lon, etc.).
    center_idx_arr : np.ndarray
        Center trace indices for each slope calculation window.
    slope_arr : np.ndarray
        Spectral slope values.
    rmse_arr : np.ndarray
        RMSE of slope fits.
    good_pct_arr : np.ndarray
        Percentage of good traces in each averaging window.
    Na_arr : np.ndarray
        Derived attenuation rates (dB/km).
    radar_atten_dir : str
        Output directory.
    seg_name : str
        Segment name for output file labeling.
    ice_sheet : str
        Either "antarctica" or "greenland", used to determine projection for x/y.
    """

    # Choose projection based on ice sheet
    if ice_sheet.lower() == "antarctica":
        polar_crs = pyproj.CRS("EPSG:3031")  # Antarctic Polar Stereographic
    elif ice_sheet.lower() == "greenland":
        polar_crs = pyproj.CRS("EPSG:3413")  # Greenland Polar Stereographic
    else:
        raise ValueError("ice_sheet must be either 'antarctica' or 'greenland'")

    wgs84 = pyproj.CRS("EPSG:4326")  # Lat/Lon
    transformer = pyproj.Transformer.from_crs(wgs84, polar_crs, always_xy=True)

    # Add x/y projection if missing
    if "x" not in radar_dict.columns or "y" not in radar_dict.columns:
        x, y = transformer.transform(radar_dict["lon"].values, radar_dict["lat"].values)
        radar_dict["x"] = x
        radar_dict["y"] = y

    # Add derived arrays
    radar_dict["center_idx"] = center_idx_arr
    radar_dict["slope"] = slope_arr
    radar_dict["rmse"] = rmse_arr
    radar_dict["std_err"] = std_err_arr
    radar_dict["good_pct"] = good_pct_arr
    radar_dict["Na_err"] = Na_err_filtered
    radar_dict["Na"] = Na_filtered

    # Save results
    os.makedirs(radar_atten_dir, exist_ok=True)
    output_path = os.path.join(
        radar_atten_dir,
        f"atten_{seg_name}_{slope_fit_start}_{slope_fit_end}_MHz_{fft_win}_fft_{window_size}_smooth.csv",
    )
    radar_dict.to_csv(output_path, index=False, na_rep="NaN")

    print(f"Results saved to: {output_path}")
