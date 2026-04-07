import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyleoclim.utils.wavelet import cwt_coherence
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import savgol_filter
from pyleoclim.utils.correlation import association
from scipy.stats import ttest_ind
from itertools import combinations
import neurokit2 as nk

condition_dict = {
    0: "static practice",
    1: "slow practice",
    2: "fast practice",
    3: "independant",
    4: "follower",
    5: "evolved",
    6: "human-human",
}


# ============================================================
# ---------------------- DATA LAYER --------------------------
# ===========================================================
def find_agent_select_times(df):
    arr_ends = df["time"][df["agentSel:1"].diff(periods=-1) != 0].values.reshape(-1, 1)
    arr_starts = df["time"][df["agentSel:1"].diff(periods=1) != 0].values.reshape(-1, 1)
    return np.hstack((arr_starts, arr_ends))


# def divide_by_trials(df, conditions=None, time_col="time", copy=True):
#     """
#     Automatically:
#     1. calls find_agent_select_times(df)
#     2. maps condition indices to readable trial names
#     3. splits df into one dataframe per trial

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Full dataframe containing a time column.
#     condition_dict : dict or None
#         Maps trial index -> original condition label.
#         If None, uses the default mapping.
#     time_col : str
#         Name of the time column in df.
#     copy : bool
#         If True, return copies of slices.

#     Returns
#     -------
#     trial_dfs : dict
#         {clean_trial_name: sliced_dataframe}
#     condition_ranges : dict
#         {clean_trial_name: (start, end)}
#     raw_trial_times : dict or list
#         Raw output from find_agent_select_times(df)
#     """

#     if conditions is None:
#         global condition_dict
#         conditions = condition_dict
#     # Rename labels to the exact names you want in the final output
#     name_map = {
#         "Practice - Fixed": "static practice",
#         "Practice - Slow": "slow practice",
#         "Practice - Fast": "fast practice",
#         "Condition - Independent": "independant",
#         "Condition - Follower": "follower",
#         "Condition - Crosser": "evolved",
#         "Condition - Human": "human-human",
#     }

#     # ---------------------------------------------------
#     # 1) Get trial times automatically
#     # ---------------------------------------------------
#     raw_trial_times = find_agent_select_times(df)

#     # ---------------------------------------------------
#     # 2) Normalize raw_trial_times into index -> (start, end)
#     #    Supports either:
#     #    - dict: {0: (start, end), 1: (start, end), ...}
#     #    - list: [(start, end), (start, end), ...]
#     # ---------------------------------------------------
#     if isinstance(raw_trial_times, dict):
#         indexed_times = raw_trial_times
#     else:
#         indexed_times = {i: t for i, t in enumerate(raw_trial_times)}

#     # ---------------------------------------------------
#     # 3) Build cleaned condition -> (start, end)
#     # ---------------------------------------------------
#     condition_ranges = {}

#     for idx, original_label in conditions.items():
#         if idx not in indexed_times:
#             raise ValueError(
#                 f"find_agent_select_times(df) did not return a time range for index {idx}"
#             )

#         start, end = indexed_times[idx]
#         clean_label = name_map.get(original_label, original_label)
#         condition_ranges[clean_label] = (start, end)

#     # ---------------------------------------------------
#     # 4) Split dataframe
#     # ---------------------------------------------------
#     trial_dfs = {}

#     for label, (start, end) in condition_ranges.items():
#         mask = (df[time_col] >= start) & (df[time_col] <= end)
#         subdf = df.loc[mask]

#         if copy:
#             subdf = subdf.copy()

#         trial_dfs[label] = subdf

#     return (
#         trial_dfs,
#         condition_ranges,
#     )  # raw_trial_times

import numpy as np
import pandas as pd


def trim_inactive_region(
    subdf,
    signal_col="p1",
    time_col="time",
    activity_thresh=1e-4,
    smooth_window=5,
    min_active_samples=5,
):
    """
    Trim leading and trailing inactivity from a trial based on p1 movement.

    Inactivity is determined from the absolute first difference of `signal_col`.
    The function keeps the region from the first sustained activity to the last
    sustained activity.

    Parameters
    ----------
    subdf : pandas.DataFrame
        One trial slice.
    signal_col : str
        Column used to detect movement/activity.
    time_col : str
        Time column name.
    activity_thresh : float
        Threshold on smoothed absolute difference to count as active.
    smooth_window : int
        Rolling window size for smoothing abs(diff).
    min_active_samples : int
        Minimum consecutive active-ish samples required to define activity region.

    Returns
    -------
    trimmed_df : pandas.DataFrame
        Cropped dataframe.
    trimmed_range : tuple
        (new_start_time, new_end_time)
    """
    if subdf.empty:
        return subdf.copy(), (np.nan, np.nan)

    if signal_col not in subdf.columns:
        raise ValueError(f"Column '{signal_col}' not found in dataframe.")

    work = subdf.copy().sort_values(time_col).reset_index(drop=True)

    # movement estimate: absolute point-to-point change
    activity = work[signal_col].diff().abs()

    # first diff creates NaN at first row
    activity = activity.fillna(0)

    # smooth so tiny jitter does not dominate
    if smooth_window > 1:
        activity = (
            activity.rolling(window=smooth_window, center=True, min_periods=1).mean()
        )

    is_active = activity > activity_thresh

    # Find first sustained active region
    active_idx = np.flatnonzero(is_active.to_numpy())

    if len(active_idx) == 0:
        # no detected activity -> return original or empty
        return work.iloc[0:0].copy(), (np.nan, np.nan)

    # Optional robustness: require local runs of activity
    first_idx = active_idx[0]
    last_idx = active_idx[-1]

    if min_active_samples > 1:
        arr = is_active.to_numpy().astype(int)

        # forward search for first window with enough active samples
        first_idx = None
        for i in range(len(arr) - min_active_samples + 1):
            if arr[i : i + min_active_samples].sum() >= min_active_samples:
                first_idx = i
                break

        # backward search for last window with enough active samples
        last_idx = None
        for i in range(len(arr) - min_active_samples, -1, -1):
            if arr[i : i + min_active_samples].sum() >= min_active_samples:
                last_idx = i + min_active_samples - 1
                break

        if first_idx is None or last_idx is None or first_idx > last_idx:
            return work.iloc[0:0].copy(), (np.nan, np.nan)

    trimmed = work.iloc[first_idx : last_idx + 1].copy()
    new_start = trimmed[time_col].iloc[0]
    new_end = trimmed[time_col].iloc[-1]

    return trimmed, (new_start, new_end)


def divide_by_trials(
    df,
    conditions=None,
    time_col="time",
    signal_col="angle_p1",
    copy=True,
    trim_inactive=True,
    activity_thresh=1e-4,
    smooth_window=5,
    min_active_samples=5,
):
    """
    Automatically:
    1. calls find_agent_select_times(df)
    2. maps condition indices to readable trial names
    3. splits df into one dataframe per trial
    4. optionally trims inactivity from the beginning/end of each trial
       based on movement in `signal_col`

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataframe containing a time column.
    conditions : dict or None
        Maps trial index -> original condition label.
        If None, uses the default mapping.
    time_col : str
        Name of the time column in df.
    signal_col : str
        Signal column used to detect inactivity trimming, e.g. "p1".
    copy : bool
        If True, return copies of slices.
    trim_inactive : bool
        If True, remove leading/trailing inactive sections from each trial.
    activity_thresh : float
        Threshold on smoothed abs(diff(signal)) to count as activity.
    smooth_window : int
        Rolling smoothing window for activity detection.
    min_active_samples : int
        Minimum consecutive active samples for activity detection.

    Returns
    -------
    trial_dfs : dict
        {clean_trial_name: sliced_dataframe}
    condition_ranges : dict
        {clean_trial_name: (start, end)}
    """
    if conditions is None:
        global condition_dict
        conditions = condition_dict

    name_map = {
        "Practice - Fixed": "static practice",
        "Practice - Slow": "slow practice",
        "Practice - Fast": "fast practice",
        "Condition - Independent": "independant",
        "Condition - Follower": "follower",
        "Condition - Crosser": "evolved",
        "Condition - Human": "human-human",
    }

    # 1) Get rough trial times automatically
    raw_trial_times = find_agent_select_times(df)

    # 2) Normalize raw_trial_times into index -> (start, end)
    if isinstance(raw_trial_times, dict):
        indexed_times = raw_trial_times
    else:
        indexed_times = {i: t for i, t in enumerate(raw_trial_times)}

    # 3) Build cleaned condition -> (start, end)
    condition_ranges = {}
    for idx, original_label in conditions.items():
        if idx not in indexed_times:
            raise ValueError(
                f"find_agent_select_times(df) did not return a time range for index {idx}"
            )

        start, end = indexed_times[idx]
        clean_label = name_map.get(original_label, original_label)
        condition_ranges[clean_label] = (start, end)

    # 4) Split dataframe and optionally trim inactivity
    trial_dfs = {}
    trimmed_ranges = {}

    for label, (start, end) in condition_ranges.items():
        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subdf = df.loc[mask]

        if copy:
            subdf = subdf.copy()

        if trim_inactive:
            subdf, new_range = trim_inactive_region(
                subdf,
                signal_col=signal_col,
                time_col=time_col,
                activity_thresh=activity_thresh,
                smooth_window=smooth_window,
                min_active_samples=min_active_samples,
            )
            trimmed_ranges[label] = new_range
        else:
            trimmed_ranges[label] = (start, end)

        trial_dfs[label] = subdf

    return trial_dfs, trimmed_ranges

def load_physio_data(physio_dir, bunting_df):
    physio_data_df = pd.read_excel(physio_dir)

    physio_data_df["pulse_raw"] = physio_data_df["Serial Receive:1(1,1)"]
    physio_data_df["gsr_raw"] = physio_data_df["Serial Receive:1(2,1)"].apply(
        lambda x: (1e6) / (((1024.0 + 2.0 * x) * 10000.0) / (775.0 - x + 1e-6))
    )

    start_time1 = pd.to_datetime("24-Mar-2026 10:50:07")  # hardware start
    start_time2 = pd.to_datetime("24-Mar-2026 10:50:47")  # physio start

    data_df_copy = bunting_df.copy()
    physio_data_df_copy = physio_data_df.copy()

    # linear scaling
    physio_exec_time = 4195.451546541
    physio_sim_time = physio_data_df_copy["time"].iloc[-1]
    physio_data_df_copy["time"] = physio_data_df_copy["time"] * (
        physio_exec_time / physio_sim_time
    )

    # convert to absolute timestamps
    data_df_copy["time"] = pd.to_timedelta(data_df_copy["time"], unit="s") + start_time1
    physio_data_df_copy["time"] = (
        pd.to_timedelta(physio_data_df_copy["time"], unit="s") + start_time2
    )

    # VERY IMPORTANT: sort both before merge_asof
    data_df_copy = data_df_copy.sort_values("time").reset_index(drop=True)
    physio_data_df_copy = physio_data_df_copy.sort_values("time").reset_index(drop=True)

    print(
        "Bunting time range:",
        data_df_copy["time"].min(),
        "to",
        data_df_copy["time"].max(),
    )
    print(
        "Physio  time range:",
        physio_data_df_copy["time"].min(),
        "to",
        physio_data_df_copy["time"].max(),
    )
    # temporarily use a looser tolerance to debug
    data_df = pd.merge_asof(
        data_df_copy,
        physio_data_df_copy,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("10ms"),  # originally 10
    )

    data_df["time"] = (data_df["time"] - start_time1).dt.total_seconds()

    physio_cols = [
        "Serial Receive:1(1,1)",
        "Serial Receive:1(2,1)",
        "pulse_raw",
        "gsr_raw",
    ]
    print(data_df[physio_cols].isna().mean())

    return data_df


def load_and_clean_data(buntpath, physiopath, time_col="time"):
    df_bunt = pd.read_excel(buntpath)
    # rename new Excel column names to your old internal names
    df_bunt = df_bunt.rename(
        columns={
            "p1": "angle_p1",
            "x": "angle_p2",
        }
    )

    # check required columns exist
    required = ["time", "angle_p1", "angle_p2"]
    missing = [c for c in required if c not in df_bunt.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\nAvailable columns: {list(df_bunt.columns)}"
        )

    df_bunt = df_bunt.sort_values(time_col)
    df_bunt = df_bunt.groupby(time_col, as_index=False).mean()
    df_combined = load_physio_data(physiopath, df_bunt)
    # df_combined = pd.concat([df_bunt, df_physio], axis=1)
    print(
        "checking physio data:",
        df_combined[df_combined["pulse_raw"].notna()].head(1)[
            ["time", "pulse_raw", "gsr_raw"]
        ],
    )
    return df_combined


def unwrap_angles(df, col1="angle_p1", col2="angle_p2"):
    df["p1_unwrapped"] = np.unwrap(df[col1].to_numpy() - np.pi)
    df["p2_unwrapped"] = np.unwrap(df[col2].to_numpy() - np.pi)
    return df


def resample_signals(
    df,
    dt_target=0.001,
    angle_cols=("p1_unwrapped", "p2_unwrapped"),
    current_cols=("i1", "i2"),
    physio_cols=("pulse_raw", "gsr_raw"),
    time_col="time",
):
    z = df[time_col].to_numpy()
    t_new = np.arange(z.min(), z.max() + dt_target / 2, dt_target)

    out = {time_col: t_new}

    # PCHIP for angles
    for c in angle_cols:
        out[c] = PchipInterpolator(z, df[c].to_numpy())(t_new)

    # Linear interpolation for currents
    for c in current_cols:
        out[c] = interp1d(
            z,
            df[c].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(t_new)

        # Linear interpolation for physio signals
    for c in physio_cols:
        out[c] = interp1d(
            z,
            df[c].to_numpy(),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )(t_new)

    return pd.DataFrame(out)


def divide_data(df, start, end):
    mask = (df["time"] >= start) & (df["time"] <= end)
    return df[mask]


# ============================================================
# -------------------- DYNAMICS LAYER ------------------------
# ============================================================


def compute_dynamics(df_rs, dt, torque_constant=0.011, smooth=True):
    theta1 = df_rs["p1_unwrapped"].to_numpy()
    theta2 = df_rs["p2_unwrapped"].to_numpy()

    omega1 = np.gradient(theta1, dt)
    omega2 = np.gradient(theta2, dt)

    if smooth:
        omega1 = savgol_filter(omega1, 11, 3)
        omega2 = savgol_filter(omega2, 11, 3)

    df_rs["omega1"] = omega1
    df_rs["omega2"] = omega2
    df_rs["torque1"] = df_rs["i1"] * torque_constant
    df_rs["torque2"] = df_rs["i2"] * torque_constant
    # heartrate = df_rs[pulse_raw]
    # Sampling interval and sampling rate
    # dt = np.diff(df_processed["time"]).mean()   # sec/sample
    fs = int(round(1 / dt))  # Hz, make sure it's an integer

    # ---------------- PPG ----------------
    ppg_raw = df_rs["pulse_raw"].to_numpy()

    ppg_signals, ppg_info = nk.ppg_process(ppg_raw, sampling_rate=fs)

    nk.ppg_plot(ppg_signals, ppg_info)
    fig = plt.gcf()
    fig.set_size_inches(10, 12, forward=True)
    # plt.show()

    print(nk.ppg_intervalrelated(ppg_signals, sampling_rate=fs))

    nk.hrv(ppg_signals["PPG_Peaks"], sampling_rate=fs, show=True)
    fig = plt.gcf()
    fig.set_size_inches(16, 6, forward=True)
    # plt.show()

    # Optional BPM output
    print("Mean BPM:", np.nanmean(ppg_signals["PPG_Rate"]))

    # ---------------- EDA ----------------
    eda_raw = df_rs["gsr_raw"].to_numpy()

    eda_signals, eda_info = nk.eda_process(eda_raw, sampling_rate=fs)

    nk.eda_plot(eda_signals, eda_info)
    fig = plt.gcf()
    fig.set_size_inches(10, 12, forward=True)
    # plt.show()

    eda_summary = nk.eda_intervalrelated(eda_signals, sampling_rate=fs)
    print(eda_summary)

    print("SCR count:", int(eda_signals["SCR_Peaks"].sum()))
    df_rs["bpm"] = ppg_signals["PPG_Rate"].values

    df_rs["eda_clean"] = eda_signals["EDA_Clean"].values
    df_rs["eda_tonic"] = eda_signals["EDA_Tonic"].values
    df_rs["eda_phasic"] = eda_signals["EDA_Phasic"].values
    df_rs["scr_peaks"] = eda_signals["SCR_Peaks"].values
    df_rs["scr_amplitude"] = eda_signals["SCR_Amplitude"].values

    return df_rs


# ============================================================
# ------------------ PHASE ANALYSIS LAYER --------------------
# ============================================================


def circular_stats(angles):
    angles = np.asarray(angles)
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))

    mean_angle = np.arctan2(sin_mean, cos_mean)
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    R = np.clip(R, 1e-8, 1.0)
    circ_std = np.sqrt(-2 * np.log(R))

    return mean_angle, circ_std


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def moving_average(x, k=3):
    x = np.asarray(x, dtype=float)

    if len(x) == 0 or k <= 1:
        return x.copy()

    if k > len(x):
        k = len(x)

    kernel = np.ones(k) / k
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    xpad = np.pad(x, (pad_left, pad_right), mode="edge")

    return np.convolve(xpad, kernel, mode="valid")


def smooth_circular(means, k=3):
    means = np.asarray(means)
    cos_m = moving_average(np.cos(means), k)
    sin_m = moving_average(np.sin(means), k)
    return np.arctan2(sin_m, cos_m)


def classify_phase_state(angle, tol=np.deg2rad(50)):
    angle = wrap_to_pi(angle)

    if np.abs(angle) <= tol:
        return "in-phase"
    elif np.abs(np.abs(angle) - np.pi) <= tol:
        return "anti-phase"
    else:
        return "intermediate"


def compute_relative_phase(x1, x2, t, freq_band=(0, 100)):
    res = cwt_coherence(x1, t, x2, t, standardize=False, freq_method="log")

    phi = res.xw_phase
    R2 = res.xw_coherence
    freq = res.freq

    band = (freq >= freq_band[0]) & (freq <= freq_band[1])

    if not np.any(band):
        raise ValueError(f"No frequencies found in freq_band={freq_band}")

    w = R2[:, band]
    z = np.exp(1j * phi[:, band])

    denom = np.sum(w, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)

    zbar = np.sum(w * z, axis=1) / denom

    # Wrapped phase for circular statistics
    phi_wrapped = np.angle(zbar)

    # Unwrapped phase only for trend visualization if needed
    phi_unwrapped = np.unwrap(phi_wrapped)

    bins = np.linspace(-np.pi, np.pi, 73)  # 5 degree bins
    hist, edges = np.histogram(phi_wrapped, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if np.isnan(phi_wrapped).all():
        print("BAD WINDOW")
        print("len(t) =", len(t))
        print("t[0], t[-1] =", t[0], t[-1])
        print("phi nan count =", np.isnan(phi).sum())
        print("R2 nan count =", np.isnan(R2).sum())
        print("freq min/max =", np.min(freq), np.max(freq))

    return phi_wrapped, phi_unwrapped, hist, centers


def windowed_phase_analysis(x1, x2, t, window_size, freq_band=(0, 100)):
    windows = []
    start = t[0]

    while start < t[-1]:
        end = start + window_size
        mask = (t >= start) & (t < end)

        if np.sum(mask) > 10:
            x1_win = x1[mask]
            x2_win = x2[mask]
            t_win = t[mask]

            # ------------------------------
            # Relative phase
            # ------------------------------
            phi_wrapped, phi_unwrapped, pdf, centers = compute_relative_phase(
                x1_win, x2_win, t_win, freq_band
            )

            mean_phase, std_phase = circular_stats(phi_wrapped)

            # ------------------------------
            # Correlation (Pearson)
            # ------------------------------
            valid = ~np.isnan(x1_win) & ~np.isnan(x2_win)

            if np.sum(valid) > 2:
                res = association(x1_win[valid], x2_win[valid], statistic="pearsonr")
                corr_value = res[0]
                try:
                    p_value = res[1]
                except Exception:
                    p_value = None
            else:
                corr_value = np.nan
                p_value = np.nan

            windows.append(
                {
                    "start": start,
                    "end": end,
                    "center": 0.5 * (start + end),
                    "mean": mean_phase,
                    "std": std_phase,
                    "phi_wrapped": phi_wrapped,
                    "phi_unwrapped": phi_unwrapped,
                    "t": t_win,
                    "pdf": pdf,
                    "centers": centers,
                    "corr": corr_value,
                    "p_value": p_value,
                }
            )

        start = end

    return windows


def circular_distance(a, b):
    # return np.abs(wrap_to_pi(a - b))
    return np.abs(np.angle(np.exp(1j * (a - b))))


def detect_phase_transitions(
    windows,
    bpm_df,
    phase_jump_thresh=np.deg2rad(50),
    std_thresh=None,
    smooth_k=3,
    min_dwell=2,
):
    if len(windows) < max(3, min_dwell + 1):
        return [], {}

    centers = np.array([w["center"] for w in windows])
    means = np.array([w["mean"] for w in windows])
    stds = np.array([w["std"] for w in windows])
    bpms = {
        "in-phase to anti-phase": [],
        "anti-phase to in-phase": [],
        "anti-phase to intermediate": [],
        "intermediate to anti-phase": [],
        "intermediate to in-phase": [],
        "in-phase to intermediate": [],
    }

    means_smooth = smooth_circular(means, k=smooth_k)
    stds_smooth = moving_average(stds, k=smooth_k)

    phase_jump = np.zeros_like(means_smooth)
    phase_jump[1:] = circular_distance(means_smooth[1:], means_smooth[:-1])

    if std_thresh is None:
        std_thresh = np.nanmean(stds_smooth) + np.nanstd(stds_smooth)

    for i, w in enumerate(windows):
        w["mean_smooth"] = means_smooth[i]
        w["std_smooth"] = stds_smooth[i]
        w["phase_jump"] = phase_jump[i]
        w["state"] = classify_phase_state(means_smooth[i])

    states = np.array([w["state"] for w in windows])

    candidate_idx = np.where(
        (phase_jump >= phase_jump_thresh) | (stds_smooth >= std_thresh)
    )[0]

    transitions = []
    used = set()
    prev_idx = 0

    for idx in candidate_idx:
        idx = int(idx)

        if idx == 0 or idx in used:
            continue

        end_idx = min(len(means_smooth), idx + min_dwell)
        future = means_smooth[idx:end_idx]
        future_states = states[idx:end_idx]

        unique, counts = np.unique(future_states, return_counts=True)
        count_dict = dict(zip(unique, counts))
        # print(count_dict)

        old_state_angle = means_smooth[idx - 1]
        new_state_angle = means_smooth[idx]

        old_label = (
            transitions[-1]["to_state"]
            if len(transitions)
            else classify_phase_state(old_state_angle)
        )

        new_label = (
            unique[np.argmax(counts)]
            if len(future_states) > 1
            else classify_phase_state(new_state_angle)
        )
        if old_label == new_label:
            continue

        if len(future) < min_dwell:
            continue

        stays_new = []
        for f in future:
            d_new = circular_distance(f, new_state_angle)
            d_old = circular_distance(f, old_state_angle)
            stays_new.append(d_new < d_old)

        if all(stays_new):
            transition = {
                "idx": idx,
                "time": float(centers[idx]),
                "from_phase": old_state_angle,
                "to_phase": new_state_angle,
                "from_state": old_label,
                "to_state": new_label,
                "phase_jump": float(phase_jump[idx]),
                "std": float(stds_smooth[idx]),
            }

            prev_i = int(prev_idx)
            transition["duration"] = {
                "start": float(centers[prev_i]),
                "end": float(centers[idx]),
                "length": float(centers[idx] - centers[prev_i]),
                "state": old_label,
            }

            transition["prev_bpm"] = np.nan
            if len(transitions) > 0:
                prev_start = transitions[-1]["duration"]["start"]
                prev_end = transitions[-1]["duration"]["end"]

                bpm_segment = bpm_df.loc[
                    (bpm_df["time"] >= prev_start) & (bpm_df["time"] < prev_end), "bpm"
                ]

                prev_bpm = bpm_segment.mean()
                transition["prev_bpm"] = prev_bpm

                bpms[f"{old_label} to {new_label}"].append(prev_bpm)

            transitions.append(transition)
            prev_idx = idx

            for j in range(max(0, idx - 1), min(len(means_smooth), idx + min_dwell)):
                used.add(j)

    summary = {
        "centers": centers,
        "means": means,
        "means_smooth": means_smooth,
        "stds": stds,
        "stds_smooth": stds_smooth,
        "phase_jump": phase_jump,
        "phase_jump_thresh": phase_jump_thresh,
        "std_thresh": std_thresh,
        "bpms": bpms,
    }

    return transitions, summary


def compute_bpm_diff(transitions):
    bpm_diffs = []
    for i in range(len(transitions) - 1):
        prev_bpm = {
            "bpm": transitions[i]["prev_bpm"],
            "state": f'{transitions[i]["from_state"]} to {transitions[i]["to_state"]}',
        }
        future_bpm = {
            "bpm": transitions[i + 1]["prev_bpm"],
            "state": f'{transitions[i+1]["from_state"]} to {transitions[i+1]["to_state"]}',
        }

        if np.isnan(prev_bpm["bpm"]) or np.isnan(future_bpm["bpm"]):
            continue
        # print(prev_bpm, future_bpm)
        bpm_diff = future_bpm["bpm"] - prev_bpm["bpm"]
        # print("BPM diff: ", bpm_diff)
    return bpm_diffs


def pairwise_independent_ttests(data_dict, correction="bonferroni"):
    """
    Run pairwise independent Welch's t-tests between all non-empty transition groups.

    Parameters
    ----------
    data_dict : dict
        Example:
        {
            'in-phase to anti-phase': [...],
            'anti-phase to in-phase': [...],
            ...
        }

    correction : str
        'bonferroni' or None

    Returns
    -------
    results_df : pandas.DataFrame
    """

    # ---------------------------------------------------
    # 1) Remove empty groups and convert to float arrays
    # ---------------------------------------------------
    clean_data = {
        k: np.array([float(x) for x in v]) for k, v in data_dict.items() if len(v) > 1
    }

    if len(clean_data) < 2:
        print("Need at least 2 non-empty groups with >1 value each.")
        return pd.DataFrame()

    # ---------------------------------------------------
    # 2) Generate all pairwise comparisons
    # ---------------------------------------------------
    group_pairs = list(combinations(clean_data.keys(), 2))
    m = len(group_pairs)  # number of comparisons

    results = []

    for g1, g2 in group_pairs:
        x1 = clean_data[g1]
        x2 = clean_data[g2]

        # Welch's independent t-test
        t_stat, p_val = ttest_ind(x1, x2, equal_var=False)

        # optional multiple comparison correction
        if correction == "bonferroni":
            p_adj = min(p_val * m, 1.0)
        else:
            p_adj = p_val

        results.append(
            {
                "group1": g1,
                "group2": g2,
                "n1": len(x1),
                "n2": len(x2),
                "mean1": np.mean(x1),
                "std1": np.std(x1, ddof=1),
                "mean2": np.mean(x2),
                "std2": np.std(x2, ddof=1),
                "t_stat": t_stat,
                "p_value": p_val,
                "p_adj": p_adj,
                "significant": p_adj < 0.05,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


# ============================================================
# -------------------- VISUALIZATION -------------------------
# ============================================================


# def plot_angles(t, theta1, theta2):
#     plt.figure(figsize=(12, 4))
#     plt.plot(t, theta1, label="p1")
#     plt.plot(t, theta2, label="p2")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Angle (rad)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def plot_1(t, data, label, y_unit="Angle (rad)", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 4))
    ax.plot(t, data, label=label)
    ax.set_ylabel(y_unit)
    ax.legend()
    ax.grid(True)

    return ax


def plot_angular_velocity(t, omega1, omega2):
    plt.figure(figsize=(12, 4))
    plt.plot(t, omega1, label="omega1")
    plt.plot(t, omega2, label="omega2")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_windowed_phase(windows):
    """
    Plot:
    1) Windowed relative phase mean ± std over time
    2) Relative phase PDF

    Assumes each window dict contains:
        'center', 'mean', 'std',
        'centers' (phase bin centers),
        'pdf' (probability density)
    """
    if len(windows) == 0:
        print("No windows to plot.")
        return
    # --------------------------------------------------
    # 1. Windowed mean ± std over time
    # --------------------------------------------------
    centers_time = np.array([w["center"] for w in windows])
    means = np.array([w["mean"] for w in windows])
    stds = np.array([w["std"] for w in windows])

    plt.figure(figsize=(24, 20))
    plt.errorbar(centers_time, means, yerr=stds, fmt="o", capsize=4)
    plt.ylim(-np.pi * 1.05, np.pi * 1.05)
    plt.xlabel("Window center time (s)")
    plt.ylabel("Relative phase (rad)")
    plt.title("Windowed Relative Phase (mean ± std)")
    plt.grid(True)
    plt.show()

    # --------------------------------------------------
    # 2. Relative phase PDF
    # --------------------------------------------------
    phase_centers = windows[0]["centers"]
    pdf = windows[0]["pdf"]

    plt.figure()
    plt.plot(phase_centers, pdf)
    plt.xlabel("Relative phase (rad)")
    plt.ylabel("PDF")
    plt.title("Relative Phase Distribution")
    plt.grid(True)
    plt.show()


def plot_transition_histogram(transitions, bins=20, density=False):
    """
    Plot histogram of transition times.

    Parameters
    ----------
    transitions : list of dict
        Output from detect_phase_transitions()
    bins : int
        Number of bins
    density : bool
        If True, normalize histogram
    """
    if len(transitions) == 0:
        print("No transitions to plot.")
        return

    times = np.array([tr["time"] for tr in transitions])

    plt.figure(figsize=(8, 4))
    plt.hist(times, bins=bins, density=density, edgecolor="black")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency" if not density else "Density")
    plt.title("Phase Transition Frequency (Histogram)")
    plt.grid(True)

    plt.show()


def plot_transition_bar(transitions, bin_width=1.0):
    """
    Plot bar chart of transition counts over time bins.

    Parameters
    ----------
    transitions : list of dict
    bin_width : float
        Width of time bins (seconds)
    """
    if len(transitions) == 0:
        print("No transitions to plot.")
        return

    times = np.array([tr["time"] for tr in transitions])

    t_min = np.min(times)
    t_max = np.max(times)

    bins = np.arange(t_min, t_max + bin_width, bin_width)
    counts, edges = np.histogram(times, bins=bins)

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 4))
    plt.bar(centers, counts, width=3)  # bin_width * 0.1)

    plt.xlabel("Time (s)")
    plt.ylabel("Number of transitions")
    plt.title("Phase Transition Frequency (Bar Plot)")
    plt.grid(True)

    plt.show()


def plot_transition_types(transitions):
    """
    Plot frequency of transition types (state-to-state).
    """
    if len(transitions) == 0:
        print("No transitions to plot.")
        return

    labels = [f"{tr['from_state']}→{tr['to_state']}" for tr in transitions]

    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts)

    plt.xlabel("Transition type")
    plt.ylabel("Count")
    plt.title("Phase Transition Types")
    plt.xticks(rotation=30)
    plt.grid(True)

    plt.show()


def plot_state_histogram(transitions):
    """
    Plot frequency of transition types (state-to-state).
    """
    if len(transitions) == 0:
        print("No transitions to plot.")
        return

    labels = [f"{tr['duration']['state']}" for tr in transitions]

    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts)

    plt.xlabel("Coordination State")
    plt.ylabel("Count")
    plt.title("Phases")
    plt.xticks(rotation=30)
    plt.grid(True)

    plt.show()


def plot_windowed_correlation(windows):
    centers = np.array([w["center"] for w in windows])
    corrs = np.array([w["corr"] for w in windows])

    plt.figure(figsize=(10, 4))
    plt.scatter(centers, corrs, marker="o")
    plt.axhline(0, color="k", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Pearson r")
    plt.title("Windowed Correlation")
    plt.grid(True)
    plt.show()


def plot_angles(time, p1, p2, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 4))

    ax.plot(time, p1, label="p1_unwrapped")
    ax.plot(time, p2, label="p2_unwrapped")
    ax.set_ylabel("Angle")
    ax.set_title("Unwrapped Angles")
    ax.legend()
    ax.grid(True)

    return ax


def plot_phase_transitions(summary, transitions, df_processed):
    if not summary:
        print("No transition summary to plot.")
        return

    centers = summary["centers"]
    means_smooth = summary["means_smooth"]
    stds_smooth = summary["stds_smooth"]
    phase_jump = summary["phase_jump"]

    # -----------------------------
    # State -> color mapping
    # -----------------------------
    state_colors = {
        "in-phase": "green",
        "anti-phase": "red",
        "intermediate": "orange",
    }

    transition_colors = {
        "in-phase to anti-phase": "red",
        "anti-phase to in-phase": "green",
        "in-phase to intermediate": "orange",
        "intermediate to in-phase": "orange",
        "anti-phase to intermediate": "orange",
        "intermediate to anti-phase": "orange",
    }

    # Classify each smoothed mean phase point
    states = [classify_phase_state(m) for m in means_smooth]
    point_colors = [state_colors.get(s, "black") for s in states]

    fig, axes = plt.subplots(4, 1, figsize=(24, 20), sharex=True)

    # =========================================================
    # 1) Positional / angle graph on top
    # =========================================================
    plot_angles(
        df_processed["time"],
        df_processed["p1_unwrapped"],
        df_processed["p2_unwrapped"],
        ax=axes[0],
    )
    axes[0].set_title("Agent Angles + Coordination Transitions")

    # =========================================================
    # 2) Mean phase
    # =========================================================
    axes[1].plot(centers, means_smooth, color="black", alpha=0.5, linewidth=1.5)
    axes[1].scatter(centers, means_smooth, c=point_colors, s=50, zorder=3)

    axes[1].axhline(0, linestyle="--", alpha=0.6, color="black", label="In-phase (0)")
    axes[1].axhline(
        np.pi, linestyle="--", alpha=0.6, color="black", label="Anti-phase (π)"
    )
    axes[1].axhline(-np.pi, linestyle="--", alpha=0.6, color="black")
    axes[1].axhline(-np.pi + 0.87, linestyle="--", alpha=0.6, color="gray")
    axes[1].axhline(np.pi - 0.87, linestyle="--", alpha=0.6, color="gray")
    axes[1].axhline(0 + 0.87, linestyle="--", alpha=0.6, color="gray")
    axes[1].axhline(0 - 0.87, linestyle="--", alpha=0.6, color="gray")
    axes[1].set_ylabel("Mean phase (rad)")
    axes[1].set_title("Windowed Relative Phase and Detected Transitions")
    axes[1].grid(True)

    # =========================================================
    # 3) Circular std
    # =========================================================
    axes[2].plot(centers, stds_smooth, marker="o", color="black")
    axes[2].axhline(
        summary["std_thresh"],
        linestyle="--",
        alpha=0.7,
        color="purple",
        label="std thresh",
    )
    axes[2].set_ylabel("Circular std")
    axes[2].legend()
    axes[2].grid(True)

    # =========================================================
    # 4) Phase jump
    # =========================================================
    axes[3].plot(centers, phase_jump, marker="o", color="black")
    axes[3].axhline(
        summary["phase_jump_thresh"],
        linestyle="--",
        alpha=0.7,
        color="blue",
        label="jump thresh",
    )
    axes[3].set_ylabel("Phase jump (rad)")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend()
    axes[3].grid(True)

    # =========================================================
    # Mark transitions on all panels with state-specific colors
    # =========================================================
    for tr in transitions:
        tag = f"{tr['from_state']} to {tr['to_state']}"
        tr_color = transition_colors.get(tag, "black")
        tr_label = tag
        # print("tag", tag)
        # print("tr_label", tr_label)
        # print("tr_color", tr_color)

        # # horizontal lines across all panels
        # for ax in axes:
        #     ax.axhline(tr["to_phase"], color=tr_color, alpha=0.8, linestyle="--")

        # vertical lines across all panels
        for ax in axes:
            ax.axvline(tr["time"], color=tr_color, alpha=0.8, linestyle="--")

        # label on mean-phase plot
        axes[1].text(
            tr["time"],
            tr["to_phase"],
            tr_label,
            color=tr_color,
            fontsize=9,
            rotation=90,
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    # =========================================================
    # Custom legend for state colors
    # =========================================================
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            linestyle="--",
            label="In-phase to Anti-phase",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            lw=2,
            linestyle="--",
            label="Anti-phase to In-phase",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            lw=2,
            linestyle="--",
            label="Transitions to/from Intermediate",
        ),
    ]

    axes[1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_bmp(df_processed, transitions, summary=None):
    if not summary:
        print("No transition summary to plot.")
        return

    centers = summary["centers"]
    means_smooth = summary["means_smooth"]
    # -----------------------------
    # State -> color mapping
    # -----------------------------
    state_colors = {
        "in-phase": "green",
        "anti-phase": "red",
        "intermediate": "orange",
    }

    transition_colors = {
        "in-phase to anti-phase": "purple",
        "anti-phase to in-phase": "blue",
        "in-phase to intermediate": "brown",
        "intermediate to in-phase": "brown",
        "anti-phase to intermediate": "brown",
        "intermediate to anti-phase": "brown",
    }

    # Classify each smoothed mean phase point
    states = [classify_phase_state(m) for m in means_smooth]
    point_colors = [state_colors.get(s, "black") for s in states]

    fig, axes = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    plot_1(
        df_processed["time"],
        df_processed["p1_unwrapped"],
        "p1_unwrapped",
        ax=axes[0],
    )
    plot_1(
        df_processed["time"],
        df_processed["bpm"],
        "Beats per minute (BPM)",
        ax=axes[1],
    )
    axes[0].set_title("Positional Data")
    axes[1].set_title("Heartbeat Rate (BPM)")

    # =========================================================
    # Mark transitions on all panels with state-specific colors
    # =========================================================
    for tr in transitions:
        tag = f"{tr['from_state']} to {tr['to_state']}"
        tr_color = transition_colors.get(tag, "black")

        for ax in axes:
            ax.axvline(tr["time"], color=tr_color, alpha=0.8, linestyle="--")

        axes[1].text(
            tr["time"],
            0.95,
            tag,
            transform=axes[
                1
            ].get_xaxis_transform(),  # x in data coords, y in axis coords
            color=tr_color,
            fontsize=9,
            rotation=90,
            va="top",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    # =========================================================
    # Custom legend for state colors
    # =========================================================
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="blue",
            lw=2,
            linestyle="--",
            label="In-phase to Anti-phase",
        ),
        Line2D(
            [0],
            [0],
            color="purple",
            lw=2,
            linestyle="--",
            label="Anti-phase to In-phase",
        ),
        Line2D(
            [0],
            [0],
            color="brown",
            lw=2,
            linestyle="--",
            label="Transitions to/from Intermediate",
        ),
    ]

    axes[1].legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_bpms_strip(transition_summary):
    """
    Plot BPM values for each transition type as:
    - jittered scatter points
    - mean line
    - std error bar

    Parameters
    ----------
    data_dict : dict
        Example:
        {
            'in-phase to anti-phase': [73.5, 70.4, ...],
            'anti-phase to in-phase': [71.9, 70.7, ...],
            ...
        }
    """

    # ---------------------------------------------------
    # 1) Remove empty groups and convert to normal floats
    # ---------------------------------------------------
    clean_data = {
        k: [float(x) for x in v]
        for k, v in transition_summary["bpms"].items()
        if len(v) > 0
    }

    if len(clean_data) == 0:
        print("No non-empty transition groups to plot.")
        return

    # ---------------------------------------------------
    # 2) Sort by mean BPM
    # ---------------------------------------------------
    clean_data = dict(sorted(clean_data.items(), key=lambda x: np.mean(x[1])))

    labels = [f"{k}\n(n={len(v)})" for k, v in clean_data.items()]
    values = list(clean_data.values())

    # ---------------------------------------------------
    # 3) Plot
    # ---------------------------------------------------
    plt.figure(figsize=(12, 6))

    for i, (label, vals) in enumerate(clean_data.items()):
        vals = np.array(vals)

        # spread x positions so points don't overlap
        x = np.linspace(i - 0.08, i + 0.08, len(vals))

        # scatter points
        plt.scatter(x, vals, alpha=0.7, s=60)

        # mean and std
        mean = np.mean(vals)
        std = np.std(vals)

        # mean line
        plt.hlines(mean, i - 0.2, i + 0.2, linewidth=2, color="green", alpha=0.7)

        # std error bar
        plt.errorbar(
            i, mean, yerr=std, fmt="o", capsize=6, markersize=8, color="grey", alpha=0.7
        )

        # label mean and std
        plt.text(
            i + 0.12,
            mean,
            f"μ={mean:.2f}\nσ={std:.2f}",
            fontsize=10,
            va="center",
            ha="left",
        )

    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("BPM")
    plt.title("BPM by Transition Type")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_transition_diagnostics(
    summary, windows, transitions=None, min_dwell=5, candidate_only=True
):
    """
    Debug plot for transition detection logic.

    Shows:
    1) smoothed mean phase over time
    2) candidate transition indices
    3) dwell-confirmation window for each candidate
    4) whether each future point is closer to old angle or new angle

    Parameters
    ----------
    summary : dict
        Output transition_summary from detect_phase_transitions()
    windows : list of dict
        Window list from windowed_phase_analysis()
    transitions : list of dict or None
        Accepted transitions from detect_phase_transitions()
    min_dwell : int
        Same min_dwell used in detection
    candidate_only : bool
        If True, only plot candidate indices.
        If False, also marks accepted transitions if provided.
    """
    if not summary:
        print("No summary provided.")
        return

    centers = np.asarray(summary["centers"])
    means_smooth = np.asarray(summary["means_smooth"])
    stds_smooth = np.asarray(summary["stds_smooth"])
    phase_jump = np.asarray(summary["phase_jump"])
    phase_jump_thresh = summary["phase_jump_thresh"]
    std_thresh = summary["std_thresh"]

    # same candidate logic as detect_phase_transitions
    candidate_idx = np.where(
        (phase_jump >= phase_jump_thresh) | (stds_smooth >= std_thresh)
    )[0]

    if len(candidate_idx) == 0:
        print("No candidate transitions found.")
        return

    nrows = len(candidate_idx)
    fig, axes = plt.subplots(nrows, 1, figsize=(18, max(4, 4 * nrows)), sharex=True)

    if nrows == 1:
        axes = [axes]

    accepted_idx = set()
    if transitions is not None:
        accepted_idx = {tr["idx"] for tr in transitions if "idx" in tr}

    for ax, idx in zip(axes, candidate_idx):
        if idx == 0:
            ax.set_title(f"Candidate idx={idx} skipped (no previous window)")
            continue

        old_angle = means_smooth[idx - 1]
        new_angle = means_smooth[idx]

        old_label = classify_phase_state(old_angle)
        new_label = classify_phase_state(new_angle)

        end_idx = min(len(means_smooth), idx + min_dwell)
        future_idx = np.arange(idx, end_idx)
        future = means_smooth[idx:end_idx]

        # distances used by your actual dwell logic
        d_new = np.array([circular_distance(f, new_angle) for f in future])
        d_old = np.array([circular_distance(f, old_angle) for f in future])
        stays_new = d_new < d_old

        # base trajectory
        ax.plot(centers, means_smooth, color="0.75", linewidth=1.2, zorder=1)
        ax.scatter(centers, means_smooth, color="0.6", s=18, zorder=2)

        # reference lines
        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
        ax.axhline(np.pi, linestyle="--", color="gray", alpha=0.4)
        ax.axhline(-np.pi, linestyle="--", color="gray", alpha=0.4)

        # candidate index
        ax.axvline(centers[idx], color="purple", linestyle="--", linewidth=2, zorder=3)

        # old/new anchor points
        ax.scatter(
            centers[idx - 1], old_angle, color="blue", s=80, zorder=5, label="old angle"
        )
        ax.scatter(
            centers[idx], new_angle, color="red", s=80, zorder=5, label="new angle"
        )

        # dwell window shading
        if len(future_idx) > 0:
            ax.axvspan(
                centers[future_idx[0]],
                centers[future_idx[-1]],
                color="gold",
                alpha=0.15,
                zorder=0,
            )

        # future points: green if closer to new, orange if closer to old
        for j, fidx in enumerate(future_idx):
            color = "green" if stays_new[j] else "orange"
            marker = "o" if stays_new[j] else "x"
            ax.scatter(
                centers[fidx],
                means_smooth[fidx],
                color=color,
                marker=marker,
                s=100,
                zorder=6,
            )

            ax.text(
                centers[fidx],
                means_smooth[fidx] + 0.18,
                f"{j}\nnew:{d_new[j]:.2f}\nold:{d_old[j]:.2f}",
                fontsize=8,
                ha="center",
                va="bottom",
                color=color,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        accepted = len(future) == min_dwell and np.all(stays_new)

        title = (
            f"Candidate idx={idx}, t={centers[idx]:.3f}s | "
            f"{old_label} -> {new_label} | "
            f"accepted_by_dwell={accepted}"
        )

        if transitions is not None and idx in accepted_idx:
            title += " | IN transitions[]"
        elif transitions is not None:
            title += " | NOT in transitions[]"

        ax.set_title(title)
        ax.set_ylabel("Mean phase (rad)")
        ax.grid(True, alpha=0.3)

        # compact legend text inside panel
        info = (
            f"phase_jump={phase_jump[idx]:.3f} (th={phase_jump_thresh:.3f})\n"
            f"std={stds_smooth[idx]:.3f} (th={std_thresh:.3f})\n"
            f"dwell window = [{idx}, {end_idx})  len={len(future)}\n"
            f"green = closer to NEW, orange = closer to OLD"
        )
        ax.text(
            0.01,
            0.98,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


# ============================================================
# --------------------- FULL PIPELINE ------------------------
# ============================================================


def run_full_pipeline(
    filepath,
    physiopath,
    start,
    end,
    dt=0.001,
    window_size=0.5,
    freq_band=(0, 100),
    phase_jump_thresh=np.deg2rad(50),
    smooth_k=3,
    min_dwell=2,
):

    df_raw = load_and_clean_data(filepath, physiopath)
    trial_dfs, condition_ranges = divide_by_trials(df_raw)
    print(condition_ranges)

    df_out = {}
    windows_out = {}
    transitions_out = {}
    transition_summary_out = {}
    # df = divide_data(df_raw, start, end)
    for i in range(3, 7):
        print("Looping through trials...")
        trial_name = condition_dict[i]
        start_t = condition_ranges[trial_name][0] + 50
        end_t = condition_ranges[trial_name][1] - 50
        print("trial: ", trial_name, "\n")
        print("Looking at..... ", start_t, " to ", end_t)
        df = divide_data(df_raw.copy(), start_t, end_t)
        df = unwrap_angles(df)
        # print(df.head(), "\n")
        df_rs = resample_signals(df, dt_target=dt)
        df_rs = compute_dynamics(df_rs, dt)

        t = df_rs["time"].to_numpy()
        theta1 = df_rs["p1_unwrapped"].to_numpy()
        theta2 = df_rs["p2_unwrapped"].to_numpy()

        windows = windowed_phase_analysis(
            theta1,
            theta2,
            t,
            window_size=window_size,
            freq_band=freq_band,
        )
        print(df_rs.head())
        transitions, transition_summary = detect_phase_transitions(
            windows,
            df_rs[["time", "bpm"]],
            phase_jump_thresh=phase_jump_thresh,
            smooth_k=smooth_k,
            min_dwell=min_dwell,
        )

        # phase analysis
        # plot_windowed_phase(windows)
        # plot_windowed_correlation(windows)
        plot_phase_transitions(transition_summary, transitions, df_rs)
        for tr in transitions:
            print(
                f"Transition at t={tr['time']:.3f}s | "
                f"{tr['from_state']} -> {tr['to_state']} | "
                f"jump={np.degrees(tr['phase_jump']):.1f} deg | "
                f"std={tr['std']:.3f} | "
                f"start={tr['duration']['start']} , end={tr['duration']['end']} | "
                # f"type: {type(tr['duration']['start'])} | "
                f"duration={tr['duration']['length']:.3f} , {tr['duration']['state']}"
            )
        plot_transition_types(transitions)
        plot_state_histogram(transitions)

        # HeartRate analysis
        plot_bmp(df_rs, transitions, transition_summary)
        plot_bpms_strip(transition_summary)
        results_df = pairwise_independent_ttests(transition_summary["bpms"])
        print(results_df.round(3))

        df_out[f"{trial_name}"] = df_rs
        windows_out[f"{trial_name}"] = windows
        transitions_out[f"{trial_name}"] = transitions
        transition_summary_out[f"{trial_name}"] = transition_summary

    return df_out, windows_out, transitions_out, transition_summary_out


def test_pipeline(
    filepath,
    start,
    end,
    dt=0.001,
    window_size=0.5,
    freq_band=(0, 100),
    phase_jump_thresh=np.deg2rad(50),
    smooth_k=3,
    min_dwell=2,
):
    df_raw = load_and_clean_data(filepath)
    df = divide_data(df_raw, start, end)
    df = unwrap_angles(df)

    df_rs = resample_signals(df, dt_target=dt)
    df_rs = compute_dynamics(df_rs, dt)

    t = df_rs["time"].to_numpy()
    theta1 = df_rs["p1_unwrapped"].to_numpy()
    theta2 = df_rs["p2_unwrapped"].to_numpy()

    windows = windowed_phase_analysis(
        theta1,
        theta2,
        t,
        window_size=window_size,
        freq_band=freq_band,
    )

    transitions, transition_summary = detect_phase_transitions(
        windows,
        phase_jump_thresh=phase_jump_thresh,
        smooth_k=smooth_k,
        min_dwell=min_dwell,
    )

    return df_rs, windows, transitions, transition_summary


# ============================================================
# ---------------------- EXAMPLE USAGE -----------------------
# ============================================================

# df_processed, phase_windows, transitions, transition_summary = run_full_pipeline(
#     "data/am0324_analysis/20260324_buntingphysio_bunt_am0324.xlsx",
#     1200,
#     2000,
#     dt=0.001,
#     window_size=0.5,
#     freq_band=(0, 100),
#     phase_jump_thresh=np.deg2rad(45),
#     smooth_k=3,
#     min_dwell=2,
# )

# plot_angles(
#     df_processed["time"],
#     df_processed["p1_unwrapped"],
#     df_processed["p2_unwrapped"],
# )

# plot_angular_velocity(
#     df_processed["time"],
#     df_processed["omega1"],
#     df_processed["omega2"],
# )

# plot_windowed_phase(phase_windows)
# plot_windowed_correlation(phase_windows)
# plot_phase_transitions(transition_summary, transitions)

# for tr in transitions:
#     print(
#         f"Transition at t={tr['time']:.3f}s | "
#         f"{tr['from_state']} -> {tr['to_state']} | "
#         f"jump={np.degrees(tr['phase_jump']):.1f} deg | "
#         f"std={tr['std']:.3f}"
#     )
