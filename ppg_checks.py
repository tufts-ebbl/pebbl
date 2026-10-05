"""
PPG-specific checks for the session summary (ppg-plan §4b), in Steps 2 and 3:

  - markers inside stretches the machine marked invalid for PPG (the
    sidecar's MachineQC.ppg BadSegments: device zero runs and bridged
    samples, per sample once physioProcess adds its per-sample mask);
  - markers not on a crest of a pulse: not the highest sample within
    +/-CREST_WINDOW_SEC, i.e. on a slope (e.g. 11 machine peaks on slopes in
    sub-001). Until 2026-10-04 this was "not at the top" (+/-50 ms). In 10
    run-10 runs the pulse top is flat or has two crests 70-130 ms apart,
    and the highest point flips between them beat to beat (physioProcess
    maintainer, 2026-10-04). The RA's peak is the anchor for physioProcess's
    upstroke search (HLU), and the upstroke under either crest is the same,
    so a mark on either crest, or either end of a flat top, passes; marks on
    slopes are still caught. Same flags as +/-50 ms on sub-001 and on the
    practice files;
  - two markers on one pulse: consecutive markers closer than 0.3 x the
    median beat interval. CareTaker's pulse has two systolic humps ~75 ms
    apart, and the machine peak sits on the early one in ~10% of beats
    (sub-001), so two reviewers can mark one beat on different humps; both
    marks survive Step 3 unless one is deleted. A warning in Step 2; in
    Step 3 it BLOCKS saving until one is removed.

Every function is pure (no MNE, no GUI) and returns plain-language lines.
"""

import numpy as np

SAME_BEAT_FRACTION = 0.3
CREST_WINDOW_SEC = 0.02
MIN_PEAKS_FOR_SAME_BEAT = 5
EXAMPLES_PER_LINE = 3


def markers_in_spans(indices, spans):
    """The indices that fall inside any [start, end) span."""
    if not spans:
        return []
    starts = np.array([s for s, _e in spans])
    ends = np.array([e for _s, e in spans])
    return [int(i) for i in indices if np.any((starts <= i) & (i < ends))]


def markers_off_top(signal, indices, window_samples):
    """The indices that are not the highest sample within +/- window_samples (NaN-safe)."""
    signal = np.asarray(signal, dtype=float)
    off = []
    for i in indices:
        lo, hi = max(0, i - window_samples), min(len(signal), i + window_samples + 1)
        window = signal[lo:hi]
        if not len(window) or np.all(np.isnan(window)) or np.isnan(signal[i]):
            continue
        if signal[i] < np.nanmax(window):
            off.append(int(i))
    return off


def same_beat_pairs(indices, fraction=SAME_BEAT_FRACTION):
    """(a, b) consecutive marker pairs closer than `fraction` of the median interval."""
    indices = sorted(int(i) for i in indices)
    if len(indices) < MIN_PEAKS_FOR_SAME_BEAT:
        return []
    gaps = np.diff(indices)
    limit = fraction * float(np.median(gaps))
    return [(indices[k], indices[k + 1]) for k, gap in enumerate(gaps) if gap < limit]


def _times(samples, sfreq):
    shown = ", ".join(f"{s / sfreq:.1f}" for s in samples[:EXAMPLES_PER_LINE])
    return shown + (" s" if len(samples) <= EXAMPLES_PER_LINE else f" s, and {len(samples) - EXAMPLES_PER_LINE} more")


def _all_times(samples, sfreq):
    return ", ".join(f"{s / sfreq:.1f}" for s in samples) + " s"


def ppg_check_lines(indices, signal, sfreq, invalid_spans, window_samples=None, label="PPG", print_all=True):
    """
    Non-blocking check lines for a PPG channel's current markers.
    window_samples is the crest test's half-width (default CREST_WINDOW_SEC). They show
    EXAMPLES_PER_LINE times each; with print_all, a longer list is printed in
    full to the terminal (with the dropout spans, which the viewer doesn't
    draw) and the line says so (run-8 review: sub-001 had 33 such markers).
    """
    lines = []
    inside = markers_in_spans(indices, invalid_spans)
    listed = print_all and len(inside) > EXAMPLES_PER_LINE
    if inside:
        lines.append(f"{label}: {len(inside)} marker(s) inside stretches the machine marked as pulse dropouts "
                     f"(near {_times(inside, sfreq)}{', all listed in the terminal' if listed else ''}) -- check "
                     f"there is a real pulse; delete any that aren't.")
    if listed:
        print(f"{label} markers inside machine-flagged dropouts (all {len(inside)}): {_all_times(inside, sfreq)}")
        print(f"{label} machine-flagged dropouts ({len(invalid_spans)}): "
              + ", ".join(f"{a / sfreq:.1f}-{b / sfreq:.1f}" for a, b in invalid_spans) + " s")
    if window_samples is None:
        window_samples = round(CREST_WINDOW_SEC * sfreq)
    off = markers_off_top(signal, indices, window_samples)
    listed = print_all and len(off) > EXAMPLES_PER_LINE
    if off:
        lines.append(f"{label}: {len(off)} marker(s) not on a crest of a pulse (near {_times(off, sfreq)}"
                     f"{', all listed in the terminal' if listed else ''}) -- move each onto the nearest crest "
                     f"(delete it, then add it there), or delete it if there's no pulse.")
    if listed:
        print(f"{label} markers not on a crest of a pulse (all {len(off)}): {_all_times(off, sfreq)}")
    return lines


def same_beat_lines(indices, sfreq, label="PPG"):
    """Lines for pairs of markers that look like two marks on one pulse (see the module docstring)."""
    pairs = same_beat_pairs(indices)
    if not pairs:
        return []
    return [f"{label}: {len(pairs)} place(s) with two markers on what looks like one pulse (near "
            f"{_times([a for a, _b in pairs], sfreq)}) -- keep only the one at the pulse's highest point."]
