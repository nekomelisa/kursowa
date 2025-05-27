from pathlib import Path
import numpy as np
import pickle


def read_series(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float)


def load_data(data_root: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {"BC": {}, "Control": {}}
    for group in data:
        folder = data_root / group
        if not folder.exists():
            raise FileNotFoundError(f"Не знайдено папку {folder}")
        for filepath in folder.glob("*_*"):
            name, channel = filepath.stem.split("_", 1)
            series = read_series(filepath)
            data[group].setdefault(name, {})[channel] = series
    return data


def compute_p_statistic(x_sorted, y, g=3):
    n = x_sorted.shape[0]
    m = y.shape[0]

    i_idx, j_idx = np.triu_indices(n, k=1)
    p0 = (j_idx - i_idx) / (n + 1)

    counts = [((x_sorted[i] <= y) & (y <= x_sorted[j])).sum()
              for i, j in zip(i_idx, j_idx)]
    h = np.array(counts, float) / m
    neff = m + g * g
    a = h * m + 0.5 * g * g
    b = g * np.sqrt(h * (1 - h) * m + 0.25 * g * g)
    p_low = (a - b) / neff
    p_up = (a + b) / neff

    I = (p_low <= p0) & (p0 <= p_up)

    return I.mean()


if __name__ == "__main__":

    project_root = Path("/Users/nekomelisa/Documents/jez/code/data").parent
    data_root = project_root / "data"

    data = load_data(data_root)

    for group in data:
        for sample_id, channels in data[group].items():
            for channel, arr in channels.items():
                arr.sort()

    bc_list, bc_labels = [], []
    for pid, channels in data["BC"].items():
        for ch, arr in channels.items():
            bc_list.append(arr)
            bc_labels.append(f"{pid}_{ch}")

    ctrl_list, ctrl_labels = [], []
    for pid, channels in data["Control"].items():
        for ch, arr in channels.items():
            ctrl_list.append(arr)
            ctrl_labels.append(f"{pid}_{ch}")

    n_bc = len(bc_list)
    S_bc = np.zeros((n_bc, n_bc), dtype=float)
    for i in range(n_bc):
        for j in range(n_bc):
            S_bc[i, j] = compute_p_statistic(bc_list[i], bc_list[j], g=3)

    m_ctrl = len(ctrl_list)
    S_ctrl = np.zeros((m_ctrl, m_ctrl), dtype=float)
    for i in range(m_ctrl):
        for j in range(m_ctrl):
            S_ctrl[i, j] = compute_p_statistic(ctrl_list[i], ctrl_list[j], g=1.96)

    S_cross = np.zeros((n_bc, m_ctrl), dtype=float)
    for i in range(n_bc):
        for j in range(m_ctrl):
            S_cross[i, j] = compute_p_statistic(bc_list[i], ctrl_list[j], g=1.96)

    payload = {
        "S_bc": S_bc,
        "S_ctrl": S_ctrl,
        "S_cross": S_cross,
        "bc_labels": bc_labels,
        "ctrl_labels": ctrl_labels
    }
    with open("sim_matrices.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("Матриці подібності обчислено й збережено в sim_matrices.pkl")
