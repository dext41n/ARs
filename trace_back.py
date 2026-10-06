import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import label
from find_dates import find_dates, group_ars, binar_search, read_czechia, prep
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d


time = "2002-08-11T12:00:00.000000000"


def get_points(time, file, only_czechia = True):
    """
    It gets all points of atmospheric river which hit czechia at the given time.
    :param time: np.datatime
    :param file: True/False file of ARs
    :return: Returns mask
    """
    data = prep(file)
    mask_czechia = read_czechia(data)

    ar_field_t = data.sel(time=time)
    ar_full = (ar_field_t == 1)
    ar_cz = ar_full & (mask_czechia == 0)

    if ar_cz.sum() == 0:
        raise ValueError("There is no AR over Czechia")

    labeled, n = label(ar_full.values)

    labels_over_cz = np.unique(labeled[ar_cz.values])
    labels_over_cz = labels_over_cz[labels_over_cz != 0]

    mask_selected = np.isin(labeled, labels_over_cz)

    ar_selected = ar_full.copy()
    ar_selected.values = mask_selected

    if only_czechia:
        ar_selected = ar_selected & (mask_czechia == 0)


    return ar_selected



def sample_points(ar_selected, k, seed=None):
    """
    Finds random point for tracing
    :param ar_selected: mask of ar
    :param k: number of points
    :param seed: only for check
    :return: Array of k points
    """
    rng = np.random.default_rng(seed)
    mask_float = ar_selected.astype(float)

    lat_vals = ar_selected.lat.values
    lon_vals = ar_selected.lon.values
    lat_min, lat_max = lat_vals.min(), lat_vals.max()
    lon_min, lon_max = lon_vals.min(), lon_vals.max()

    points = []
    max_iter = 100  # pojistka proti nekonečné smyčce
    it = 0

    while len(points) < k and it < max_iter:
        n_try = max((k - len(points)) * 3, 50)
        lat_cand = rng.uniform(lat_min, lat_max, n_try)
        lon_cand = rng.uniform(lon_min, lon_max, n_try)

        interp_vals = mask_float.interp(
            lat=xr.DataArray(lat_cand, dims="points"),
            lon=xr.DataArray(lon_cand, dims="points"),
            method="linear",
            kwargs={"fill_value": 0.0},
        ).values

        valid = interp_vals > 0.5
        points.extend(zip(lat_cand[valid], lon_cand[valid]))
        it += 1

    if len(points) < k:
        raise RuntimeError(f"Nepodařilo se najít {k} bodů, mám jen {len(points)}")

    return np.array(points[:k])


def initiate_particles(time, use_arrival = True ,num = 100):

    file = "ERA5.ar_tag.GuanWaliser_v2.1hr.20020101-20021231.nc"

    dates = find_dates(file)
    groups = group_ars(dates)
    i = binar_search(groups, time)
    arrival = groups[i][0]
    if use_arrival:
        AR = get_points(arrival, file)
    else:
        AR = get_points(time, file)
    points = sample_points(AR, num)

    return points

#print(initiate_particles(time))

def traceback(time, num, level):

    def time_to_float(time, t_ref):
        return (time.values - t_ref) / np.timedelta64(1, "h") if hasattr(time, "values") else (time - t_ref) / np.timedelta64(1, "h")


    def read_data():

        address = "/mnt/raid1/home/choutkam/q_u_v_omega_2002_08.nc"

        quvomega = xr.open_dataset(address)
        #print(quvomega)
        q = quvomega["q"]
        u = quvomega["u"]
        v = quvomega["v"]
        omega = quvomega["w"]
        return q, u, v, omega

    q, u, v, omega = read_data()
    t_ref = q.valid_time[0].values         # referenční čas — spočti jen jednou

    t = time_to_float(q.valid_time, t_ref)
    lev = q.pressure_level.values
    lat = q.latitude.values
    lon = q.longitude.values

    print("t rozsah:", t.min(), t.max())


    q_vals = q.values.astype("float32")
    u_vals = u.values.astype("float32")
    v_vals = v.values.astype("float32")
    w_vals = omega.values.astype("float32")

    interp_q = RegularGridInterpolator(
        (t, lev, lat, lon),
        q_vals,
        bounds_error=False,
        fill_value=np.nan)

    interp_u = RegularGridInterpolator(
        (t, lev, lat, lon),
        u_vals,
        bounds_error=False,
        fill_value=np.nan)

    interp_v = RegularGridInterpolator(
        (t, lev, lat, lon),
        v_vals,
        bounds_error=False,
        fill_value=np.nan)

    interp_w = RegularGridInterpolator(
        (t, lev, lat, lon),
        w_vals,
        bounds_error=False,
        fill_value=np.nan)

    def interpolate(t, p, y, x):
        point = np.array([t, p, y, x])
        u_val = interp_u(point)[0]
        v_val = interp_v(point)[0]
        w_val = interp_w(point)[0]
        return u_val, v_val, w_val

    #t_val = time_to_float(np.datetime64(time), t0)
    #print(interpolate(t_val, 333, 67, -20))

    a = 6371000.0  # poloměr Země [m]


    def rhs(t_val, z):
        x, y, p = z
        u, v, w = interpolate(t_val, p, y, x)
        phi = np.deg2rad(y)
        dxdt = u / (a * np.cos(phi)) * 180 / np.pi * 3600
        dydt = v / a * 180 / np.pi * 3600
        dpdt = w / 100.0 * 3600   # dělení 100 jen pokud w je v Pa/s a p v hPa — ověř si units
        return [dxdt, dydt, dpdt]


    def backtrack(points, time, level = level):
        m, _ = np.shape(points)
        result = []
        for i in range(m):
            x0, y0, p0 = points[i][1], points[i][0], level  # počáteční bod
            start_date = np.datetime64(time)

            t_start = time_to_float(start_date, t_ref)
            t_end = t_start - 240.0 # zpětně 240 h


            sol = solve_ivp(rhs, [t_start, t_end], [x0, y0, p0], method="RK45",
                             max_step=1.0, atol=1e-6, rtol=1e-3, dense_output=True)
            result.append(sol)
        return result


    def interpolate_q_along_traj(t_vals, p_vals, y_vals, x_vals):
        points = np.column_stack([t_vals, p_vals, y_vals, x_vals])  # (N, 4)
        q_vals = interp_q(points)
        return q_vals.ravel()


    def compute_q_along_trajectories(result):
        """
        Computes Dq/Dt = (e-p)/m along trajectories.
        :param result: solve_ivp output
        :return: list of dictionaries {t,x,y,p,q,dq/dt}
        """
        out = []

        for sol in result:
            #t_vals = np.linspace(sol.t[0], sol.t[-1], 100)
            t_vals = sol.t
            x_vals = sol.sol(t_vals)[0]
            y_vals = sol.sol(t_vals)[1]
            p_vals = sol.sol(t_vals)[2]

            q_vals = interpolate_q_along_traj(t_vals, p_vals, y_vals, x_vals)
            q_smooth = uniform_filter1d(q_vals, size=5, mode="nearest")
            DqDt = np.gradient(q_smooth, t_vals) * 1000  # g/kg/h
            out.append({
                "t": t_vals,
                "x": x_vals,
                "y": y_vals,
                "p": p_vals,
                "q": q_vals,
                "DqDt": DqDt,})
        return out


    points = initiate_particles(time, num = num, use_arrival=False)
    solulu = backtrack(points, time)
    trajs_with_q = compute_q_along_trajectories(solulu)
    return solulu, trajs_with_q



def plot_trajectories(result, set_extent = True, save = False):
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        for sol in result:
            traj_x, traj_y = sol.y[0], sol.y[1]
            ax.plot(traj_x, traj_y, transform=ccrs.PlateCarree(), linewidth=0.1)

        if set_extent: ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
        plt.title("Back-trajectories")
        if save: plt.savefig("trajektorie_from_cz.png", dpi=300, bbox_inches="tight")
        plt.show()


def plot_trajectories_with_DqDt(trajs_with_q, set_extent=True, save=False, colour = False):
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        all_dqdt = np.concatenate([tr["DqDt"] for tr in trajs_with_q])
        vmin, vmax = np.percentile(all_dqdt,10), np.percentile(all_dqdt, 90)

        if colour:
            cmap = ListedColormap(['blue', 'red'])
            norm = BoundaryNorm([-1000, 0, 1000], cmap.N)
        else:
            cmap = "RdYlBu"
            norm = None

        for tr in trajs_with_q:
            sc = ax.scatter(tr["x"], tr["y"], c=tr["DqDt"], cmap=cmap, norm = norm,
                            vmin=vmin if norm is None else None,
                            vmax=vmax if norm is None else None,
                            s=0.5, edgecolor="none",
                            transform=ccrs.PlateCarree())

        if set_extent: ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
        ax.set_title("Back-trajectories colored by Dq/Dt")
        fig.colorbar(sc, label="Dq/Dt [g/kg/h]")
        if save: plt.savefig("trajektorie_DqDt.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    trajs, trajs_q = traceback(time, 1000, 837)
    plot_trajectories_with_DqDt(trajs_q, set_extent=False, save=True)
    plot_trajectories_with_DqDt(trajs_q, set_extent=False, save=True, colour=True)
    plot_trajectories(trajs, set_extent=False, save=True)

#e-p plot sum for different days
#composite plots