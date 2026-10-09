import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ivt_calc import ivt, get_data
import matplotlib.animation as animation
from AR_stats import munch, alfons_mucha, klimt
import matplotlib.colors as mcolors


time = "2002-08-12T12:00:00.000000000"

def convert_time(time):
    str_time = str(time)
    time_list = str_time.split("-")
    last = time_list[2].split(":")[0]
    new_string = f"{time_list[0]}_{time_list[1]}_{last}"
    return new_string


def make_ax(ax, figsize=(10, 5)):
    """Returns ax, True/False if it should make new ax"""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()})
        return ax, True
    return ax, False


def multi(draw, times, cbar_label, save, filename):
    """
    Helps to plot more pics in series.
    :param draw: Specific plot function.
    :param times: List of times
    :param cbar_label: Label of color_bar
    :param save: True/False
    :param filename: If save
    :return: helper
    """
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, times):
        im = draw(t, ax)

    ims = im if isinstance(im, tuple) else (im,)
    labels = cbar_label if isinstance(cbar_label, (tuple, list)) else (cbar_label,)
    for obj, lab in zip(ims, labels):
        fig.colorbar(obj, ax=axes, shrink=0.8, pad=0.02, label=lab)

    if save:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def finish(own_fig, save, filename):
    """Saves only when own_fig"""
    if own_fig:
        if save:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()


def picasso(ivt_file, time, set_extent = True, save = False, ax = None):
    """
    For plotting stuff at one time.
    :param ivt_file: it needs dataset with selected ax
    :param time: np.datetime
    :param set_extent: default True
    :return: plot
    """
    own_fig = ax is None
    ivt_plot = ivt_file.sel(time = time)
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    data_set = ivt_plot
    im = data_set.plot(ax=ax, x="longitude", y="latitude", yincrease=True,
                       transform=ccrs.PlateCarree(),
                       add_colorbar=own_fig,
                       cbar_kwargs={"label": "IVT [kg/(m·s)]"} if own_fig else None,
                       vmin=0, vmax=1200)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth = 0.5)
    ax.set_title(f"IVT, time = {time[:13]}")
    if set_extent:
        ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    if own_fig:
        if save: plt.savefig(f"ivt_{convert_time(time)}.png", dpi=300, bbox_inches="tight")
        plt.show()
    return im



def monet(time, level, save = False, axes = None):
    """
    Plots q, u, v for specific time in 2002
    :param time: np.datetim
    :param level: only some levels
    :return: plot
    """
    quv = get_data((time, time))
    q, u, v = quv[0]
    for i, (key, val) in enumerate({"q": q, "u": u, "v": v}.items()):
        with xr.open_dataset(val) as data:
            if level not in data["level"].values:
                raise TypeError(f"Only values from this list allowed: {data['level'].values}")
            ax, own_fig = make_ax(None if axes is None else axes[i], figsize=(9, 6))
            data[key].sel(time=time, level=level).plot(ax=ax, transform=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
            ax.set_title(f"{key}, time = {str(time)[:13]}, level = {level}")
            finish(own_fig, save, f"{key}_{convert_time(time)}.png")


def rembrandt(time):
    address = "/net/meop-nas13.priv/volume1/data1/nmcrespo/ARs_Choutka/ERA5_hourlyIVT_200208.nc"
    data = xr.open_dataset(address)
    data = data.rename({
    "lat": "latitude",
    "lon": "longitude"})
    #fakt už nevím ty moje výsledky jsou dobře, netuším proč to nefunguje, ať dělám co dělám tak vždy je tenhle soubor
    picasso(data["ivt"], time)




#monet(time)
#rembrandt(time)
#picasso(ivt((time,time)), time)


def animatev2(data, name):
    """
    creates animation based on the input file
    :param file: netcdf file
    :return: it saves a gif
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth = 0.5)
    ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    mesh = data.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs={"label": "IVT [kg/(m·s)]"}, vmin=0, vmax=1200)
    title = ax.set_title(f"time = {str(data.time.isel(time=0).values)}")

    def update(frame):
        new_array = data.isel(time=frame).values
        mesh.set_array(new_array.ravel())
        title.set_text(f"time = {str(data.time.isel(time=frame).values)}")
        return mesh

    ani = animation.FuncAnimation(fig, update,frames=len(data.time), interval=5/2, blit=False)
    ani.save(name, writer="ffmpeg", fps=5)


def create_animation(time_interval, name):
    """
    Creates and saves animation of AR in year 2002.
    :param time_interval: list or tuple
    :param name: name.mp4 or name.gif
    :return: Saves animation
    """
    ivt_file = ivt((time_interval[0], time_interval[1]))
    animatev2(ivt_file,name)

#create_animation(time_interval=("2002-07-27T12:00:00.000000000","2002-08-17T12:00:00.000000000"),name="ivt.mp4")


def data_for_plot(time,level):
    """
    It gets data for plotting windfield.
    :param time: np.datetim
    :param level: only some of them available
    :return: lon, lat, U, V, norm
    """
    time_interval = (time, time)
    uv = get_data(time_interval, True)
    u, v = uv[0]
    u_data = xr.open_dataset(u)
    v_data = xr.open_dataset(v)

    if level not in u_data["level"].values:
        raise TypeError(f"Only values from this list allowed: {u_data["level"].values}")

    u_data_set = u_data["u"].sel(time=time).sel(level=level)
    v_data_set = v_data["v"].sel(time=time).sel(level=level)

    lon = u_data_set["longitude"].values
    lat = u_data_set["latitude"].values

    U = u_data_set.values
    V = v_data_set.values

    norm = np.sqrt(U ** 2 + V ** 2)

    lon = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon)
    lon = lon[sort_idx]
    U = U[:, sort_idx]
    V = V[:, sort_idx]
    norm = norm[:, sort_idx]

    return lon, lat, U, V, norm

def plot_vector_field(time, level, set_extent=True, streamline = False, save = False, ax = None, vmax = None):
    """
    Plots a vector field of wind.
    :param time: np.datetim
    :param level: only levels from the file
    :param set_extent: True/False
    :param streamline: True/False, decides whether it will plot streamlines or vector field
    :return: plots
    """
    lon, lat, U, V, norm = data_for_plot(time, level)
    ax, own_fig = make_ax(ax)
    cnorm = mcolors.Normalize(0, vmax) if vmax else None

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title(f"Wind field, time = {str(time)[:13]}, level = {level}")
    if set_extent:
        ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())

    if streamline:
        lw = 0.5 + 2.5 * (norm / norm.max())
        wind = ax.streamplot(lon, lat, U, V, color=norm, norm=cnorm,
                             transform=ccrs.PlateCarree(), density=2.5, linewidth=lw, cmap="turbo")
        im, name = wind.lines, f"streamlines_{convert_time(time)}.png"
    else:
        skip = 10
        im = ax.quiver(lon[::skip], lat[::skip], U[::skip, ::skip], V[::skip, ::skip],
                       norm[::skip, ::skip], norm=cnorm, transform=ccrs.PlateCarree(),
                       scale=1200, width=0.002, cmap="turbo")
        name = f"wind_field_{convert_time(time)}.png"

    if own_fig:
        ax.figure.colorbar(im, ax=ax, label="wind speed (m/s)", shrink=0.9)
    finish(own_fig, save, name)
    return im

#plot_vector_field(time, level=850, set_extent=True, streamline=False)
#plot streamlines or vectors of wind at different levels

def get_precipitaion_evaporation(time, eva = False):
    """
    It gets data for plotting precipitation or evaporation.
    :param time: np.datetime
    :param eva: True/False whether you want evaporation or precipitation
    :return: dataset for the time
    """
    address = "/net/meop-nas13.priv/volume1/data1/nmcrespo/ARs_Choutka/e_p_200207_200208.nc"
    data = xr.open_dataset(address)
    day = time[:10]
    if eva:
        evaporation = data["e"].sel(valid_time = day)
        daily_te = evaporation.sum(dim = "valid_time")
        output = daily_te * 1000
    else:
        precipitation = data["tp"].sel(valid_time = day)
        daily_tp = precipitation.sum(dim = "valid_time")
        output = daily_tp * 1000

    return output


def plot_e_or_p(time, eva = False, set_extent = True, save = False, ax = None):
    """
    Plots evaporation or percipitation for a specific day. It takes the day from any time.
    :param time: np.datetime
    :param eva: True/False, default False
    :param set_extent: True/False, default True
    :return: plot something maybe nice
    """
    data = get_precipitaion_evaporation(time, eva=eva)
    ax, own_fig = make_ax(ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    if eva:
        im = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="turbo", add_colorbar=own_fig)
        if set_extent:
            ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    else:
        im = data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="YlGnBu", vmin=0, vmax=20,
                       add_colorbar=own_fig,
                       cbar_kwargs={"label": "Precipitation [mm/day]"} if own_fig else None)
        if set_extent:
            ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())

    ax.set_title(f"{'Evaporation' if eva else 'Precipitation'}, time = {str(time)[:10]}")
    prefix = "e" if eva else "p"
    finish(own_fig, save, f"{prefix}_{convert_time(time)}.png")
    return im



def get_q(time, level):
    quv = get_data((time,time))
    q, _, _ = quv[0]
    q_data = xr.open_dataset(q)
    if level not in q_data["level"].values:
        raise TypeError(f"Only values from this list allowed: {q_data["level"].values}")

    return q_data["q"].sel(level = level).sel(time = time)




def plot_q_with_wind(time, level, set_extent = True, save = False, ax = None, skip=10, scale=1000, width=0.003, qmax=0.015, wmax=40):
    """
    Plots q at specific level with wind.
    :param time: np.datetime
    :param level: Only some values
    :param set_extent: True/False
    :return: Plots a figure
    """
    lon, lat, U, V, norm = data_for_plot(time, level)
    q = get_q(time, level)
    ax, own_fig = make_ax(ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    if set_extent:
        ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())

    q_im = q.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=own_fig, vmin=0, vmax=qmax)
    wind = ax.quiver(lon[::skip], lat[::skip], U[::skip, ::skip], V[::skip, ::skip],
                     norm[::skip, ::skip], transform=ccrs.PlateCarree(),
                     scale=scale, width=width, cmap="turbo",clim=(0, wmax))

    if own_fig:
        fig = ax.figure
        fig.colorbar(q_im, ax=ax, label="q [kg/kg]", shrink=0.9, pad=0.02)
        fig.colorbar(wind, ax=ax, label="wind speed (m/s)", shrink=0.9, pad=0.08)
    ax.set_title(f"Windfield and q, time = {str(time)[:13]}, level = {level}", fontsize=9)
    finish(own_fig, save, f"q_wind_{convert_time(time)}.png")
    return q_im, wind



def plot_diff_e_p(time, set_extent = True, save = False, ax = None):
    """
    Plots the difference of evaporation and precipitation.
    :param time: np.datetim
    :param set_extent: True/False
    :param save: True/False
    :param ax: technicall
    :return: plots
    """
    data_eva = get_precipitaion_evaporation(time, eva=True)
    data_precip = get_precipitaion_evaporation(time, eva=False)

    E_P = -1*data_eva - data_precip

    ax, own_fig = make_ax(ax)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    if set_extent:
        ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    im = E_P.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu", vmin=-10, vmax=10,
                  add_colorbar=own_fig,
                  cbar_kwargs={"label": "E-P [mm/day]"} if own_fig else None)
    ax.set_title(f"E-P time = {str(time)[:10]}")
    finish(own_fig, save, f"ep_{convert_time(time)}.png")
    return im


def plots(kind, time, **kwargs):
    """
    Can do all kind of plots from this file.
    :param kind: str | list[str] | tuple[str] | set[str], options: "ivt", "quv", "wind_field",
                 "streamlines", "evaporation", "precipitation", "q_wind", ...
    :param time: np.datetime, nebo list/tuple časů (několik panelů vedle sebe;
                 podporují "ivt", "wind_field", "streamlines", "evaporation",
                 "precipitation", "q_wind")
    :param kwargs:  level: tlaková hladina, výchozí 700 hPa
                    save: uložit graf, výchozí False
                    set_extent: nastavit geografický výřez, výchozí True
                    vmax: horní mez barevné škály větru při více panelech, výchozí 40 m/s
    :return: Saves or shows a plot.
    """
    if isinstance(kind, (list,tuple,set)):
        for one_kind in kind:
            plots(one_kind, time, **kwargs.copy())
        return

    level = kwargs.pop("level", 700)
    save = kwargs.pop("save", False)
    extent = kwargs.pop("set_extent", True)
    vmax = kwargs.pop("vmax", 40)
    precip = kwargs.pop("precip",False)

    if isinstance(time, (list, tuple)):
        times = list(time)
        name = f"{kind}_{convert_time(times[0])}_{convert_time(times[-1])}.png"

        if kind == "ivt":
            data = ivt((min(times), max(times)))
            draw = lambda t, ax: picasso(data, t, set_extent=extent, ax=ax)
            label = "IVT [kg/(m·s)]"
        elif kind == "precipitation":
            draw = lambda t, ax: plot_e_or_p(t, eva=False, set_extent=extent, ax=ax)
            label = "Precipitation [mm/day]"
        elif kind == "evaporation":
            draw = lambda t, ax: plot_e_or_p(t, eva=True, set_extent=extent, ax=ax)
            label = "Evaporation"
        elif kind in ("wind_field", "streamlines"):
            draw = lambda t, ax: plot_vector_field(t, level, set_extent=extent,
                                                   streamline=(kind == "streamlines"),
                                                   ax=ax, vmax=vmax)
            label = "wind speed (m/s)"
        elif kind == "q_wind":
            draw = lambda t, ax: plot_q_with_wind(t, level, set_extent=extent, ax=ax)
            label = ("q [kg/kg]", "wind speed (m/s)")
        elif kind == "ep":
            draw = lambda t, ax: plot_diff_e_p(t, set_extent=extent, ax=ax)
            label = "mm/day"
        else:
            raise ValueError(f"Druh grafu '{kind}' nepodporuje více časů!")

        multi(draw, times, label, save, name)
        return

    if kind == "ivt":
        data = ivt((time, time))
        picasso(data, time, set_extent=extent, save = save)

    elif kind == "quv":
        monet(time, level, save = save)

    elif kind == "wind_field":
        plot_vector_field(time, level,set_extent=extent ,save = save, streamline = False)

    elif kind == "streamlines":
        plot_vector_field(time, level, set_extent=extent ,save = save, streamline = True)

    elif kind == "evaporation":
        plot_e_or_p(time, set_extent=extent, save = save, eva = True)

    elif kind == "precipitation":
        plot_e_or_p(time, set_extent=extent, save = save, eva = False)

    elif kind == "q_wind":
        plot_q_with_wind(time, level, set_extent=extent, save = save)

    elif kind == "IVT_composite_czechia":
        munch(set_extent=extent, save = save)

    elif kind == "IVT_composite_2002":
        munch(whole_year=True, set_extent=extent, save = save)

    elif kind == "AR_count":
        alfons_mucha(save_fig=save, precip=precip)

    elif kind == "q_distribution":
        klimt(time, savefig = save)

    elif kind == "ep":
        plot_diff_e_p(time, set_extent=extent, save = save)
    else:
        raise ValueError(f"Neznámá druh grafu {kind}!")


#plots(["ivt", "wind_field", "q_wind", "precipitation", "IVT_composite_czechia", "streamlines",
# "AR_count", "quv"], time, level=850)

times = ["2002-08-05T12:00:00", "2002-08-06T12:00:00", "2002-08-07T12:00:00"]

#plots("precipitation", times, set_extent=False)                               # tři IVT vedle sebe
#plots("ep", times, save=False, set_extent=False)             # tři srážkové mapy, uloží se
#plots(["ivt", "wind_field"], times, level=850)       # dvě samostatné figury, každá s třemi panely
#plots("ivt", times[0])
plots(["AR_count"], time, precip=True, save=True)
# jeden čas jako dřív

