import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ivt_calc import ivt, get_data
from find_dates import find_dates, group_ars, read_czechia
from trace_back import get_points

time = "2002-08-11T06:00:00.000000000"

def get_precipitaion_evaporation(time, eva = False):
    """
    It gets data for plotting precipitation or evaporation.
    :param time: np.datetime
    :param eva: True/False whether you want evaporation or precipitation
    :return: dataset for the time
    """
    address = "/net/meop-nas13.priv/volume1/data1/nmcrespo/ARs_Choutka/e_p_200207_200208.nc"
    data = xr.open_dataset(address)
    if eva:
        output = data["e"].sel(valid_time = time)
    else:
        day = time[:10]
        precipitation = data["tp"].sel(valid_time = day)
        daily_tp = precipitation.sum(dim = "valid_time")
        output = daily_tp * 1000

    return output


def precipitation_after_AR(time, days=2, threshold = 5):
    """
    Decides whether there occured a strong daily precipitation over Czechia after selected date
    :param time: np.datetime
    :param days: How many days after AR hitting CZ will be checked, default 2
    :param threshold: threshold for precipitation, default 5
    :return: True/False
    """
    time = np.datetime64(time, "D")
    mask = None

    for i in range(days + 1):
        day = np.datetime_as_string(time + np.timedelta64(i, "D"), unit="D")
        tp = (get_precipitaion_evaporation(day)
              .rename({"latitude": "lat", "longitude": "lon"})
              .sel(lat=slice(52, 47), lon=slice(10, 20)))

        if mask is None:
            mask = read_czechia(tp).notnull()

        mean_tp = tp.where(mask).mean(dim=("lat", "lon"))
        if (mean_tp > threshold).any():
            return True
    return False



def monthly_ar_sum(groups, precip = False):
    """
    Sums up ARs every month, if AR is included in more monhts, it will be counted in both.
    Also counts if there was a strong rain after the AR.
    :param groups: list of ARs
    :return: list of 12 integers.
    """
    monthly_sum = np.zeros(12, dtype=int)
    monthly_precipitation_sum = np.zeros(12, dtype=int)

    for event in groups:
        first = np.datetime64(event[0])
        last = np.datetime64(event[-1])
        interval = np.array([first, last])
        months = interval.astype("datetime64[M]").astype("int32")
        if precip:
            strong_rain = precipitation_after_AR(last, days=3)
        monts = months%12
        first, last = monts[0],monts[-1]
        for i in range(first,last+1):
            monthly_sum[i] += 1
            if precip:
                if strong_rain:
                    monthly_precipitation_sum[i] += 1
    return monthly_sum, monthly_precipitation_sum



def alfons_mucha(save_fig = False, precip = False):
    """
    Plots a graph of number of ARs in different months.
    :return: plot
    """
    dates = find_dates("ERA5.ar_tag.GuanWaliser_v2.1hr.20020101-20021231.nc")
    groups = group_ars(dates)
    monthly_sum, monthly_precip = monthly_ar_sum(groups, False)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize = (8,6))
    colors = [
        "#1f77b4",  # modrá
        "#ff7f0e",  # oranžová
        "#2ca02c",  # zelená
        "#d62728",  # červená
        "#9467bd",  # fialová
        "#8c564b",  # hnědá
        "#e377c2",  # růžová
        "#7f7f7f",  # šedá
        "#bcbd22",  # olivová
        "#17becf",  # tyrkysová
        "#aec7e8",  # světle modrá
        "#ffbb78",  # světle oranžová
    ]
    ax.bar(months, monthly_sum, color = colors)
    if precip:
        ax.bar(months, monthly_precip, width=0.4, color="navy", label="AR with precipitation")
    ax.set_title("Number of ARs in different months")
    ax.set_ylabel("Number of ARs")
    if save_fig: plt.savefig("monthly_arsum.png", dpi = 300)
    plt.show()



def IVT_composite():
    """
    Counts IVT for whole year and makes and average.
    :return: saves IVT composite
    """
    save_address = "/home/choutkam/ARs/"

    ivt_files = []

    for i in range (1,13):
        start = pd.Timestamp(year=2002, month=i, day=1)
        end = start + pd.offsets.MonthEnd()
        time_interval = (str(start), str(end))
        ivt_month = ivt(time_interval)
        file_name = f"{save_address}ivt_month_{i:02d}.nc"
        ivt_month.name = "ivt"
        ivt_month.attrs["units"] = "kg m-1 s-1"
        ivt_month.attrs["long_name"] = "Integrated vapor transport"
        ivt_month.to_netcdf(file_name)
        ivt_month.to_netcdf(file_name)
        ivt_files.append(file_name)
        print(f"Měsíc {i:02d} hotov.")

    ivt_year = xr.open_mfdataset(ivt_files, combine="by_coords", chunks={"time": 1})
    composite = ivt_year.mean(dim="time")
    composite.to_netcdf(f"{save_address}ivt_composite_2002.nc")



def IVT_composite_specific_days(AR_groups=None):
    """
    Counts the composite only for AR days over Czechia
    :param AR_groups: List of AR dates groups, or None, default None
    :return:
    """
    if AR_groups is None:
        ar_dates = find_dates()
        dates = group_ars(ar_dates)
    else:
        dates = AR_groups

    save_address = "/home/choutkam/ARs/"
    print(len(dates))

    total, count = None, None

    for i in dates:
        start = pd.Timestamp(i[0]).normalize()
        end = pd.Timestamp(i[-1]).normalize()
        ivt_month = ivt((str(start), str(end)))  # jeden AR event

        s = ivt_month.sum(dim="time", skipna=True)
        c = ivt_month.count(dim="time")
        total = s if total is None else total + s
        count = c if count is None else count + c

    composite = (total / count)
    composite.name = "ivt"
    composite.attrs["units"] = "kg m-1 s-1"
    composite.to_netcdf(f"{save_address}ivt_composite_AR_days.nc")
    return composite



def q_distribution(time):
    """
    Randomly selects a point from grid inside czechia
    :param time: np.datetime
    :return: lat, lon
    """
    ar_czechia = get_points(time, file = "ERA5.ar_tag.GuanWaliser_v2.1hr.20020101-20021231.nc")
    ys, xs = np.where(ar_czechia == 1)
    rng = np.random.randint(len(ys))
    y,x = ys[rng], xs[rng]
    lat = ar_czechia.lat.values[y]
    lon = ar_czechia.lon.values[x]
    return lat, lon


def klimt(time, savefig = False):
    """
    Klimt shows for what p we have most q.
    :param time: time with AR over Czechia.
    :param savefig:
    :return: plot
    """
    lat, lon = q_distribution(time)
    q_address, _, _ = get_data((time,time))[0]
    q = xr.open_dataset(q_address)
    q_plot = q.sel(time = time).sel(latitude = lat, longitude = lon)
    q_vals = q_plot["q"].values
    level_vals = (q.level.values)

    fig, ax = plt.subplots(figsize = (8,6))
    ax.plot(q_vals, level_vals, marker = "o")
    ax.invert_yaxis()
    ax.set_xlabel("Specific humidity q [kg/kg]")
    ax.set_ylabel("Pressure level [hPa]")
    ax.set_title(f"Vertical q profile at {lat:.2f}°N, {lon:.2f}°E\n{time}")
    if savefig: plt.savefig("q_distribution.png", dpi = 300)
    plt.show()



def munch(whole_year = False ,save = False, set_extent = True):
    """
    Plots the IVT composite.
    :param composite_file: The composite file
    :param save: True/False
    :param set_extent: True/False
    :return: plots
    """
    if whole_year:
        composite_file = "/home/choutkam/ARs/ivt_composite_AR_days.nc"
    else:
        composite_file = "/home/choutkam/ARs/ivt_composite_2002.nc"
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    data = xr.open_dataset(composite_file)
    data = data["ivt"]
    data.plot(ax=ax, transform = ccrs.PlateCarree(),cbar_kwargs={"label": "IVT [kg/(m·s)]"})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_title("IVT composite 2002")
    if set_extent:
        ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    if save: plt.savefig(f"ivt_composite.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    munch()
    klimt(time)
    alfons_mucha()