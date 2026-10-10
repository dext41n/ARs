import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
from matplotlib.patches import Patch


file = "ERA5.ar_tag.GuanWaliser_v2.1hr.20020101-20021231.nc"
time = "2002-08-11T12:00:00.000000000"
cmap = ListedColormap(["white","blue"])

def print_time(file,time,info=True):
    """
    Only opens the file and selects given time on time ax
    :param file: netcdf file
    :param time: np.datetime works
    :param info: True/False
    :return: dataset for the one moment
    """
    data = xr.open_dataset(file)                #otevře soubor
    var = list(data.data_vars)[0]
    data_fixed_time = data[var].sel(time = time)
    if info:
        print(data)                 #ukáže informace o souboru
    return data_fixed_time


def data_daily(file):
    #useless
    data = xr.open_dataset(file)
    var = list(data.data_vars)[0]
    daily = data[var].isel(time = data.time.dt.hour == 12)
    return daily

def davinci(file, time, save = False):
    """
    Plots and prints information of dataset for specific time
    :param file: netcdf
    :param time: np.datetime
    :return: plots
    """
    fig, ax = plt.subplots(figsize=(9,6),subplot_kw={"projection": ccrs.PlateCarree()})
    data_test = print_time(file,time)
    data_test.plot(ax=ax, transform=ccrs.PlateCarree(), cmap = cmap, add_colorbar = False, label = "AR")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth = 0.5)
    ax.set_extent([-90, 60, -10, 80], crs=ccrs.PlateCarree())
    ax.set_title(f"AR detection, time = {time[:13]}")
    ax.legend(handles=[Patch(facecolor="blue", edgecolor="k", label="atmospheric river")],
              loc="lower left")
    if save: plt.savefig("ar_detection.png", dpi = 300)
    plt.show()


def animate(data, name):
    """
    creates animation based on the input file
    :param file: netcdf file
    :return: it saves a gif
    """
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth = 0.5)
    ax.set_extent([-40, 60, 15, 80], crs=ccrs.PlateCarree())
    mesh = data.isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap = cmap, add_colorbar = False)
    title = ax.set_title(f"time = {str(data.time.isel(time=0).values)}")

    def update(frame):
        new_array = data.isel(time=frame).values
        mesh.set_array(new_array.ravel())
        title.set_text(f"time = {str(data.time.isel(time=frame).values)}")
        return mesh

    ani = animation.FuncAnimation(fig, update,frames=len(data.time), interval=5/2, blit=False)
    ani.save(name, writer="pillow", fps=5)
    plt.show()

if __name__ == "__main__":
    davinci(file,time, save=True)

#animate(file)