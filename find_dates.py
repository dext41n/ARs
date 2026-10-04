import numpy as np
import xarray as xr
import regionmask
import cartopy.io.shapereader as shpreader

soubor = "ERA5.ar_tag.GuanWaliser_v2.1hr.20020101-20021231.nc"

def prep(file):
    """
    opens the file and makes a box of it around Czechia
    :param file: netcdf file
    :return:
    """
    data = xr.open_dataset(file)
    var = list(data.data_vars)[0]
    data_subset = data[var].sel(lat=slice(47, 52), lon=slice(10, 20))
    #data_subset = data_subset.isel(time = data.time.dt.hour == 12)
    #print(data)
    return data_subset


def read_czechia(data):
    """
    It reads countour of Czechia and returns mask
    :return: region mask
    """
    shpfilename = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    czechia = [c.geometry for c in reader.records() if c.attributes["NAME"] == 'Czechia'][0]
    #vytvořit masku
    region = regionmask.Regions([czechia], names=["Czechia"])
    mask = region.mask(data)
    return mask


def find_dates(file = soubor):
    """
    It finds the times from IVT file in which we can see atmospheric river over Czechia.
    :param file: netcdf file
    :return: time values array
    """
    data_subset = prep(file)
    mask = read_czechia(data_subset)
    data_czechia = data_subset.where(mask == 0)
    ar_czechia = data_czechia.max(dim=["lon","lat"])
    times_ar = data_subset.time.where(ar_czechia == 1, drop = True)
    return times_ar.values


def group_ars(dates, tol = 12):
    """
    Seperates dates with given tol into groups.
    :param dates: array of dates (np.datetime)
    :param tol: the highest value between two dates to be considered same event
    :return: list of groups of dates
    """
    #jedna hodina je málo, zkusit třeba 12
    groups = []
    new_group = []
    prev = None
    for date in dates:
        if prev is None:
            new_group.append(date)
            prev = date
        else:
            diff = date - prev
            if diff < np.timedelta64(tol, "h"):
                new_group.append(date)
                prev = date
            else:
                groups.append(new_group)
                new_group = []
                prev = date

    if dates[-1] - dates[-2] == np.timedelta64(12, "h"):
        groups.append(new_group)

    not_empty = [group for group in groups if group]
    return not_empty


def lies_in(point ,start_point, end_point):
    if start_point <= point <= end_point:
        return True
    else:
        return False


def binar_search(groups, time):
    """
    Finds a group where the given times should be in.
    :param groups: list of groups
    :param time: np.datetime format
    :return: list of times of the whole group
    """
    time = np.datetime64(time, "D")

    left = 0
    right = len(groups) - 1

    while left <= right:
        mid = (left + right) // 2

        if groups[mid][-1] < time:
            left = mid + 1
        else:
            right = mid - 1

    return left

def test():
    dates = find_dates(soubor)
    skupiny = group_ars(dates)
    print(skupiny)
    print(len(skupiny))
    čas = np.datetime64("2002-08-11T12:00:00.000000000")
    num = binar_search(skupiny, čas)
    print(skupiny[num])
    print(len(skupiny[num]))
    print(lies_in(čas,skupiny[num][0], skupiny[num][-1]))



if __name__ == "__main__":
    test()

