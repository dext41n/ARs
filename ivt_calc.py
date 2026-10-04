import numpy as np
import xarray as xr
from scipy.integrate import simpson

g = 9.80665

def get_data(time, only_wind=False):
    """
    It gets all adresses of needed files.
    :param time: An interval
    :param only_wind: True/False
    :return: list of triplets of adresses
    """

    base = "/net/meop-nas13.priv/volume1/data1/nmcrespo/ARs_Choutka/"
    date1, date2 = str(time[0]), str(time[1])
    month1 = date1.split("-")[1]
    month2 = date2.split("-")[1]
    m1_int, m2_int = int(month1), int(month2)
    result = []
    if only_wind:
        for month in range(m1_int, m2_int +1):
            month_str = f"0{month}" if month < 10 else f"{month}"
            u = f"{base}sel_uwnd_2002_{month_str}.nc"
            v = f"{base}sel_vwnd_2002_{month_str}.nc"
            result.append((u,v))
    else:
        for month in range(m1_int, m2_int +1):
            month_str = f"0{month}" if month < 10 else f"{month}"
            q = f"{base}sel_qhum_2002_{month_str}.nc"
            u = f"{base}sel_uwnd_2002_{month_str}.nc"
            v = f"{base}sel_vwnd_2002_{month_str}.nc"
            result.append((q, u, v))
    return result


def fix_lon(file):
    """
    fixes longitude to the format used by me(-180,180)
    :param file: dataset
    :return: fixed dataset
    """
    return file.assign_coords(longitude=(((file.longitude + 180) % 360) - 180)).sortby("longitude")


def prep_data(time, files, lat=[80, -10], lon=[270, 60]):
    """
    Puts files for different months together and cuts only a box of them.
    :param time: interval
    :param files: only adresses from list
    :param lat: format -90,90
    :param lon: format 0,360
    :return: q, u, v
    """
    qs, us, vs = [], [], []

    for q_data, u_data, v_data in files:
        q = xr.open_dataset(q_data)
        u = xr.open_dataset(u_data)
        v = xr.open_dataset(v_data)
        #takhle by to mohlo jet rychleji, než přejmenovávat souřadnic obřích souborů

        q1 = q.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(lon[0], 360))
        q2 = q.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(0, lon[1]))
        u1 = u.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(lon[0], 360))
        u2 = u.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(0, lon[1]))
        v1 = v.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(lon[0], 360))
        v2 = v.sel(time=slice(time[0], time[1]), latitude=slice(lat[0], lat[1]), longitude=slice(0, lon[1]))

        q = xr.concat([q1, q2], dim="longitude").sortby("longitude")
        u = xr.concat([u1, u2], dim="longitude").sortby("longitude")
        v = xr.concat([v1, v2], dim="longitude").sortby("longitude")

        qs.append(q)
        us.append(u)
        vs.append(v)

    q_final = xr.concat(qs, dim="time")
    u_final = xr.concat(us, dim="time")
    v_final = xr.concat(vs, dim="time")

    q_final = fix_lon(q_final)
    u_final = fix_lon(u_final)
    v_final = fix_lon(v_final)

    return q_final, u_final, v_final


def count_ivt(q,u,v):
    """
    For given q, u, v, counts IVT. Uses Simpson rule for integration.
    :param q: q specific humidity
    :param u: x-wind
    :param v: y-wind
    :return: IVT dataset
    """
    q_var = q[list(q.data_vars)[0]]
    u_var = u[list(u.data_vars)[0]]
    v_var = v[list(v.data_vars)[0]]

    pressure_pa = q_var.level.values * 100
    qu = q_var * u_var
    qv = q_var * v_var

    level_axis = qu.get_axis_num("level")
    ivt_u_vals = simpson(qu.values, pressure_pa, axis=level_axis)
    ivt_v_vals = simpson(qv.values, pressure_pa, axis=level_axis)

    remaining_dims = [d for d in qu.dims if d != "level"]
    remaining_coords = {d: qu.coords[d] for d in remaining_dims}

    ivt_u = xr.DataArray(ivt_u_vals, dims=remaining_dims, coords=remaining_coords)
    ivt_v = xr.DataArray(ivt_v_vals, dims=remaining_dims, coords=remaining_coords)

    ivt = np.sqrt((ivt_u/g)**2 + (ivt_v/g)**2)

    return ivt


def ivt(time):
    """
    For given time interval from 2002 calculates IVT.
    :param time: Time interval
    :return: IVT dataset
    """
    files = get_data(time)
    q, u, v = prep_data(time, files)
    result = count_ivt(q,u,v).sortby("latitude")
    #print(result)
    return result


if __name__ == "__main__":
    time = ("2002-07-29T12:00:00.000000000","2002-08-16T12:00:00.000000000")
    ivt_file = ivt(time)


#srovnat s tím spočítaným někým jiným
#plotnout q, normu celýho vektoru větru
