import numpy as np
import cdsapi
dataset = "reanalysis-era5-pressure-levels-monthly-means"
client = cdsapi.Client()
satellite = "Aqua"#"Terra"

if satellite=="Terra":
    local_t = 10.5
elif satellite=="Aqua":
    local_t = 13.5

for utc_hour in range(24):
    lon_min = 15 * (local_t - (utc_hour + 0.5))
    lon_max = 15 * (local_t - (utc_hour - 0.5))
    if lon_min > 180:
        lon_min = lon_min - 360 
    elif lon_min < -180:
        lon_min = lon_min + 360 

    if lon_max > 180:
        lon_max = lon_max - 360 
    elif lon_max < -180:
        lon_max = lon_max + 360

    if abs(lon_min) == 180:
        lon_min = 180 * lon_max/abs(lon_max)
    elif abs(lon_max) == 180:
        lon_max = 180 * lon_min/abs(lon_min)
    
    print(f"UTC {utc_hour:02d}:00 → 对应经度范围：{lon_min:.1f}° ~ {lon_max:.1f}°") 

    output_filename = f"era5_pl_monthly_{satellite}_{utc_hour}.nc"

    request = {
        "product_type": ["monthly_averaged_reanalysis_by_hour_of_day"],
        "variable": [
            "geopotential",
            "ozone_mass_mixing_ratio",
            "specific_humidity",
            "temperature"
        ],
        "pressure_level": [
            "10", "50", "100", 
            "200", "300", "400", "500",
            "550", "600", "650", "700", "750",
            "800", "850", "900", "925", "950", "1000"
        ],
        "year": [
            "2002", "2003", "2004", "2005", "2006",
            "2007", "2008", "2009", "2010", "2011",
            "2012", "2013", "2014", "2015", "2016",
            "2017", "2018", "2019", "2020", "2021",
            "2022", "2023"
                 ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": [
            f"{utc_hour:02d}:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [60, lon_min, -60, lon_max-0.4]
    }

    client.retrieve(dataset, request, output_filename)

