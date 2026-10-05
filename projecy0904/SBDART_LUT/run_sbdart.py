# -*- coding: utf-8 -*-
"""
Build the SBDART COT-SZA albedo lookup table for one model configuration.

Four independent switches select the configuration (each 'dcp' or 'cp' for
the first three, and 'vis'/'sw' for the band).  The dcp grid (single
representative ocean/season) is used, so one run produces one LUT:

    GAS  : 'dcp' -> atms_profile_gasdcp/atms_{ocean}_{season}.dat
           'cp'  -> atms_profile_cp/atms_{ocean}_{season}.dat
    AOD  : 'dcp' -> default INPUT TBAER
           'cp'  -> mean of the aod_mod08 column of
                    L3_product/{ocean}_{season}.csv
    SFC  : 'dcp' -> ALBCON = 0
           'cp'  -> ALBCON = sfc_albedo_results.csv.loc[ocean, season]
    BAND : 'vis' -> wlinf = 0.4, wlsup = 0.7
           'sw'  -> wlinf = 0.3, wlsup = 5.0

Output: gas{GAS}_aod{AOD}_sfc{SFC}_{BAND}/albedo_cot_cer_sza_LUT_{ocean}_{season}.nc
        (netCDF, dims sza x nre x cot)
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path('/home/chenyiqi/251028_albedo_cot/project0904')
L3_DIR = BASE_DIR / 'L3_product'
SFC_ALBEDO_CSV = SCRIPT_DIR / 'sfc_albedo_results.csv'

# Path of the INPUT file edited by modify_input (rebound to out_dir/INPUT in main).
INPUT_FILE = SCRIPT_DIR / 'INPUT'
SBDART_BIN = '/home/chenyiqi/SBDART-master/sbdart'


def modify_input(tval=None, sz=None, ALBCON=None, TBAER=None, wlinf=None, wlsup=None, nre=None, input_path=None):
    """Modify one or more parameters in the SBDART INPUT file.

    Any subset of parameters may be given; every provided parameter is
    updated in the same pass, so several INPUT entries can be combined.
    """
    path = Path(input_path) if input_path is not None else INPUT_FILE
    with open(path, 'r') as f:
        content = f.read()

    replacements = [
        ('tcloud', tval, '{:.1f}'),
        ('sza', sz, '{:.1f}'),
        ('ALBCON', ALBCON, '{:.4f}'),
        ('TBAER', TBAER, '{:.4f}'),
        ('wlinf', wlinf, '{:.3f}'),
        ('wlsup', wlsup, '{:.3f}'),
        ('nre', nre, '{:.0f}'),
    ]

    changed = False
    for name, value, fmt in replacements:
        if value is None:
            continue
        pattern = rf'\b{name}\s*=\s*-?\d+(?:\.\d+)?'
        content = re.sub(pattern, f'{name}={fmt.format(value)}', content)
        changed = True

    if not changed:
        print('modify_input: no modification requested')

    with open(path, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Configuration switches (each 'dcp' or 'cp'); BAND is 'vis' or 'sw'
    # ------------------------------------------------------------------
    GAS = 'cp'    # gas profile: 'dcp' (gasdcp) or 'cp' (gascp)
    AOD = 'cp'    # aerosol AOD:  'dcp' (aoddcp) or 'cp' (aodcp)
    SFC = 'dcp'    # surface albedo: 'dcp' (sfcdcp) or 'cp' (sfccp)
    BAND = 'sw'   # spectral band: 'vis' or 'sw'

    # run grid: dcp -> single representative ocean/season; cp -> 8 oceans x 4 seasons
    if GAS == 'dcp':
        oceans = ['TPO']
        season_dict = {'MAM': [3, 4, 5]}
    elif GAS == 'cp':
        oceans = ['SPO', 'SAO', 'SIO']#'NPO', 'NAO', 'TPO', 'TAO', 'TIO', 
        season_dict = {
            'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],
            'SON': [9, 10, 11],
            'DJF': [12, 1, 2],
        }
    else:
        raise ValueError(f"Unknown GAS option: {GAS}")

    gas_folder = 'atms_profile_gasdcp' if GAS == 'dcp' else 'atms_profile_cp'

    if BAND == 'vis':
        wlinf, wlsup = 0.4, 0.7
    elif BAND == 'sw':
        wlinf, wlsup = 0.3, 5.0
    else:
        raise ValueError(f"Unknown BAND: {BAND}")

    out_dir = SCRIPT_DIR / f'gas{GAS}_aod{AOD}_sfc{SFC}_{BAND}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Everything runs inside out_dir: INPUT / atms.dat / output.txt live there.
    INPUT_FILE = out_dir / 'INPUT'
    shutil.copyfile(SCRIPT_DIR / 'INPUT', INPUT_FILE)
    target_link = out_dir / 'atms.dat'
    output_file = out_dir / 'output.txt'
    df_a = pd.read_csv(SFC_ALBEDO_CSV, index_col=0)

    for ocean in oceans:
        for season in season_dict.keys():
            # ---- gas: choose the atmospheric profile ----
            source_file = SCRIPT_DIR / gas_folder / f'atms_{ocean}_{season}.dat'
            if target_link.is_symlink() or target_link.exists():
                target_link.unlink()
            os.link(source_file, target_link)
            print(f"atms_dat -> {source_file.name}")

            # ---- aod: corrected -> mean aod_mod08 from L3; default -> DEFAULT_AOD ----
            if AOD == 'cp':
                l3 = pd.read_csv(L3_DIR / f'{ocean}_{season}.csv')
                tbaer = float(l3['aod_mod08'].mean())
            elif AOD == 'dcp':
                tbaer = 0.0
            else:
                raise ValueError(f"Unknown AOD option: {AOD}")

            # ---- sfc: corrected -> L3 surface albedo; default -> 0 ----
            if SFC == 'cp':
                albcon = float(df_a.loc[ocean, season])
            elif SFC == 'dcp':
                albcon = 0.0
            else:
                raise ValueError(f"Unknown SFC option: {SFC}")

            # setup that does not change with sza / tcloud: apply once
            modify_input(ALBCON=albcon, TBAER=tbaer, wlinf=wlinf, wlsup=wlsup)

            tcloud_values = np.exp(np.linspace(np.log(0.03), np.log(180), 40))
            sz_values = np.arange(0, 76.1, 4)
            nre_values = np.arange(4, 30, 3.5)

            # 3D LUT: albedo[sza, nre, cot]
            albedo_grid = np.full(
                (len(sz_values), len(nre_values), len(tcloud_values)), np.nan
            )

            for nre_idx, nre in enumerate(nre_values):
                print(f"Processing nre = {int(nre)}...")
                modify_input(nre=nre)

                for sz_idx, sz in enumerate(sz_values):
                    print(f"  Processing sz = {sz:.1f}...")
                    modify_input(sz=sz)

                    for tval_idx, tval in enumerate(tcloud_values):
                        modify_input(tval=tval)

                        # run sbdart inside out_dir,
                        # redirecting its console output to out_dir/output.txt
                        with open(output_file, 'w') as fout:
                            subprocess.run(
                                SBDART_BIN,
                                check=True, cwd=str(out_dir),
                                stdout=fout, stderr=subprocess.STDOUT, text=True,
                            )

                        # parse SBDART's inband summary line (iout=10):
                        #   wlinf wlsup phidw topdn topup topdir botdn botup botdir
                        with open(output_file, 'r') as f:
                            lines = [ln.strip() for ln in f if ln.strip()]
                        parts = lines[-1].split()
                        val1 = float(parts[3])   # topdn (downward flux at TOA)
                        val2 = float(parts[4])   # topup (upward flux at TOA)
                        albedo_grid[sz_idx, nre_idx, tval_idx] = val2 / val1

            # save as netCDF, dims (sza, nre, cot)
            ds = xr.Dataset(
                {'albedo': (('sza', 'nre', 'cot'), albedo_grid)},
                coords={
                    'sza': sz_values,
                    'nre': nre_values,
                    'cot': tcloud_values,
                },
                attrs={
                    'ocean': ocean,
                    'season': season,
                    'GAS': GAS,
                    'AOD': AOD,
                    'SFC': SFC,
                    'BAND': BAND,
                },
            )
            out_nc = out_dir / f'albedo_cot_cer_sza_LUT_{ocean}_{season}.nc'
            ds.to_netcdf(out_nc)
            print(f"Saved: {out_nc}")

