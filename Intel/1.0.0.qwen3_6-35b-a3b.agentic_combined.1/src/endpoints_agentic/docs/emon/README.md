# EMON setup

Install Intel VTune/SEP with EMON at
`/opt/intel/oneapi/vtune/latest/bin64/emon`.
Load the drivers and grant access if needed:

```bash
sudo /opt/intel/oneapi/vtune/latest/sepdk/src/insmod-sep -g vtune
sudo usermod -aG vtune "$USER"
```

Log out and back in after changing group membership, then verify driver access:

```bash
source /opt/intel/oneapi/vtune/latest/sep_vars.sh
emon -v
```

PMU-in-use warnings can affect counter accuracy.
