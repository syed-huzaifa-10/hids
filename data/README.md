# Data sources

| Item | Detail |
|---|---|
| Dataset | **NSL-KDD** (improved version of KDD Cup 1999) |
| Files used | `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows) - two public files, combined |
| Original authors | M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, *A Detailed Analysis of the KDD CUP 99 Data Set*, IEEE CISDA 2009 (Canadian Institute for Cybersecurity, Univ. of New Brunswick) |
| Mirror used for download | https://github.com/defcom17/NSL_KDD (raw files) |
| Collection method | Downloaded with `data/download_data.py` (plain HTTP download of public files) |
| Date accessed | 22 November 2026 |
| Licence / terms | Published by UNB/CIC for research use; cite Tavallaee et al. (2009) when used |
| Personal identifiers | None. Records contain only connection/host statistics; no IP addresses or user names |
| Size after combining + de-duplicating | 147,888 rows x 41 features (+ label) |

## Columns
41 features: 3 categorical (`protocol_type`, `service`, `flag`) and 38 numeric. They fall into
(1) basic connection features, (2) **content / host-level features** (e.g. `num_failed_logins`, `root_shell`,
`su_attempted`, `num_shells`, `num_file_creations`) and (3) time- and host-based traffic statistics
(`count`, `dst_host_*`). The label names ~40 attack types, mapped in `src/hids_core.py` to five classes:
Normal, DoS, Probe, R2L, U2R.

The raw `.txt` files are not committed if the repository size matters; run `python data/download_data.py` to fetch them.
