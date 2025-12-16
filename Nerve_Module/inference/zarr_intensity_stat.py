import argparse
import dask.array as da
import numpy as np
import dask


def raw_zarr_percentile(z, q, bins=10000):
    """
    Compute percentile(s) for a large Zarr/Dask array safely.

    Parameters
    ----------
    z : dask.array.Array
        Input Dask array (e.g. from da.from_zarr).
    q : float or list of floats
        Percentile(s) to compute in [0,100].
    bins : int
        Number of bins to use for histogram fallback.

    Returns
    -------
    result : float or np.ndarray
        The percentile value(s).
    """
    try:
        # Newer Dask versions support quantile
        return da.quantile(z, np.array(q) / 100.0).compute()
    except Exception as e:
        print("⚠️ Falling back to histogram method because quantile failed:", e)

        # Compute global min/max
        zmin, zmax = dask.compute(z.min(), z.max())

        # Histogram
        hist, bin_edges = da.histogram(z, bins=bins, range=(float(zmin), float(zmax)))
        hist, bin_edges = dask.compute(hist, bin_edges)

        # Cumulative distribution
        cdf = np.cumsum(hist) / hist.sum()

        results = []
        for qq in np.atleast_1d(q):
            idx = np.searchsorted(cdf, qq/100.0)
            results.append(bin_edges[min(idx, len(bin_edges)-1)])
        results = np.array(results)

        if np.isscalar(q):
            return results.item()
        return results

def main():       
    parser = argparse.ArgumentParser(description="Convert TIFF stack to chunked Zarr format")
    parser.add_argument("-i", '--input_zarr_path', help="Path of raw Zarr file", required=True)
    parser.add_argument("-p", '--percentile', type = float, default=0.1, help="Pencentile for normalization")

    args = parser.parse_args()
    z = da.from_zarr(args.input_zarr_path)
    results = raw_zarr_percentile(z, args.percentile)
    print(results)

if __name__ == '__main__':  
    main()
