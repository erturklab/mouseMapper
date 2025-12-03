import argparse
from pathlib import Path
import sys
import shutil

import dask
import dask.array as da
from dask.array.image import imread as imread_dask
from dask.array import rechunk


print("Python executable:", sys.executable)
print("Dask version:", dask.__version__)

def parse_chunk_size(value):
    """Parse chunk size from string like '64,256,256' into tuple."""
    try:
        return tuple(int(x.strip()) for x in value.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid chunk size: {value}. Use format: 64,256,256"
        )
    
def tif2zarr(tiff_dir, zarr_path, chunk_size = (64, 256, 256)):
    
    tiff_dir = Path(tiff_dir)
    zarr_path = Path(zarr_path)
    
    # Validate input
    tif_files = list(tiff_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {tiff_dir}")
    
    print(f"Found {len(tif_files)} TIFF files")
    
    # Read TIFFs lazily and convert them into a zarr file under intermediate zarr path
    # This step will save every slice as one chunk
    im = imread_dask(str(tiff_dir / "*.tif"))
    print(f"Input array shape: {im.shape}, dtype: {im.dtype}")
    print(f"Input chunk sizes: {im.chunksize}")
    inter_zarr_path = zarr_path.with_name(zarr_path.stem + "_prep.zarr")
    im.to_zarr(str(inter_zarr_path), overwrite=True)
    
    # rechunk the intermediate zarr file into smaller chunks for inference
    inter_zarr = da.from_zarr(str(inter_zarr_path))
    rechunked = rechunk(inter_zarr, chunks=chunk_size)
    rechunked.to_zarr(zarr_path, overwrite=True)
      
    print(f"Output zarr shape: {rechunked.shape}")
    print(f"Output chunk sizes: {rechunked.chunksize}")
    
   
    shutil.rmtree(inter_zarr_path)

def main():       
    parser = argparse.ArgumentParser(description="Convert TIFF stack to chunked Zarr format")
    parser.add_argument("-i", '--input_tiff_path', help="Directory of raw tiff slices", required=True)
    parser.add_argument("-o", '--output_zarr_path', help="Output zarr path", required=True)
    parser.add_argument("-c", '--chunk_size', type=parse_chunk_size, default = (64, 256, 256), help="Chunk size as 'z,y,x' (default: 64,256,256)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of Dask workers (default: 4)")
    args = parser.parse_args()
    
    tiff_dir = args.input_tiff_path
    zarr_path = args.output_zarr_path
    chunk_size = args.chunk_size
    with dask.config.set(scheduler='threads', num_workers=args.workers):
        tif2zarr(tiff_dir, zarr_path, chunk_size)


if __name__ == '__main__':  
    main()
    
