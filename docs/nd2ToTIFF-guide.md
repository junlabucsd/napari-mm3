# nd2ToTIFF
This widget takes an .nd2 file from Nikon Elements and makes individual TIFFs for every FOV and time point. Colors are saved into a stack for that timepoint (multi-page TIFF). 

The script uses the [pims_nd2](https://github.com/soft-matter/pims_nd2) package based off the Nikon SDK to be able to open .nd2 and read the metadata. Saves TIFF files using the package tifffile.py (contained in the `external_lib/` directory. 

**Input**
* .nd2 file as produced by Nikon Elements

**Output**
* Individual TIFF files. 

## Notes on use

mm3_nd2ToTIFF.py reads the metadata directly from the .nd2 and then writes it into the header of the TIFF file when saving. The format is a json representation of a Python dictionary, and is recognized later by Compile.  
