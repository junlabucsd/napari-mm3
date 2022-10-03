## Generating a TIFF stack 

**Input**
* .nd2 file as produced by Nikon Elements

**Output**
* Individual TIFF files. 

mm3 currently currently takes individual TIFF images as its input. 
If there are multiple color layers, then each TIFF image should be a stack of planes corresponding to a color. 
The quality of your images is important for mm3 to work properly.

The working directory now contains:
```
.
├── 20170720_SJ388_mopsgluc12aa.nd2
├── TIFF
```

### Notes on metadata

mm3_nd2ToTIFF.py reads the metadata directly from the .nd2 and then writes it into the header of the TIFF file when saving. The format is a json representation of a Python dictionary, and is recognized later by Compile.  


