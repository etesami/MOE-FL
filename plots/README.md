## Plot the results of the study
To automatically plot all figures and save them as pdf file edit the **plot-all-single-line-manually.sh** and run it. This file runs **plot.sh** when it is called.
```
# manually edit the file plot-all-single-line-manually.sh
./plot-all-single-line-manually.sh
```

To plot merged figures (ex. optimized vs. averaged version) you might edit and run the **plot-all-two-lines-manually.sh** file. This file calls **plot-two-lines.sh** and save the output as pdf in the current directory.

To plot individual files run **plot.sh** or **plot-two-lines.sh**. To plot weights run **plot-weights.sh** script.
