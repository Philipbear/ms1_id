# This script is to generate pseudo MS2 spectra using xcms and CAMERA

# Load required libraries
library(xcms)
library(CAMERA)
library(BiocParallel)

# Set up parallel processing
register(MulticoreParam(workers = 6))  # Adjust number of workers based on your CPU

# Set your working directory and get file paths
setwd("../data/nist/fullscan_20ev/data")
mzML_files <- list.files(pattern = "*.mzML", full.names = TRUE)

# Create basic phenodata data frame without group information
pd <- data.frame(
  sample_name = basename(mzML_files),
  stringsAsFactors = FALSE
)

# Read the data
raw_data <- readMsExperiment(
  spectraFiles = mzML_files,
  sampleData = pd
)

# Peak detection parameters for DDA data
cwp <- CentWaveParam(
  ppm = 10,                     # Mass accuracy
  peakwidth = c(5, 60),       # Peak width in seconds
  snthresh = 5,               # Signal-to-noise threshold
  prefilter = c(6, 5000),      # Prefilter step
  mzdiff = 0.01,               # Minimum difference in m/z
  noise = 20000,                # Noise threshold
  integrate = 1,               # Integration method
  verboseColumns = TRUE,       # Include additional peak information
  fitgauss = FALSE            # No Gaussian fitting
)

# Perform peak detection
xdata <- findChromPeaks(raw_data, param = cwp)

# Initial peak grouping for retention time correction
pre_pdp <- PeakDensityParam(
  sampleGroups = rep(1, length(mzML_files)),  # All samples in one group
  minFraction = 0.5,
  bw = 30,
  binSize = 0.01
)

# Perform initial peak grouping
xdata <- groupChromPeaks(xdata, param = pre_pdp)

# Retention time correction
pgp <- ObiwarpParam(
  binSize = 0.6,              # Create warping functions in m/z bins of 0.6
  response = 1,               # Default response
  distFun = "cor",           # Correlation distance function
  gapInit = 0.3,             # Gap penalty
  gapExtend = 2.4            # Gap extend penalty
)

# Perform retention time correction
xdata <- adjustRtime(xdata, param = pgp)

# Final peak grouping parameters
pdp <- PeakDensityParam(
  sampleGroups = rep(1, length(mzML_files)),  # All samples in one group
  minFraction = 0.5,
  bw = 5,                     # Bandwidth for peak grouping
  binSize = 0.01              # Mass bin size
)

# Perform final peak grouping
xdata <- groupChromPeaks(xdata, param = pdp)

# Fill in missing peaks
xdata <- fillChromPeaks(xdata, param = ChromPeakAreaParam())

# Annotate peaks using CAMERA
xdata_peaks <- as(xdata, "xcmsSet")
xa <- xsAnnotate(xdata_peaks)
xa <- groupFWHM(xa)
xa <- findIsotopes(xa)
xa <- groupCorr(xa)
# xa <- findAdducts(xa)

# Export peak list with annotations
feature_table <- getPeaklist(xa)
write.csv(annotated_peaks, "feature_table.csv", row.names = FALSE)

######################################
# write pseudo MSMS spectra
######################################
# Create pseudo MS2 spectra from feature table
create_pseudo_ms2_list <- function(feature_table, filled_intensity=2e4) {
  # Get unique pcgroups
  pcgroups <- unique(feature_table$pcgroup)
  
  # Initialize list to store pseudo MS2 spectra
  pseudo_ms2_list <- list()
  
  # Process each pcgroup
  for (group in pcgroups) {
    # Skip NA pcgroup if any
    if (is.na(group)) next
    
    # Get features for this group
    group_features <- feature_table[feature_table$pcgroup == group, ]
    
    # Get all intensity columns (assuming they start with "X" followed by numbers)
    intensity_cols <- grep("^X\\d+", names(group_features), value = TRUE)
    
    if (length(intensity_cols) == 0) {
      warning(paste("No intensity columns found for pcgroup", group))
      next
    }
    
    # Replace NA values with filled_intensity in intensity columns
    intensity_matrix <- group_features[, intensity_cols, drop = FALSE]
    intensity_matrix[is.na(intensity_matrix)] <- filled_intensity
    
    # Calculate sum of intensities for each sample
    sample_sums <- colSums(intensity_matrix)
    
    # Find the sample with maximum total intensity
    best_sample <- names(sample_sums)[which.max(sample_sums)]
    
    # Get mz values and intensities from best sample
    mz_values <- group_features$mz
    intensities <- intensity_matrix[, best_sample]
    
    # Calculate mean retention time
    mean_rt <- mean(group_features$rt, na.rm = TRUE)
    
    # Create spectrum object
    spectrum <- list(
      mzs = mz_values,
      intensities = intensities,
      rt = mean_rt
    )
    
    # Add to list
    pseudo_ms2_list[[length(pseudo_ms2_list) + 1]] <- spectrum
  }
  
  return(pseudo_ms2_list)
}

write_pseudoms2_to_mgf <- function(pseudoms2_ls, save_dir) {
  # Create the full file path
  mgf_path <- file.path(save_dir, "pseudo_ms2.mgf")

  # Open file connection for writing
  con <- file(mgf_path, "w")

  # Write spectra
  idx <- 1
  cnt <- 0
  for (spec in pseudoms2_ls) {
    # Write header information
    writeLines("BEGIN IONS", con)
    writeLines("PEPMASS=0", con)
    writeLines(paste0("SCANS=", idx), con)
    writeLines(paste0("RTINSECONDS=", spec$rt), con)

    # Convert mz and intensity arrays to vectors if they aren't already
    mz_vec <- as.numeric(spec$mzs)
    intensity_vec <- as.numeric(spec$intensities)

    # Sort by mz
    sort_idx <- order(mz_vec)
    mz_vec <- mz_vec[sort_idx]
    intensity_vec <- intensity_vec[sort_idx]

    # Write peaks
    for (i in seq_along(mz_vec)) {
      writeLines(sprintf("%.5f %.0f", mz_vec[i], intensity_vec[i]), con)
    }

    # Write end of spectrum
    writeLines("END IONS\n", con)

    idx <- idx + 1

    if (length(mz_vec) >= 3) {
      cnt <- cnt + 1
    }
  }

  # Print the number of spectra written
  message(paste0("Wrote ", cnt, " spectra with at least 3 peaks, to ", mgf_path))

  # Close file connection
  close(con)
}

# Create pseudo MS2 list
pseudo_ms2_list <- create_pseudo_ms2_list(feature_table)

# Write to MGF file
output_dir <- "."
write_pseudoms2_to_mgf(pseudo_ms2_list, output_dir)
