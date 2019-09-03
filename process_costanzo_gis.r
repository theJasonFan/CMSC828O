library(readr)
library(tidyverse)

# read data file
mat <- read_tsv("data/output/Data File S2. Raw genetic interaction datasets: Matrix format/SGA_NxN_clustered.cdt")

amat <- mat %>%
  # extract data rows and columns
  slice(-(1:5)) %>%
  select(starts_with("dma")) %>%
  
  # convert to numeric matrix
  type_convert() %>%
  as.matrix()
  
# turn into unweighted, undirected adjacency matrix  
amat <- 1 * (abs(amat) > 0.2)
  
# extract data about arrays (columns)
coldata <- mat %>%
  slice(1:5) %>%
  select(GID, starts_with("dma")) %>%
  slice(2) %>% select(-1) %>%
  gather(dma, orf, starts_with("dma"))
 
# extract ORF ids for queries (rows)
row_orf <- mat %>%
  slice(-(1:5)) %>%
  pull(ORF)

# match row and column ORF ids
m <- match(row_orf, coldata$orf)
rows_to_use <- !is.na(m)
cols_to_use <- m[rows_to_use]

# subset matrix into ORFs found in both rows and columns
amat <- amat[rows_to_use, cols_to_use]

# set diagonal and missing entries in matrix to 0
diag(amat) <- 0
amat[is.na(amat)] <- 0

# make the adjacency matrix diagonal
amat <- ceiling(0.5 * (amat + t(amat)))

write.table(amat, file='test.tsv', sep='\t', quote=FALSE)