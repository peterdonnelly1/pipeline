
library(biomaRt)

ensembl=useMart( host="https://sep2019.archive.ensembl.org", "ENSEMBL_MART_ENSEMBL")

ensembl = useDataset("hsapiens_gene_ensembl", mart=ensembl)

filters = listFilters(ensembl)

genes <- getBM ( attributes = c("ensembl_gene_id", "chromosome_name","transcript_biotype"), 
                          filters = c("transcript_biotype","chromosome_name"),
                          values = list("protein_coding",c(1:22)), 
                          mart = ensembl,
                          useCache = FALSE )

genes["external_gene_name"]
genes["transcript_biotype"]

ensembl_id = genes[2]

write.table( ensembl_id ,'just_genes.csv',  sep=",", row.names = FALSE,  col.names = FALSE, append = FALSE)
