


#library loading
library(tidyverse)
library(ggplot2)

#comparing FPKM expression between breast and normal tissue for BRCA1
data.long %>%
  filter(gene =='BRCA1') %>%
  ggplot(., aes(x = samples, y = FPKM, fill = tissue)) +
  geom_col()

#density plot
data.long%>%
  filter(gene == 'BRCA1') %>%
  ggplot(., aes(x = FPKM, fill = tissue)) +
  geom_density(alpha=0.4)

#boxplots to compare different metastatic statuses
data.long%>%
  filter(gene == 'BRCA1')%>%
  ggplot(., aes(x = metastasis, y = FPKM)) +
  #geom_boxplot()
  geom_violin()


data.long%>%
  filter(gene == 'BRCA1' | gene == 'BRCA2')%>%
  spread(key = gene, value = FPKM) %>%
  ggplot(., aes(x = BRCA1, y = BRCA2, color = tissue)) +
  geom_point() + 
  geom_smooth(method = 'lm', se = FALSE)


genes.of.interest <- c('BRCA1', 'BRCA2', 'TP53', 'ALK', 'MYCN')  

p <- data.long%>%
  filter(gene %in% genes.of.interest)%>%
  ggplot(., aes(x = samples, y = gene, fill = FPKM)) +
  geom_tile() + scale_fill_gradient(low = 'white', high = 'blue')

ggsave(p, filename = 'heatmap_save1.pdf', width = 10, height= 8)



