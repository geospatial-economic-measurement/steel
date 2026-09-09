
library(ggplot2)
library(dplyr)
library(readxl)
library(lubridate)

start_date_bs <- as.Date("2010-01-01")  
end_date_bs <- as.Date("2022-12-01")   

start_date_cisa <- as.Date("2015-01-01") 
end_date_cisa <- as.Date("2020-12-01")    

start_date_a <- as.Date("2020-04-01")  
end_date_a <- as.Date("2022-12-01")    

default_start_date <- min(start_date_bs, start_date_cisa, start_date_a)
default_end_date <- max(end_date_bs, end_date_cisa, end_date_a)
full_dates <- data.frame(Date = seq.Date(from = default_start_date, to = default_end_date, by = "month"))

data_bs <- read_excel("/Replication/Data/Steel_NBS.xlsx")
data_cisa <- read_excel("/Replication/Data/Steel_CISA.xlsx")
data_a <- read_excel("/Replication/Data/CISA_largemedium_plant.xlsx")

data_bs$Date <- as.Date(data_bs$Date, format = "%Y-%m-%d")
data_cisa$Date <- as.Date(data_cisa$Date, format = "%Y-%m-%d")
data_a$Date <- as.Date(data_a$Date, format = "%Y-%m-%d")

data_bs <- full_join(full_dates, data_bs, by = "Date") %>%
  filter(Date >= start_date_bs & Date <= end_date_bs) %>%
  mutate(Production_Line = "National Bureau of Statistics of China", Production = Production / 1000) 

data_cisa <- full_join(full_dates, data_cisa, by = "Date") %>%
  filter(Date >= start_date_cisa & Date <= end_date_cisa) %>%
  mutate(Production_Line = "China Iron and Steel Association", Production = (Production * 1.01) / 1000)  

data_a <- full_join(full_dates, data_a, by = "Date") %>%
  filter(Date >= start_date_a & Date <= end_date_a) %>%
  mutate(Production_Line = "Large and Medium-sized Steel Plants (CISA)", Production = Production / 1000)  

data_combined <- bind_rows(data_bs, data_cisa, data_a)

ggplot(data_combined, aes(x = Date, y = Production, color = Production_Line)) +
  geom_line(size = 1) + 
  scale_y_continuous(
    name = "Production (thousand tons)", 
    sec.axis = sec_axis(~./1.01, name = "CISA Production (thousand tons)")  
  ) +
  labs(
    title = "Production Data Comparison by Time Scale",
    x = "Date", y = "Production (thousand tons)", color = "Production Line"
  ) +
  theme_minimal() +
  scale_color_manual(values = c(
    "National Bureau of Statistics of China" = "blue",
    "China Iron and Steel Association" = "red",
    "Large and Medium-sized Steel Plants (CISA)" = "skyblue"
  )) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(0.2, "cm"),
    legend.position = c(0.05, 0.95),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "white", color = "black"),
    legend.key.size = unit(0.8, "cm")
  )

